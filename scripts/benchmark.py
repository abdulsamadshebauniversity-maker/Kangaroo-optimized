#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Reproducible throughput benchmark harness (CPU tested; GPU path requires hardware).

Examples:
  # CPU smoke benchmark
  python3 scripts/benchmark.py --binary ./kangaroo --threads 2 --dp 8 --duration 6

  # GPU A/B on the target machine (build baseline and candidate first)
  python3 scripts/benchmark.py --baseline ./kangaroo-baseline --candidate ./kangaroo \
      --gpu --threads 0 --gpu-id 0 --grid 56,256 --dp 16 --duration 30 --repeats 5 \
      --json gpu-ab.json

The search is intentionally run on an interval too large to solve during the
window. We compare sustained reported throughput, not lucky time-to-solution.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from secp256k1 import pubkey_hex  # noqa: E402

RATE = re.compile(r"\[([0-9.]+) MK/s\]\[GPU ([0-9.]+) MK/s\]")
LOST = re.compile(r"Warning, (\d+) items lost")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_info():
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return p.stdout.strip() if p.returncode == 0 else None
    except Exception:
        return None


def make_fixture(td, bits):
    start = 1 << (bits - 1)
    end = (1 << bits) - 1
    secret = start + (0x123456789ABCDEF % (end - start + 1))
    path = Path(td) / f"bench{bits}.txt"
    path.write_text(f"{start:X}\n{end:X}\n{pubkey_hex(secret)}\n")
    return path


def run_once(binary, args, fixture, duration, is_baseline):
    cmd = [str(binary), "-t", str(args.threads), "-d", str(args.dp), "-m", "1000000", str(fixture)]
    if args.gpu:
        cmd += ["-gpu", "-gpuId", str(args.gpu_id)]
        if args.grid:
            cmd += ["-g", args.grid]
    if args.gpu_batch is not None and not is_baseline:
        cmd += ["-gpuBatch", str(args.gpu_batch)]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id) if args.gpu else "")
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True, env=env)
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=duration)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            out, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate(timeout=5)
    elapsed = time.monotonic() - t0
    samples = [(float(a), float(b)) for a, b in RATE.findall(out)]
    # Discard the first smoothing window as warm-up when enough samples exist.
    use = samples[1:] if len(samples) > 2 else samples
    rates = [(b if args.gpu else a) for a, b in use]
    lost = LOST.search(out)
    return {
        "command": cmd,
        "elapsed_s": round(elapsed, 3),
        "timed_out": timed_out,
        "returncode": proc.returncode,
        "samples": len(rates),
        "median_MKps": round(statistics.median(rates), 4) if rates else None,
        "min_MKps": round(min(rates), 4) if rates else None,
        "max_MKps": round(max(rates), 4) if rates else None,
        "lost_items_warning": int(lost.group(1)) if lost else 0,
        "log_tail": out[-2000:],
    }


def summarize(runs):
    vals = [r["median_MKps"] for r in runs if r["median_MKps"] is not None]
    return {
        "runs": runs,
        "median_of_run_medians_MKps": round(statistics.median(vals), 4) if vals else None,
        "stdev_MKps": round(statistics.stdev(vals), 4) if len(vals) > 1 else None,
        "any_lost_items": any(r["lost_items_warning"] for r in runs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", help="single binary to benchmark")
    ap.add_argument("--baseline")
    ap.add_argument("--candidate")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--grid", help="e.g. 56,256")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--dp", type=int, default=12)
    ap.add_argument("--gpu-batch", type=int, help="candidate-only -gpuBatch value")
    ap.add_argument("--bits", type=int, default=64)
    ap.add_argument("--input", help="existing input file; otherwise a synthetic fixture is generated")
    ap.add_argument("--duration", type=float, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json", help="write machine-readable results")
    args = ap.parse_args()

    targets = []
    if args.baseline or args.candidate:
        if not (args.baseline and args.candidate):
            ap.error("--baseline and --candidate must be used together")
        targets = [("baseline", Path(args.baseline).resolve(), True),
                   ("candidate", Path(args.candidate).resolve(), False)]
    elif args.binary:
        targets = [("binary", Path(args.binary).resolve(), False)]
    else:
        ap.error("provide --binary or both --baseline and --candidate")
    for _, path, _ in targets:
        if not path.exists():
            ap.error(f"binary not found: {path}")

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu_info": gpu_info(),
        "settings": vars(args),
        "binaries": {name: {"path": str(path), "sha256": sha256(path)} for name, path, _ in targets},
        "results": {},
    }
    with tempfile.TemporaryDirectory() as td:
        fixture = Path(args.input).resolve() if args.input else make_fixture(td, args.bits)
        for name, path, is_baseline in targets:
            runs = [run_once(path, args, fixture, args.duration, is_baseline) for _ in range(args.repeats)]
            report["results"][name] = summarize(runs)
            s = report["results"][name]
            print(f"{name}: median {s['median_of_run_medians_MKps']} MK/s, "
                  f"stdev {s['stdev_MKps']}, lost_items={s['any_lost_items']}")
    if len(targets) == 2:
        b = report["results"]["baseline"]["median_of_run_medians_MKps"]
        c = report["results"]["candidate"]["median_of_run_medians_MKps"]
        if b and c:
            report["candidate_speedup"] = round(c / b, 4)
            print(f"candidate/baseline throughput ratio: {report['candidate_speedup']}x")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
