#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""CPU regression suite. Runs without a GPU; GPU parity is a separate target.

Usage: python3 tests/run_tests.py --binary ./kangaroo
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from secp256k1 import G, add, mul, pubkey_hex

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, timeout=90, expect=0):
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    if expect is not None and p.returncode != expect:
        raise AssertionError(f"command {cmd} exited {p.returncode}, expected {expect}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p


def write_fixture(path, bits, offset):
    start = 1 << (bits - 1)
    end = (1 << bits) - 1
    secret = start + offset
    assert start <= secret <= end
    path.write_text(f"{start:X}\n{end:X}\n{pubkey_hex(secret)}\n")
    return secret, start, end


def extract_priv(text):
    m = re.search(r"Priv:\s*0x([0-9a-fA-F]+)", text)
    return int(m.group(1), 16) if m else None


def test_oracle():
    # Independent known-answer check for the test oracle itself.
    two_g = add(G, G)
    assert mul(2) == two_g
    assert pubkey_hex(1) == "0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
    print("ok: independent secp256k1 oracle")


def test_batch_policy(cxx):
    with tempfile.TemporaryDirectory() as td:
        exe = Path(td) / "test_batch_policy"
        run([cxx, "-std=c++11", "-O2", "-Wall", "-Wextra", "-I.", str(ROOT / "tests/test_batch_policy.cpp"), "-o", str(exe)])
        p = run([str(exe)])
        assert "batch policy: OK" in p.stdout
    print("ok: GPU batch policy unit tests")


def test_arithmetic(binary):
    p = run([str(binary), "-check"], timeout=120)
    assert "Results OK" in p.stdout
    print("ok: arithmetic self-check")


def test_solve(binary, tmp, bits, offset):
    fixture = tmp / f"puzzle{bits}.txt"
    secret, start, end = write_fixture(fixture, bits, offset)
    out = tmp / f"result{bits}.txt"
    p = run([str(binary), "-t", "2", "-d", "0", "-m", "80", "-o", str(out), str(fixture)], timeout=120)
    text = p.stdout + p.stderr + (out.read_text() if out.exists() else "")
    found = extract_priv(text)
    assert found == secret, f"{bits}-bit: expected {secret:X}, got {found}\n{text}"
    assert start <= found <= end
    assert pubkey_hex(found) == pubkey_hex(secret)
    print(f"ok: {bits}-bit interval solve and public-key verification")


def test_cli_validation(binary):
    for value in ("0", "4097", "-1"):
        p = run([str(binary), "-gpuBatch", value], expect=2)
        assert "Invalid -gpuBatch" in p.stderr
    print("ok: -gpuBatch validation")


def test_save_resume(binary, tmp):
    # A 56-bit interval is far too large to solve in the short window; this
    # exercises checkpoint creation, reload, and integrity checking instead.
    fixture = tmp / "checkpoint.txt"
    write_fixture(fixture, 56, 0x1234567)
    work = tmp / "state.work"
    cmd = [str(binary), "-t", "2", "-d", "8", "-w", str(work), "-ws", "-wi", "1", str(fixture)]
    try:
        subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=4)
    except subprocess.TimeoutExpired:
        pass
    assert work.exists() and work.stat().st_size > 0, "checkpoint was not written"
    run([str(binary), "-winfo", str(work)], timeout=30)
    resumed = tmp / "resumed.work"
    cmd2 = [str(binary), "-t", "2", "-i", str(work), "-w", str(resumed), "-ws", "-wi", "1"]
    try:
        subprocess.run(cmd2, cwd=ROOT, text=True, capture_output=True, timeout=4)
    except subprocess.TimeoutExpired:
        pass
    assert resumed.exists() and resumed.stat().st_size > 0, "resumed checkpoint was not written"
    run([str(binary), "-wcheck", str(resumed)], timeout=60)
    print("ok: checkpoint save/resume/integrity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(ROOT / "kangaroo"))
    ap.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    ap.add_argument("--skip-checkpoint", action="store_true")
    args = ap.parse_args()
    binary = Path(args.binary).resolve()
    assert binary.exists(), binary

    test_oracle()
    test_batch_policy(args.cxx)
    test_arithmetic(binary)
    test_cli_validation(binary)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_solve(binary, tmp, 24, 0x34567)
        test_solve(binary, tmp, 28, 0x1234567 % (1 << 27))
        test_solve(binary, tmp, 32, 0x12345678)
        if not args.skip_checkpoint:
            test_save_resume(binary, tmp)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
