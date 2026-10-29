# Security and responsible use

## Authorized use only
This tool performs interval discrete-log search. Use it only on keys, puzzles, or
systems you own or have explicit written permission to test. Do not use it to
recover third-party private keys.

## Reporting
Open a GitHub issue for non-sensitive correctness/build problems. For sensitive
reports, contact the repository owner through GitHub without publishing secrets.

## Operational hygiene
- Do not commit private keys, tokens, or production work files.
- Treat work files as sensitive state.
- The self-hosted GPU workflow is `workflow_dispatch` only; do not enable it for
  untrusted pull requests on a public repository.
- Revoke the GitHub CLI device authorization after publishing if it is no longer
  needed.
