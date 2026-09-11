# Historical caches with an unknown GPU identity

These nine files were stored under `autotune/data/<op>/cpu.json`. They contain GPU kernel
timings, but the filename does not identify a GPU and their origin cannot be established
from that key. They must not participate in GPU cache freshness, coverage, or winning-config
statistics. The original bytes are preserved here; `manifest.json` records their old paths
and SHA-256 hashes. No timings were reassigned to a guessed GPU.

The existing `dev merge` command already rejects a missing CUDA device unless an explicit
GPU key is supplied. This archive removes the historical artifacts from active cache data.
