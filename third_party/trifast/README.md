# trifast 0.1.13 (MIT) - triangle attention, Triton, forward + backward

Already a backend of E5 (`trifast`, `benchmarks/paper/E5_other_kernels/externals.py`). The three running measurement campaigns import it from `~/paper-ext` (`PAPER_EXT_PATH`);
that install is NOT touched. This directory records the same pinned version (wheel sha256 in UPSTREAM.json) with an idempotent `fetch.sh` (`DEST=<dir> ./fetch.sh`; platformdirs 4.14.8 is
the only dependency and is pure Python). Layout of q / k / v `[B, H, N, N, d]`, bias `[B, H, N, N]`, mask `[B, N, N]` (True = masked); fp32 / bf16 / fp16.
