# DeepSpeed 0.20.0 (Apache-2.0) - DeepSpeed4Science Evoformer attention (`ds4sci`)

Already a backend of E5 (`ds4sci`, `externals.py`; bf16 / fp16, CUTLASS JIT build, q scaled by d^-0.5, key mask and pair bias as the two biases). The running campaigns import it from `~/paper-ext`;
that install is not touched. `fetch.sh` reproduces the pinned install (`DEST=<dir> PIP="<card python> -m pip" ./fetch.sh`; needs a python of the card: run on a compute node, not the login node).
