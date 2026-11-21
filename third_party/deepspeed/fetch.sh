#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/../_attn_common.sh"
DEST=${DEST:-$HERE/upstream}
# deepspeed 0.20.0 (sdist sha256 67195f94b2bc63350f15f2da9126b489122261d2c858f34120f4aa33910bca7d): only its op sources are used (DS4Sci_EvoformerAttention JIT-builds with CUTLASS_PATH / nvcc / ninja).
# Same recipe as ~/paper-ext: pip needs a python of the card (PIP, default "python -m pip"); run on a compute node / card host, never the login node.
PIP=${PIP:-python -m pip}
printf "deepspeed==0.20.0 --hash=sha256:67195f94b2bc63350f15f2da9126b489122261d2c858f34120f4aa33910bca7d\n" > "${TMPDIR:-/tmp}/ds_req_$$.txt"
$PIP install --target "$DEST" --no-deps --require-hashes -r "${TMPDIR:-/tmp}/ds_req_$$.txt"
$PIP install --target "$DEST" --no-deps hjson==3.1.0 py-cpuinfo==9.0.0 psutil==7.2.2 ninja==1.13.2 msgpack==1.2.4
