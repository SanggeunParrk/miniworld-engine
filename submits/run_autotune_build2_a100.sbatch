#!/bin/bash
# (1) Correctness: bidir d=512 L=768 fwd+bwd (the int32-offset crash shape) vs the pytorch
#     reference after the int64-offset fix. (2) Rebuild the full shipped cache (now completes
#     d=512 L>=768). (3) Copy runtime cache -> committed package data.
#SBATCH --job-name=mwk_autotune_build2
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=benchmarks/autotune_build2_a100_%j.out

set -uo pipefail
cd /home/psk6950/miniworld-engine
echo "host=$(hostname) date=$(date)"

echo "### (1) int64-offset correctness: bidir d=512 L=768 fwd+bwd vs reference ###"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python - <<PY
import torch
from miniworld_engine.modules import BidirectionalTriangleMultiplication as Bidir, ImplementationType as IT
torch.manual_seed(0)
dev, bf16 = "cuda", torch.bfloat16
d, L = 512, 768
mw  = Bidir(d, implementation=IT.MINIWORLD).to(dev).to(bf16)
ref = Bidir(d, implementation=IT.PYTORCH).to(dev).to(bf16)
ref.load_state_dict(mw.state_dict())
def cos(a,b):
    a,b=a.float().flatten(),b.float().flatten()
    return torch.nn.functional.cosine_similarity(a,b,dim=0).item()
pair = torch.randn(1, L, L, d, device=dev, dtype=bf16)
mask = torch.ones(1, L, dtype=torch.bool, device=dev)
xa = pair.clone().requires_grad_(True); xb = pair.clone().requires_grad_(True)
ya = mw(xa, mask); yb = ref(xb, mask)
torch.manual_seed(1); g = torch.randn_like(ya.float())
ya.float().mul(g).sum().backward(); yb.float().mul(g).sum().backward()
fc, gc = cos(ya, yb), cos(xa.grad, xb.grad)
print(f"  d={d} L={L}: fwd_cos={fc:.5f} grad_cos={gc:.5f} (no illegal-access)")
assert fc > 0.99 and gc > 0.99, "int64-offset fix broke correctness"
print("  int64 fix OK")
PY'
echo "[correctness exit=$?]"

echo "### (2) rebuild full shipped cache ###"
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python -m miniworld_engine.autotune.build --op all'
echo "[build exit=$?]"

RT="${MINIWORLD_ENGINE_CACHE_DIR:-$HOME/.cache/miniworld_engine}/autotune"
DST="src/miniworld_engine/autotune/data"
echo "### (3) copy $RT -> $DST ###"
[ -d "$RT" ] && cp -rv "$RT/." "$DST/" && find "$DST" -name '*.json' -print
echo "AUTOTUNE BUILD2 DONE"
