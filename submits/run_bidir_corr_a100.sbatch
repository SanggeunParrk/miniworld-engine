#!/bin/bash
#SBATCH --job-name=mwk_bidir_corr
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:25:00
#SBATCH --output=benchmarks/bidir_corr_a100_%j.out
set -uo pipefail
cd /home/psk6950/miniworld-engine
pixi run --frozen bash -c 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python - <<PY
import torch
from miniworld_engine.modules import BidirectionalTriangleMultiplication as Bidir, ImplementationType as IT
dev,bf16="cuda",torch.bfloat16
def cos(a,b):
    a,b=a.float().flatten(),b.float().flatten(); return torch.nn.functional.cosine_similarity(a,b,dim=0).item()
for L in (384, 768):
    torch.manual_seed(0)
    mw=Bidir(512, implementation=IT.MINIWORLD).to(dev).to(bf16)
    # lift zero-init out/gate layers off zero so the output is non-degenerate (cf. test_numerical._sync_weights)
    g=torch.Generator().manual_seed(0)
    for p in mw.parameters():
        if p.detach().abs().max().item() < 1e-6:
            p.data.copy_((torch.randn(p.shape,generator=g,dtype=torch.float32)*0.02).to(p.dtype))
    ref=Bidir(512, implementation=IT.PYTORCH).to(dev).to(bf16); ref.load_state_dict(mw.state_dict())
    pair=torch.randn(1,L,L,512,device=dev,dtype=bf16); mask=torch.ones(1,L,dtype=torch.bool,device=dev)
    xa=pair.clone().requires_grad_(True); xb=pair.clone().requires_grad_(True)
    ya=mw(xa,mask); yb=ref(xb,mask)
    torch.manual_seed(1); cg=torch.randn_like(ya.float())
    ya.float().mul(cg).sum().backward(); yb.float().mul(cg).sum().backward()
    fc,gc=cos(ya,yb),cos(xa.grad,xb.grad)
    print(f"  d=512 L={L}: fwd_cos={fc:.5f} grad_cos={gc:.5f}")
    assert fc>0.99 and gc>0.99, f"bidir d=512 L={L} correctness fail"
print("BIDIR CORR OK")
PY'
echo "[exit=$?]"
