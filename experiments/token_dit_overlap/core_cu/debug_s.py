"""Score tile after bias, first key block, against torch."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core_cu import attn_core  # noqa: E402

H, DS, NB, dev, bf = 16, 768, 24, "cuda", torch.bfloat16
D, S, L = DS // H, 1, 64
torch.manual_seed(0)
base = (torch.randn(S * L, 4 * DS, device=dev) * DS ** -0.5).to(bf)
bias = (torch.randn(NB * H, L, L, device=dev) * 0.3).to(bf)
dbg = torch.zeros(64, 64, device=dev)
got = base.clone()
attn_core(got, bias, 0, S, H, dbg)
torch.cuda.synchronize()
q = base[:64, 0:48].float()
k = base[:64, 768:768 + 48].float()
ref = q @ k.t() + bias[0, :64, :64].float()
print("score tile rel", float((dbg - ref).norm() / ref.norm()))
print("without bias  ", float((dbg - q @ k.t()).norm() / (q @ k.t()).norm()))
print("dbg[0,:6]", dbg[0, :6].tolist())
print("ref[0,:6]", ref[0, :6].tolist())
