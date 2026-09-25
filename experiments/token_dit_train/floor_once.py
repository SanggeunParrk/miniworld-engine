import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from token_dit_train.ref import make  # noqa
from token_dit_train.fwd import prep, attn_fwd  # noqa
for L, A in ((384, 1), (384, 4), (768, 48)):
    q, k, v, bias, _ = make(A, L)
    args = prep(q, k, v, bias, None)
    attn_fwd(*args, A, L); torch.cuda.synchronize()
    print("floor ok", L, A, flush=True)
