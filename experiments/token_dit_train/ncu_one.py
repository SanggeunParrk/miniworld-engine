"""One call of the chosen forward/dq kernel at A=48 for ncu (after a warm-up call)."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from token_dit_train.ref import make  # noqa: E402
from token_dit_train.fwd import prep, attn_fwd  # noqa: E402
L = int(sys.argv[1])
q, k, v, bias, _ = make(48, L)
args = prep(q, k, v, bias, None)
attn_fwd(*args, 48, L); torch.cuda.synchronize()
attn_fwd(*args, 48, L); torch.cuda.synchronize()
