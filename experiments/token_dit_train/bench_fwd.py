"""Forward kernel latency (do_bench, L2 evicted) against the unit floors at A=48."""
import os
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "token_dit_overlap"))
from bench import us  # noqa: E402
from token_dit_train.ref import make  # noqa: E402
from token_dit_train.fwd import prep, attn_fwd  # noqa: E402

TAG = (f"fwd{os.environ.get('TDT_FWD', '2')} sch{os.environ.get('TDT_SCH', 'auto')} "
       f"{os.environ.get('ATTN_FWD_DEFS', '')}{os.environ.get('ATTN_FWD2_DEFS', '')}")
TC, CLK, SMS = 757e12, 1.755e9, 132
A, H = 48, 16
for L in (384, 768):
    q, k, v, bias, _ = make(A, L)
    args = prep(q, k, v, bias, None)
    t = us(lambda: attn_fwd(*args, A, L))
    pairs = A * H * L * L
    f_tc = pairs * 192 / TC * 1e6
    f_mufu = pairs / (16 * SMS * CLK) * 1e6
    print(f"bench: {TAG} L{L} A{A}: {t:8.1f} us  {pairs * 192 / t / 1e6:6.1f} TF/s | "
          f"floors: tensor {f_tc:6.1f}  mufu {f_mufu:6.1f}  -> {max(f_tc, f_mufu) / t * 100:5.1f} % of the max", flush=True)
