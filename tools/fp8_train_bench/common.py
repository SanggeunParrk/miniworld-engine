import pin  # noqa: F401
import torch

from foldjet_engine.measure import config as C
from foldjet_engine.modules.token_dit import TokenDiTProtenix


def block(seed=0, dp=128):
    m = TokenDiTProtenix(d_single=768, d_cond=384, d_pair=dp, n_head=16, n=2).cuda().float()
    C.randomize(m, torch.Generator(device="cuda").manual_seed(seed))
    m = C.cast(m, "master")
    return m.train().set_backend("cuda")


def inputs(m, A, L, seed=1, shared=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return m.make_inputs(
        length=L,
        samples=A,
        depth=None,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
        stream="token_single_shared_cond" if shared else None,
    )


def run(m, inp, dout):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(**inp)
    out.backward(dout)
    return out
