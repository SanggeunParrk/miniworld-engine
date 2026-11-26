"""FP8 (e4m3) row kernels of the B200 local atom DiT block (agent E, 2026-11-19; opt-in, OFF by default, inference only).

The H100 route (``sm90_atom/f8.py``: ``pre90_f8`` / ``post90_f8``) on tcgen05: ``pre_fwd.cu -DF8=1`` (this file's ``pre_fwd`` / ``pre_fwd_kv``) -- x1 (and the
K / V AdaLN's xkv) leave the AdaLN threads as e4m3 with a dynamic per-row scale, the projections run on ``kind::f8f6f4`` with the weights (e4m3,
static per-output-channel scale) in TMEM, the epilogue scales the fp32 accumulator by sx[row] sw[channel].  Nothing here changes a default
route: ``ENABLED`` is empty unless ``FOLDJET_ENGINE_ATOMDIT_FP8_B200`` / ``enable()`` names a stage.
"""

from __future__ import annotations

import foldjet_engine._envcompat  # noqa: F401  (old MINIWORLD_* environment names)

import os

import torch

from foldjet_engine.kernels import _capture
from foldjet_engine.kernels._device import num_sms

from . import (
    DM,
    KERNELS,
    KERNELS_ESM,
    KERNELS_VAR,
    _dev,
    _load_atom_kernel,
    _map,
    _pre_maps,
)

#: stages on e4m3: "pre" (the q / k / v / gate projections), "post" (the out projection / SwiGLU / squeeze)
ENABLED: frozenset = frozenset()
#: row gate (atoms x samples = rows M); 0 = every size
MIN_ROWS = 0
STATS = {"pre": 0, "pre_declined": 0, "post": 0, "post_declined": 0}


def _parse(v: str) -> frozenset:
    v = (v or "").strip().lower()
    if v in ("", "0", "off", "false", "no"):
        return frozenset()
    if v in ("1", "all", "on", "true", "yes"):
        return frozenset({"pre", "post"})
    return frozenset(s for s in v.replace("+", ",").split(",") if s in ("pre", "post"))


def enable(v) -> frozenset:
    """Select the stages (``True`` / ``"all"``, ``"pre"``, ``"post"``, ``False``); returns the previous set."""
    global ENABLED
    old = ENABLED
    ENABLED = _parse("all" if v is True else "" if v is False else str(v))
    return old


ENABLED = _parse(os.environ.get("FOLDJET_ENGINE_ATOMDIT_FP8_B200", ""))
MIN_ROWS = int(os.environ.get("FOLDJET_ENGINE_ATOMDIT_FP8_B200_MIN_ROWS", "0"))
#: row ceiling (rows M = A F N): the block loses 1.5-6 % of its time above ~2048 atoms x 5 samples (docs/b200/fp8.md, "Agent E"); 10240 = 2048 atoms x 5
MAX_ROWS = int(os.environ.get("FOLDJET_ENGINE_ATOMDIT_FP8_B200_MAX_ROWS", "10240"))


def serves(stage: str, rows: int, save: bool) -> bool:
    """The route's gate: stage selected, inference (no saved activations), MIN_ROWS <= rows <= MAX_ROWS."""
    ok = stage in ENABLED and not save and MIN_ROWS <= rows <= MAX_ROWS
    if not ok and stage in ENABLED:
        STATS[stage + "_declined"] += 1
    return ok


# --------------------------------------------------------------------------------------------------- weights
def _kperm(device) -> torch.Tensor:
    """Thread ak of a row owns channels 64 kb + 8 ak + i (kb 0 / 1, i 0..7) = k slots 16 ak + 8 kb + i of the e4m3 row."""
    return torch.tensor([64 * kb + 8 * ak + i for ak in range(8) for kb in range(2) for i in range(8)], device=device)


def _pack8(w: torch.Tensor):
    """[N, 128] bf16 / fp32 -> (e4m3 [N, 128] as uint8 with the K axis permuted, per-output-channel scale fp32 [N] = max / 448)."""
    wf = w.float()[:, _kperm(w.device)]
    sc = (wf.abs().amax(1) / 448.0).clamp_min(1e-30)
    return (wf / sc[:, None]).to(torch.float8_e4m3fn).view(torch.uint8).contiguous(), sc.float().contiguous()


def pre_pack8(wpack: torch.Tensor):
    """The e4m3 form of ``pre_params``' W [512, 128] = [Wq; Wk; Wv; Wg]: cached while ``wpack`` lives (a capture reads the eager entry)."""
    return _capture.param_forms(("b200_atomdit_f8_pre",), [wpack], lambda: _pack8(wpack), capture_reads_eager=True)


# --------------------------------------------------------------------------------------------------- launchers
def _kernel(name: str, d: int):
    table = KERNELS_ESM if name in KERNELS_ESM else KERNELS_VAR if name in KERNELS_VAR else KERNELS
    stem, func, smem, pdl, *defs = table[name]
    return _load_atom_kernel(stem + "_f8", func, d, smem, pdl, (*(defs[0] if defs else ()), *(("F8ABL=1",) if os.environ.get("MW_F8_ABL") == "1" else ())))


def pre_fwd(a, mod, W, bq, *, eps=1e-5, ada1=False, eb=None, mkv=None, bs=None, parallel=False, rms=False, raw=False, rope=None, kq=None, kq_eps=1e-5):
    """``b200_atom.pre_fwd`` / ``pre_fwd_kv`` / ``pre_fwd_kq`` (inference) on e4m3 projections: -> Q, K, V, G [M, 128] bf16.
    ``mkv`` / ``bs`` given: the cross-attention build (K / V from xkv; ``parallel`` / ``rms`` / ``raw`` / ``rope`` as ``pre_fwd_kv``);
    ``kq``: RoseTTAFold3's flat q / k LayerNorm (``pre_fwd_kq``)."""
    M = a.shape[0]
    Q, K, V, G = (torch.empty(M, DM, device=a.device, dtype=torch.bfloat16) for _ in range(4))
    we, sw = pre_pack8(W)
    maps = list(_pre_maps(a, W, mod, Q, K, V, G, G, mkv, G))
    maps[1] = _map(we, [DM, 4 * DM], DM, [64, 128], swizzle=64, dtype="u8")
    ntile = M // 32
    d = _dev(a)
    cross = mkv is not None
    extra = ()
    if cross:
        name = (("pre_rms_raw_rope" if rope is not None else "pre_rms_raw") if raw else "pre_rms") if rms else "prexp" if parallel else "prex"
        second = bs.float().contiguous()
        if rope is not None:
            wq, wk, cs, sn, qeps = rope
            extra = (wq.contiguous(), wk.contiguous(), cs, sn, int(cs.shape[0]), float(qeps))
    else:
        name = ("pre_kq" if kq is not None else "pre") + ("_a1" if ada1 else "")
        second = bq
        if kq is not None:
            extra = (kq, float(kq_eps), torch.empty(0, 2 * DM, device=a.device, dtype=torch.bfloat16))
    if eb is not None:
        name += "_eb"
    _kernel(name, d)(
        (min(num_sms(d), ntile), 1, 1),
        (384, 1, 1),
        *maps,
        bq,
        second,
        int(ntile),
        float(eps),
        0,
        int(mod.shape[0]),
        *extra,
        *((a,) if (ada1 and not cross) else ()),
        *(eb if eb is not None else ()),
        sw,
    )
    STATS["pre"] += 1
    return Q, K, V, G


# --------------------------------------------------------------------------------------------------- post_fwd
def post_pack8(wo: torch.Tensor, wu: torch.Tensor, wup: torch.Tensor | None = None):
    """The e4m3 forms of ``post_params``' Wo [128, 128] and Wu = [Wa; Wb] [512, 128]: (Wo8, SWO [128], Wu8, SWU [512])."""

    def build():
        return (*_pack8(wo), *_pack8(wu), *(_pack8(wup) if wup is not None else ()))

    return _capture.param_forms(("b200_atomdit_f8_post",), [wo, wu] + ([wup] if wup is not None else []), build, capture_reads_eager=True)


def _post_kernel(name: str, d: int):
    table = KERNELS_ESM if name in KERNELS_ESM else KERNELS_VAR if name in KERNELS_VAR else KERNELS
    stem, func, smem, pdl, *defs = table[name]
    return _load_atom_kernel(stem + "_f8", func, d, smem, pdl, (*(defs[0] if defs else ()), "BOX3=1", *(("SQABL=1",) if os.environ.get("MW_F8_SQABL") == "1" else ()), *(("F8ABL=1",) if os.environ.get("MW_F8_ABL") == "1" else ())))


def post_fwd(a, o, g, mod, Wo, Wu, Wd, *, eps=1e-5, rms=False, raw=False, pars=False, ada1=False, wup=None):
    """``b200_atom.post_fwd`` (inference, no up gate) with the out projection and the SwiGLU expansion on e4m3 -> the block output [M, 128] bf16."""
    M = a.shape[0]
    out = torch.empty(M, DM, device=a.device, dtype=torch.bfloat16)
    wo8, swo, wu8, swu, *upk = post_pack8(Wo, Wu, wup)

    def r3(t, cols=DM):
        return _map(t, [64, t.shape[0], cols // 64], [t.stride(0) * 2, 128], [64, 16, 2 if cols == DM else 8])

    maps = (
        r3(a),
        r3(o),
        r3(g),
        r3(mod, 6 * DM),
        _map(wo8, [DM, DM], DM, [DM, DM], swizzle=128, dtype="u8"),
        _map(wu8, [64, 4 * DM, 2], [DM, 64], [64, 128, 2], swizzle=64, dtype="u8"),
        _map(Wd, [32, DM, 8], [4 * DM, 64], [32, 128, 4], swizzle=64),
        r3(out),
        *(r3(out) for _ in range(4)),
    )
    ntile = M // 16
    d = _dev(a)
    name = ("post_rms_raw" if raw else "post_rms") if rms else "post_up" if wup is not None else "post_pars" if pars else "post"
    if ada1:
        assert name in ("post_pars", "post_up"), name
        name += "_a1"
    half = os.environ.get("MW_LD10_PG", "1") != "0" and 2 * ntile > num_sms(d)
    grid = (ntile + 1) // 2 if half else min(num_sms(d), ntile)
    _post_kernel(name, d)((min(num_sms(d), grid), 1, 1), (384, 1, 1), *maps, int(ntile), float(eps), 0, int(mod.shape[0]), *((_map(upk[0], [DM, 2 * DM], DM, [DM, DM], swizzle=128, dtype="u8"),) if upk else ()), swo, swu, *upk[1:])
    STATS["post"] += 1
    return out


# --------------------------------------------------------------------------------------------------- prebuild
_PRE = ("pre", "pre_a1", "pre_a1_eb", "pre_kq_a1", "pre_kq_a1_eb", "prex", "prexp", "prex_eb", "prexp_eb", "pre_rms_raw_rope")
_POST = ("post", "post_pars_a1", "post_up_a1", "post_rms_raw")


def prebuild_items():
    """Prebuild plan entries (``_prebuild --arch sm100``): the e4m3 ``pre_fwd_f8`` / ``post_fwd_f8`` builds the route can launch."""
    import functools

    from . import cubin

    for names, extra in ((_PRE, ()), (_POST, ("BOX3=1",))):
        for name in names:
            table = KERNELS_ESM if name in KERNELS_ESM else KERNELS_VAR if name in KERNELS_VAR else KERNELS
            stem, _, _, _, *defs = table[name]
            yield functools.partial(cubin, stem + "_f8", (*(defs[0] if defs else ()), *extra))
