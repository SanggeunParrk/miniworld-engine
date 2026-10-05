"""Serve TriangleMultiplication from Anthropic's native TriMul payload (inference only).

The payload is the `trimul_native` package of Anthropic's `uplifting-biomolecular-modeling` release, optionally with this repo's
`archive/experiments-20260928:experiments/trimul_k1k3_inference` overlay; it is NOT vendored here. It is named at runtime by `TRIMUL_NATIVE_BUILD_DIR`, which points at
a payload's `build/` directory (its `python/`, `csrc/` and `testvectors/` sit beside it — `archive/experiments-20260928:experiments/trimul_k1k3_inference/build_payload.py`
assembles one). Setting that variable is the opt-in: with it set, a module built with `implementation="miniworld"` uses the payload for the
forward passes it can serve and its own kernels for everything else; `implementation="anthropic"` demands it and refuses rather than reroute.

What it serves (anything else falls back to the engine's own backends):

  * bf16 pair, no autograd and no live dropout scale — the release calls are forward-only;
  * sm_80 (A100): the release's sm80 K1/K3 units, built unchanged for sm_80; one direction D64-D384,
    bidirectional D64/D128, assembled through :func:`_serve_sm100` (the shared sm80-member launcher);
  * sm_90: the payload's `sm_90a` units, one for the module's (c_z, c_hidden): `tmn90_z128_h128` for one direction,
    `tmn90_z128_h256` for the bidirectional module (outgoing + incoming share one input LayerNorm and one 2*c_hidden output
    LayerNorm, so it is ONE unit at twice the hidden width, not two unidirectional calls, which would normalise each half
    separately and compute a different function);
  * sm_100 (B200): the release's sm_80 member (`trimul_k1_sm80` / `trimul_k3_sm80`: cp.async + mma.sync, the member that can run on
    sm_100 -- the sm_90a units are TMA + WGMMA) compiled from the UNMODIFIED sources for `sm_100a` into `build/sm_100a/` by
    :func:`build_sm100a` (``miniworld-engine dev build-anthropic-sm100a <payload>/build``), with a tile row for (c_z, c_hidden) in
    `sm80_ops.TILES`: one direction D64-D384, bidirectional D64 / D128. No binary for sm_100 ships with the release.

One direction on sm_90 goes through `trimul_native.face.serve`, which applies the release's own manifest and test-vector gate. The
bidirectional composition has no face entry, so it is composed here from the package's own primitives — K1 over the doubled hidden
width, the two half-channel contractions (outgoing NT, incoming TN), K3 with the residual fused — after the same `face.check()`.
On sm_100 the face has no architecture to route to (the release knows cc 9.0 and 8.x) and `sm80_ops.serve_sm80` admits cc 8.x only,
so :func:`_serve_sm100` assembles the sm_80 member exactly as `serve_sm80` does (K1 -> torch.bmm -> K3, same launch structs, same
default TN contraction form), with the bidirectional unit composed as on sm_90. Its cubins load through the release's own loader,
manifest-verified (cubin and source hashes), once cc 10.0 is registered as `sm_100a` in `launch.ARCH_OF_CC`. The release's test-vector
gate has no sm_100 class; `tests/integrations/test_anthropic_trimul_b200_gpu.py` holds these paths to the fp32 reference instead.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import torch

ENV = "TRIMUL_NATIVE_BUILD_DIR"
SM100_ARCH = "sm_100a"
SM100_UNITS = ("trimul_k1_sm80", "trimul_k3_sm80")      # the release's sm_80 member, compiled for sm_100a by build_sm100a
_LOADED: dict[str, Any] = {}
_CHECKED: set[int] = set()
_BYTE_CHECKS: dict[int, dict] = {}


class PayloadUnavailable(RuntimeError):
    """The payload named by the environment cannot serve this call."""


def payload_dir() -> str | None:
    """The payload `build/` directory named by the environment, or None."""
    return os.environ.get(ENV) or None


def _load() -> dict[str, Any]:
    """Import `trimul_native` from the payload beside `$TRIMUL_NATIVE_BUILD_DIR`, once per process."""
    build = payload_dir()
    if not build:
        raise PayloadUnavailable(f"{ENV} is not set (it must name a payload's build/ directory)")
    py = Path(build).resolve().parent / "python"
    if _LOADED.get("python") == str(py):
        return _LOADED
    if not (py / "trimul_native" / "face.py").is_file():
        raise PayloadUnavailable(f"{py} has no trimul_native package (a payload keeps python/ beside build/)")
    already = sys.modules.get("trimul_native")
    if already is not None and (already.__file__ is None or Path(already.__file__).resolve().parent != py / "trimul_native"):
        raise PayloadUnavailable(f"trimul_native is already imported from {already.__file__}, not from {py}")
    if _LOADED:
        raise PayloadUnavailable("changing the payload in a live process is not supported")
    if str(py) not in sys.path:
        sys.path.insert(0, str(py))
    _LOADED.update(python=str(py), face=importlib.import_module("trimul_native.face"),
                   ops=importlib.import_module("trimul_native.ops"), sm80=importlib.import_module("trimul_native.sm80_ops"),
                   launch=importlib.import_module("trimul_native.launch"))
    return _LOADED


def _sm100_refusal(c_z: int, c_hidden: int, arch: str = SM100_ARCH) -> str | None:
    build = Path(payload_dir() or "")
    missing = [u for u in SM100_UNITS if not (build / arch / f"{u}.cubin").is_file()]
    if missing:
        return (f"the payload has no {arch} build of its sm_80 member ({', '.join(missing)} under {build / arch}); "
                + (f"build it: miniworld-engine dev build-anthropic-sm100a {build}" if arch == SM100_ARCH else "build it with the release's trimul_native.build for sm_80"))
    if _load()["sm80"].TILES.get((c_z, c_hidden, "bf16")) is None:
        return f"the payload's sm_80 member has no tile row for (c_z={c_z}, c_hidden={c_hidden})"
    return None


def _checked(device: torch.device) -> dict[str, Any]:
    p = _load()
    idx = 0 if device.index is None else device.index
    if idx not in _CHECKED:
        # Verify source/cubin manifests and loader ABI. Local rebuilds are not
        # the release cubins named by its byte-vector manifest; this does not
        # claim the release's bitwise gate (independent qualification is separate).
        p["face"].check(device=idx, gate=False)
        if torch.cuda.get_device_capability(idx)[0] == 8:
            # The release explicitly supports --ignore-build for development
            # rebuilds. It still checks unchanged input/output digests; only
            # equality to the original compiler's cubin hash is waived.
            vectors = importlib.import_module("trimul_native.vectors")
            report = vectors.replay(device_index=idx, which="gate", ignore_build=True,
                                    raise_on_fail=True, quiet=True)
            if report["failed"] or not report["passed"]:
                raise PayloadUnavailable("A100 rebuilt TriMul failed the unchanged upstream byte vectors")
            _BYTE_CHECKS[idx] = report
        _CHECKED.add(idx)
    return p


def wanted(implementation) -> bool:
    """Should this module even ask the payload?  ``anthropic`` always (and refuses if it cannot serve); ``miniworld`` only when a payload
    is named and the engine is not pinned to Triton; every other option is the caller's explicit choice and is left alone."""
    from miniworld_engine import settings
    from miniworld_engine.modules.exceptions import ImplementationType
    if implementation == ImplementationType.ANTHROPIC:
        return True
    if implementation != ImplementationType.MINIWORLD:
        return False
    return bool(payload_dir()) and settings.current().engine_backend != "triton"


def refusal(pair: torch.Tensor, c_z: int, c_hidden: int, *, grad: bool, dropout: bool) -> str | None:
    """None if the payload named by the environment can run this forward, else why it cannot. Never raises."""
    try:
        if not payload_dir():
            return f"{ENV} is not set"
        if grad:
            return "the release calls are forward-only (grad is enabled)"
        if dropout:
            return "a live row-dropout scale needs the training path"
        if pair.dtype != torch.bfloat16:
            return f"the payload units are bf16, got {pair.dtype}"
        if not pair.is_cuda:
            return "the pair is not on a CUDA device"
        if pair.shape[0] != 1 or pair.shape[1] != pair.shape[2]:
            return f"one square pair plane per call, got {tuple(pair.shape)}"
        cc = torch.cuda.get_device_capability(pair.device)
        if cc == (10, 0):
            return _sm100_refusal(c_z, c_hidden)
        if cc[0] == 8:                                    # A100 (sm_80): the release's sm_80 member as built, assembled by `_serve_sm100`
            return _sm100_refusal(c_z, c_hidden, "sm_80")
        if cc != (9, 0):
            return "the payload serves sm_90 (its sm_90a units), sm_100 (its sm_80 member rebuilt for sm_100a) and sm_80 (that member as built)"
        p = _load()
        if not p["ops"].kernels().has_unit(c_z, c_hidden):
            return f"the payload has no unit for (c_z={c_z}, c_hidden={c_hidden})"
        return None
    except Exception as exc:                          # an unusable payload is a fallback, not a crash
        return f"{type(exc).__name__}: {exc}"


def serves(pair: torch.Tensor, c_z: int, c_hidden: int, *, grad: bool, dropout: bool) -> bool:
    """Can the payload run this forward? The caller falls back to its own backends when not."""
    return refusal(pair, c_z, c_hidden, grad=grad, dropout=dropout) is None


def require(pair: torch.Tensor, c_z: int, c_hidden: int, *, grad: bool, dropout: bool) -> None:
    """An explicit ``implementation="anthropic"`` refuses with the reason instead of rerouting."""
    why = refusal(pair, c_z, c_hidden, grad=grad, dropout=dropout)
    if why is not None:
        raise PayloadUnavailable("the anthropic TriMul payload cannot serve this call: " + why)


def _weights(module: Any) -> dict[str, torch.Tensor]:
    return {"ln_in_w": module.ln_pair.weight, "ln_in_b": module.ln_pair.bias,
            "w_ag": module.to_left_gate.weight, "w_ap": module.to_left.weight,
            "w_bg": module.to_right_gate.weight, "w_bp": module.to_right.weight,
            "ln_out_w": module.ln_out.weight, "ln_out_b": module.ln_out.bias,
            "w_o": module.to_out.weight, "w_og": module.to_gate.weight}


def _signature(tensors) -> tuple:
    # Replacement, load_state_dict/copy_, device and dtype changes all have to invalidate the packed copy.
    return tuple((id(t), t._version, t.device, t.dtype, tuple(t.shape)) for t in tensors)


def _prepared(module: Any) -> tuple[dict, dict]:
    w = _weights(module)
    key = (_signature(w.values()), module.ln_pair.eps)
    if getattr(module, "_native_key", None) != key:
        module._native_weights = {k: t.detach().contiguous() for k, t in w.items()}
        module._native_cache = {}
        module._native_key = key
    return module._native_weights, module._native_cache


def _pair_mask(pair: torch.Tensor, mask: torch.Tensor | None, ops) -> torch.Tensor | None:
    """The pair mask in an element type THIS payload's K1 can read.

    A payload built with the templated mask declares the types it instantiates (`ops.MASK_NATIVE_DTYPES`) and takes the module's own
    bool mask as-is; the upstream package has only the fp32 kernel and reads whatever buffer it is handed AS fp32 — handing it a bool
    tensor is a four-times-too-long read, which at L384 returned wrong numbers and at L768 was an illegal access. So: bool when the
    payload says it can, fp32 otherwise.
    """
    if mask is None:
        return None
    m = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).reshape(pair.shape[1], pair.shape[2])
    native = getattr(ops, "MASK_NATIVE_DTYPES", ())
    return m.contiguous() if torch.bool in native else m.to(torch.float32).contiguous()


def update_unidirectional(module: Any, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """`pair + trimul(pair)` for one direction, residual fused in K3, through the payload's served face."""
    if module.ln_pair.eps != module.ln_out.eps:
        raise PayloadUnavailable("the native TriMul normalises input and output with one epsilon")
    cc = torch.cuda.get_device_capability(pair.device)
    if cc == (10, 0):
        return _serve_sm100(module, pair, mask, "outgoing" if module.outgoing else "incoming")
    p = _checked(pair.device)
    weights, cache = _prepared(module)
    if cc[0] == 8:
        # Use the release face and its packaged size/tile lookup, instead of
        # reconstructing sm80 launch structs in our bidirectional adapter.
        out = p["face"].serve(pair, _pair_mask(pair, mask, p["ops"]),
                              direction="outgoing" if module.outgoing else "incoming",
                              weights=weights, residual=True, cache=cache, eps=module.ln_pair.eps,
                              config={"gate": False})
        module.native_selection = {"unit": "sm_80 member built for sm_80", "entry": "trimul_native.face.serve",
                                   "local_rebuild": True, "release_byte_gate": False,
                                   "validation": "manifest/loadcheck and unchanged upstream byte vectors (--ignore-build)",
                                   "byte_vector_passed": _BYTE_CHECKS[pair.device.index or 0]["passed"]}
        return out
    out = p["face"].serve(pair, _pair_mask(pair, mask, p["ops"]), direction="outgoing" if module.outgoing else "incoming",
                          weights=weights, residual=True, cache=cache, eps=module.ln_pair.eps)
    module.native_selection = {k: v for k, v in cache.items() if isinstance(k, tuple) and k and k[0] == "_sel"}
    return out


def update_bidirectional(module: Any, pair: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """`pair + bidir_trimul(pair)` as one unit at c_hidden = 2 * d_hidden: K1 (natural layout) -> the two half-channel
    contractions -> K3 over both halves with the residual fused."""
    if module.ln_pair.eps != module.ln_out.eps:
        raise PayloadUnavailable("the native TriMul normalises input and output with one epsilon")
    if torch.cuda.get_device_capability(pair.device) == (10, 0) or torch.cuda.get_device_capability(pair.device)[0] == 8:
        return _serve_sm100(module, pair, mask, "bidirectional")
    p = _checked(pair.device)
    ops = p["ops"]
    weights, cache = _prepared(module)
    z3 = pair[0] if pair.shape[0] == 1 else pair.reshape(pair.shape[1], pair.shape[2], pair.shape[3])
    z3 = z3.contiguous()
    packed = ops._packed(weights, cache, z3.device)      # memoised by weight address: a fresh pack per call would be captured into the graph
    ops._check(z3, packed)
    ch, h = packed["ch"], module.d_hidden
    Np = ops.ceil16(z3.shape[0])
    ab = ops.planes(z3, _pair_mask(pair, mask, ops), packed, transpose=False, lnm=2, Np=Np, cache=cache, eps=module.ln_pair.eps)
    tri = ops._buf(cache, ("tri", Np, ch), (ch, Np, Np), torch.bfloat16, z3.device)
    a, b = ab[:ch], ab[ch:]
    torch.bmm(a[:h], b[:h].transpose(1, 2), out=tri[:h])                 # outgoing half: sum_k a[i,k] b[j,k]
    torch.bmm(a[h:].transpose(1, 2), b[h:], out=tri[h:])                 # incoming half: sum_k a[k,i] b[k,j]
    out = ops.epilogue(tri, z3, packed, residual=True, lnm=1, cache=cache, eps=module.ln_pair.eps)
    module.native_selection = {"unit": f"z{packed['cz']}_h{ch}", "composed": "K1 + 2 contractions + K3"}
    return out.unsqueeze(0)


def _serve_sm100(module: Any, pair: torch.Tensor, mask: torch.Tensor | None, direction: str) -> torch.Tensor:
    """`pair + trimul(pair)` on sm_100 from the release's sm_80 member built for sm_100a, assembled as `sm80_ops.serve_sm80` assembles
    it (bf16, B = 1, the fast rows): K1 -> torch.bmm -> K3 with the residual fused, the launch structs packed once per shape and their
    tensor addresses patched per call. ``direction`` "outgoing" / "incoming": one unit at c_hidden = d_hidden in the release's contraction
    form (TN unless TRIMUL_SM80_FORM says NT). "bidirectional": ONE unit at c_hidden = 2 * d_hidden, natural planes, the outgoing half
    NT and the incoming half TN -- the composition `update_bidirectional` runs with the sm_90a unit."""
    if module.ln_pair.eps != module.ln_out.eps:
        raise PayloadUnavailable("the native TriMul normalises input and output with one epsilon")
    p = _checked(pair.device) if torch.cuda.get_device_capability(pair.device)[0] == 8 else _load()
    s, launch = p["sm80"], p["launch"]
    launch.ARCH_OF_CC.setdefault((10, 0), SM100_ARCH)    # the release keys cubins by device arch and knows no cc 10.0
    weights, cache = _prepared(module)
    z4 = pair.contiguous()
    n, c = int(z4.shape[1]), int(z4.shape[3])
    dev = z4.device.index if z4.device.index is not None else torch.cuda.current_device()
    pk = cache.get("_sm100_pk")
    if pk is None:
        pk = cache["_sm100_pk"] = s.pack_weights(weights, z4.device)
    d = pk["D"]                                           # the unit's c_hidden: d_hidden, or 2 * d_hidden bidirectionally
    tiles = s.TILES.get((c, d, "bf16"))
    if tiles is None:
        raise PayloadUnavailable(f"the payload's sm_80 member has no tile row for (c_z={c}, c_hidden={d})")
    k1n, k3n = tiles
    k1, k3 = s._kernel(k1n, dev), s._kernel(k3n, dev)    # load_unit: build/sm_100a/<unit>.cubin, manifest-verified
    g1, g3 = k1.geo, k3.geo
    preln = k1n.startswith("k1z_")                        # K1 also writes its LayerNorm rows; K3 reads those instead of z
    np_ = (n + s.PAD - 1) // s.PAD * s.PAD
    bkey = ("_sm100_buf", n, d, np_, preln, dev)
    bufs = cache.get(bkey)
    if bufs is None:
        bufs = cache[bkey] = (torch.empty((2 * d, np_, np_), dtype=torch.bfloat16, device=z4.device),
                              torch.empty((d, np_, np_), dtype=torch.bfloat16, device=z4.device),
                              torch.empty_like(z4) if preln else None)
    ab, x, zln = bufs
    m = None if mask is None else (mask.reshape(n, 1) & mask.reshape(1, n)).to(torch.float32).contiguous()
    out = torch.empty_like(z4)
    form = "NT" if direction == "bidirectional" else s._form(cache)
    rowtok = True if direction == "bidirectional" else (direction == "outgoing") != (form == "TN")
    ctas0, ctas1 = s._ctas_per_sm(k3, g3["smem"]), s._ctas_per_sm(k3, g3["smem"] + g3.get("stash_bytes", 0))
    stash = 1 if (g3.get("stash_bytes") and ctas1 >= ctas0) else 0
    smem3 = g3["smem"] + (g3["stash_bytes"] if stash else 0)
    eps = float(module.ln_pair.eps)
    pkey = ("_sm100_packs", k1n, k3n, n, c, d, direction, form, m is not None, stash, eps, dev)
    packs = cache.get(pkey)
    if packs is None:
        zl = zln if zln is not None else launch.u64(0)
        p1 = launch.Struct([z4, m if m is not None else launch.u64(0), ab, pk["wg"], pk["wp"], pk["g_in"], pk["b_in"], zl,
                            launch.i64(n * n * c), launch.i64(0), launch.i64(np_ * np_), launch.i32(n), launch.i32(np_), launch.i32(1),
                            launch.i32(d), launch.i32(1 if rowtok else 0), launch.f32(eps)])
        p3 = launch.Struct([x, z4, out, pk["wo"], pk["wog"], pk["g_out"], pk["b_out"], pk["g_in"], pk["b_in"], zl,
                            launch.i64(n * n * c), launch.i64(np_ * np_), launch.i32(n), launch.i32(np_), launch.i32(1), launch.i32(d),
                            launch.i32(1), launch.i32(stash), launch.f32(eps)])
        packs = cache[pkey] = (k1.argpack([p1]), k3.argpack([p3]), ((np_ + g1["BM"] - 1) // g1["BM"], np_, 1),
                               (g1["threads"], 1, 1), ((n + g3["BM"] - 1) // g3["BM"], n, 1), (g3["threads"], 1, 1))
    a1, a3, grid1, blk1, grid3, blk3 = packs
    a1.set_ptr(0, z4, field=0)
    a1.set_ptr(0, m if m is not None else 0, field=1)
    a1.set_ptr(0, ab, field=2)
    a3.set_ptr(0, x, field=0)
    a3.set_ptr(0, z4, field=1)
    a3.set_ptr(0, out, field=2)
    if zln is not None:
        a1.set_ptr(0, zln, field=7)
        a3.set_ptr(0, zln, field=9)
    s._launch(k1, grid1, blk1, a1, g1["smem"])
    a, b = ab[:d], ab[d:]
    if direction == "bidirectional":
        h = d // 2
        torch.bmm(a[:h], b[:h].transpose(1, 2), out=x[:h])                 # outgoing half: sum_k a[i,k] b[j,k]
        torch.bmm(a[h:].transpose(1, 2), b[h:], out=x[h:])                 # incoming half: sum_k a[k,i] b[k,j]
    elif form == "TN":
        torch.bmm(a.transpose(1, 2), b, out=x)                             # planes [k][i] (either direction)
    else:
        torch.bmm(a, b.transpose(1, 2), out=x)                             # planes [i][k] (either direction)
    s._launch(k3, grid3, blk3, a3, smem3)
    arch = launch.ARCH_OF_CC[torch.cuda.get_device_capability(pair.device)]
    module.native_selection = {"unit": f"sm_80 member built for {arch}", "kernels": f"{k1n} | torch.bmm {form} | {k3n}"}
    if arch == "sm_80":
        module.native_selection.update(entry="composed native K1 + two contractions + K3",
                                       byte_vector_passed=_BYTE_CHECKS[dev]["passed"],
                                       validation="unchanged upstream byte vectors (--ignore-build)")
    return out


def build_sm100a(build_dir: str | os.PathLike, nvcc: str | None = None) -> list[str]:
    """Compile the release's sm_80 member for sm_100a into ``<build_dir>/sm_100a/`` with the release's OWN builder
    (`trimul_native.build.build_unit`: its fixed flags, dependency hashes, ptxas report) and record the entries in ``manifest.json``.
    The sources are untouched; their ``// build: archs=sm_80`` line only filters the builder's default job list, which is why the two
    units are named here. ``nvcc`` must know sm_100a (CUDA 12.8+) and be no newer (major) than the driver. Returns the entry keys."""
    build = Path(build_dir).resolve()
    py = build.parent / "python"
    if str(py) not in sys.path:
        sys.path.insert(0, str(py))
    b = importlib.import_module("trimul_native.build")
    m = importlib.import_module("trimul_native.manifest")
    src = str(build.parent / "csrc")
    units = b.discover_units(src)
    exe = b.find_nvcc(nvcc)
    nv = b.nvcc_version(exe)
    man = m.load(str(build))
    keys = []
    for unit in SM100_UNITS:
        ok, key, ent = b.build_unit(exe, nv, unit, units[unit], SM100_ARCH, src, str(build), list(b.BASE_FLAGS))
        if not ok or ent is None:
            raise RuntimeError(f"building {unit} for {SM100_ARCH} failed (see {build / SM100_ARCH / (unit + '.ptxas.txt')})")
        man.setdefault("units", {})[key] = ent
        keys.append(key)
    m.save(str(build), man)
    return keys
