"""Native K1/K3 inference, packaged with its measured tile table and CUDA sources."""

import json
from collections import OrderedDict

import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_bmm as _H100_BMM
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_infer_kernel as K


#: Weight sets whose K1 / K3 launch plans and tensor maps are kept (one per TriMul layer of a model; least recently used dropped).
PLAN_SETS = 256


class _Kernels(K.Kernels):
    def __init__(self, native_residual=False):
        super().__init__()
        self.native_residual = native_residual
        self._plans = OrderedDict()

    def plans(self, w):
        """The launch-plan cache of one weight set (K1 / K3 argument packs, tensor maps by address), kept across calls. A fresh
        dict per call re-encoded every descriptor and argument pack on the host (~0.2 ms of the ~0.5 ms a call took, 2026-10-30);
        pointers that move between calls (z, planes, output) are re-patched by K1 / K3 themselves, so only the host work is saved."""
        key = tuple(t.data_ptr() for t in w.get("wq", ())) + tuple(
            w[k].data_ptr() for k in ("w1", "ln_in_w", "ln_in_b", "ln_out_w", "ln_out_b", "wo", "wg") if k in w)
        cache = self._plans.get(key)
        if cache is None:
            cache = self._plans[key] = {}
            if len(self._plans) > PLAN_SETS:
                self._plans.popitem(last=False)
        else:
            self._plans.move_to_end(key)
        return cache

    def has_unit(self, cz, ch):
        return ("sm_90a", cz, ch, "b") in K.TILE_TABLE

    def unit(self, cz, ch):
        if (cz, ch) not in self._units:
            inc = T.SOURCES / "inference"
            defs = json.loads((inc / "defines.json").read_text())
            if self.native_residual:
                defs["TMN_K3W_NATIVE_RESIDUAL"] = 1
            source = (
                '#include "tmn_kernels.cuh"\n#define TMN_CZ %d\n#define TMN_CH %d\n#include "tmn_sm90_unit.cuh"\n'
                % (cz, ch)
            )
            flags = [
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "-O3",
                "-arch=sm_90a",
                "--cubin",
                "-lineinfo",
                "-I" + str(inc),
                "-I" + str(T.ANTHROPIC_V5),
            ] + ["-D%s=%s" % kv for kv in defs.items()]
            path = T.compile_text(source, flags)
            u = T.load_unit(str(path), "inference_%d_%d" % (cz, ch))
            u.entry = {
                "flags": flags,
                "kernels": {n: {} for n in K.serve_names("sm_90a", cz, ch, "b")},
            }
            self._units[cz, ch] = u
        return self._units[cz, ch]


@T.device_cache
def _kernels(native_residual=False):
    return _Kernels(native_residual)


def _inference_fake(x, weights, mask, direction, eps):
    """Return the input shape and dtype, including its batch axis."""
    return torch.empty_like(x)


@opaque(fake=_inference_fake, name="trimul_h100_infer")
def inference(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    mask: torch.Tensor | None,
    direction: int,
    eps: float,
) -> torch.Tensor:
    """Run packaged K1, contractions, and K3 with a GPU/shape-specific schedule. mask: None, a bool token mask [N] (K1
    forms m[i] & m[j] itself) or a pair mask [N, N]. x [B, N, N, cz] with B > 1 (N % 16 == 0): the samples run in the same launches,
    mask a token mask [B N] (``_run_inference``)."""
    config = _select_config(x, weights, mask, direction, eps)
    return _run_inference(x, weights, mask, direction, eps, config)


def _inference_into_fake(x, weights, mask, direction, eps, out):
    """Writes ``out`` in place; returns nothing."""
    return None


@opaque(fake=_inference_into_fake, name="trimul_h100_infer_into", mutates_args=("out",))
def inference_into(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    mask: torch.Tensor | None,
    direction: int,
    eps: float,
    out: torch.Tensor,
) -> None:
    """``inference`` writing y into ``out`` (contiguous, like x): a chunk of a batch writes its rows of the caller's output."""
    config = _select_config(x, weights, mask, direction, eps)
    _run_inference(x, weights, mask, direction, eps, config, out)


def _select_config(x, weights, mask, direction, eps):
    from miniworld_engine.autotune.fused_config import trimul_candidates, validator
    from miniworld_engine.autotune.native import choose_config, tensor_key
    grid = trimul_candidates(x.shape[-1], weights[0].shape[0], x.shape[1], direction)
    if (x.shape[-1] == 128 and weights[0].shape[0] == 256
            and x.shape[1] == 128 and direction == 0):
        # L128: 128-token CTAs cover 128 of the 132 SMs, versus 86 CTAs
        # with the wide default. The bounded paired Graph search selected
        # four weight slots; retain the existing output tile and live packing.
        short = dict(grid[0], k1_0=2, k1_1=64, k1_2=4, k1_3=2)
        grid = [short, *[c for c in grid if c != short]]
    run = lambda c: _run_inference(x, weights, mask, direction, eps, c)
    return choose_config("trimul_fwd_sm90_cuda", grid, dtype=str(x.dtype),
                         bucket=tensor_key(x, *weights, mask, extra=(direction, eps)),
                         device_index=x.device.index, run=run, validate=validator(run, grid[0]))


def _run_inference(x, weights, mask, direction, eps, config, dest=None):
    # weights: left, left gate, right, right gate, output gate, output, four LN affine tensors.
    wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = weights
    # Batched samples (x [B, N, N, cz], B > 1): tokens b-major, the B samples stacked along the slow token axis (rows = B N) of one K1 / K3
    # launch each, planes [c, B N, N] so the contraction batches over c x B; the mask is a token mask [B N]. Needs N % 16 == 0 (no pad).
    bsz = x.shape[0]
    n = x.shape[1]
    cz = x.shape[-1]
    ch = wl.shape[0]
    z = x[0] if bsz == 1 else x.view(bsz * n, n, cz)
    npad = (n + 15) // 16 * 16
    if bsz > 1 and (npad != n or (mask is not None and mask.ndim != 1)):
        raise ValueError("batched H100 TriMul inference needs N % 16 == 0 and a token mask [B N]")
    rows = bsz * npad
    with torch.cuda.device(x.device):
        T._launch_module()._make_context_current(x.device.index)
        # Measured D384/L512 outgoing candidate. Keep a separate loaded unit
        # so toggling the opt-in never reuses a cubin with different arithmetic.
        native_residual = cz == ch == 384 and n == 512 and direction == 1
        kk = _kernels(True) if native_residual else _kernels()
        from miniworld_engine.autotune.fused_config import unpack_trimul
        cfg = unpack_trimul(config)
        # K1 reads row-major weights in place (no pack kernel, nothing snapshotted); any other layout is packed live, which a
        # CUDA graph captures like every other launch. (A transposed-B K1 for the column-major D128 bidirectional storage was
        # bitwise-correct but 11 % slower at L768 and 2x at L128, 2026-09-29, so that storage keeps the pack.)
        quad = (wl, wlg, wr, wrg)
        if all(t.dtype == torch.bfloat16 and t.stride() == (cz, 1) for t in quad):
            wmap = {"wq": quad}
        elif cz in (64, 128) or native_residual or T.cuda_aux_enabled():
            from ._h100_pack import pack_into_cuda
            packed = x.new_empty((4 * ch, cz))
            pack_into_cuda(packed, *quad)
            wmap = {"w1": packed}
        else:
            wmap = {"w1": torch.stack(
                (
                    torch.cat((wlg, wrg)).reshape(-1, 32, cz),
                    torch.cat((wl, wr)).reshape(-1, 32, cz),
                ),
                1,
            ).reshape(4 * ch, cz).contiguous()}
        w = {
            **wmap,
            "ln_in_w": gi,
            "ln_in_b": bi,
            "ln_out_w": go,
            "ln_out_b": bo,
            "wo": wp,
            "wg": wg,
        }
        ab = x.new_empty((2 * ch, rows, npad))
        tri = x.new_empty((ch, rows, npad))
        out = torch.empty_like(z) if dest is None else dest.view(z.shape)   # dest: a caller-owned buffer like x (chunked batches)
        cache = kk.plans(w)
        kk.k1(
            z,
            mask if mask is None or mask.ndim == 1 else mask.reshape(n, n).float(),
            w,
            ab,
            N=n,
            Np=npad,
            cz=cz,
            ch=ch,
            cfg=cfg["k1"],
            lnm=2,
            eps=eps,
            cache=cache,
            token_mask=mask is not None and mask.ndim == 1,
            rows=None if bsz == 1 else rows,   # one sample: N rows, planes padded to Np
        )
        # [c, B N, N] -> [c B, N, N]: the samples join the channel planes in cuBLAS's batch dim (a view; B = 1 is the same shape as before)
        fold = lambda t: t.view(-1, npad, npad)
        left, right = ab[:ch], ab[ch:]
        if direction == 0:
            h = ch // 2
            _H100_BMM.bmm(fold(left[:h]), fold(right[:h]).transpose(-1, -2), out=fold(tri[:h]))
            _H100_BMM.bmm(fold(left[h:]).transpose(-1, -2), fold(right[h:]), out=fold(tri[h:]))
        elif direction == 1:
            _H100_BMM.bmm(fold(left), fold(right).transpose(-1, -2), out=fold(tri))
        else:
            _H100_BMM.bmm(fold(left).transpose(-1, -2), fold(right), out=fold(tri))
        kk.k3(
            tri,
            z,
            w,
            out,
            N=n,
            Np=npad,
            cz=cz,
            ch=ch,
            residual=True,
            cfg=cfg["k3"],
            lnm=1,
            eps=eps,
            cache=cache,
            rows=None if bsz == 1 else rows,
        )
        return out.unsqueeze(0) if bsz == 1 else out.view(x.shape)
