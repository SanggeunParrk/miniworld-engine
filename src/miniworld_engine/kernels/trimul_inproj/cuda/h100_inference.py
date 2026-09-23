"""Native K1/K3 inference, packaged with its measured tile table and CUDA sources."""

import json
import torch
from miniworld_engine.kernels._compile import opaque
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_infer_kernel as K


class _Kernels(K.Kernels):
    def has_unit(self, cz, ch):
        return ("sm_90a", cz, ch, "b") in K.TILE_TABLE

    def unit(self, cz, ch):
        if (cz, ch) not in self._units:
            inc = T.SOURCES / "inference"
            defs = json.loads((inc / "defines.json").read_text())
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
def _kernels():
    return _Kernels()


def _inference_fake(x, weights, mask, direction, eps):
    """Return the input shape and dtype, including its batch axis."""
    return torch.empty_like(x)


@opaque(fake=_inference_fake, name="trimul_h100_infer")
def inference(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    mask: torch.Tensor,
    direction: int,
    eps: float,
) -> torch.Tensor:
    """Run packaged K1, contractions, and K3 with live weights."""
    # weights: left, left gate, right, right gate, output gate, output, four LN affine tensors.
    wl, wlg, wr, wrg, wg, wp, gi, bi, go, bo = weights
    z = x[0]
    n = x.shape[1]
    cz = x.shape[-1]
    ch = wl.shape[0]
    npad = (n + 15) // 16 * 16
    with torch.cuda.device(x.device):
        T._launch_module()._make_context_current(x.device.index)
        kk = _kernels()
        cfg = K.lookup(
            "sm_90a", cz, ch, "b", n, "incoming" if direction == 2 else "outgoing", True
        )
        # Live packing is captured in CUDA graphs; there is no stale parameter snapshot.
        w1 = (
            torch.stack(
                (
                    torch.cat((wlg, wrg)).reshape(-1, 32, cz),
                    torch.cat((wl, wr)).reshape(-1, 32, cz),
                ),
                1,
            )
            .reshape(4 * ch, cz)
            .contiguous()
        )
        w = {
            "w1": w1,
            "ln_in_w": gi,
            "ln_in_b": bi,
            "ln_out_w": go,
            "ln_out_b": bo,
            "wo": wp,
            "wg": wg,
        }
        ab = x.new_empty((2 * ch, npad, npad))
        tri = x.new_empty((ch, npad, npad))
        out = torch.empty_like(z)
        cache = {}
        kk.k1(
            z,
            mask.reshape(n, n).float(),
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
        )
        left, right = ab[:ch], ab[ch:]
        if direction == 0:
            h = ch // 2
            torch.bmm(left[:h], right[:h].transpose(-1, -2), out=tri[:h])
            torch.bmm(left[h:].transpose(-1, -2), right[h:], out=tri[h:])
        elif direction == 1:
            torch.bmm(left, right.transpose(-1, -2), out=tri)
        else:
            torch.bmm(left.transpose(-1, -2), right, out=tri)
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
        )
        return out.unsqueeze(0)
