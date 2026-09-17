"""F567 builder inputs share the pair model's width/length declarations."""
import torch
from miniworld_engine.kernels.drivers import BF16, dev, driver_heads, driver_width, driver_length, ragged


def operands():
    width = driver_width(128)
    hidden = driver_heads(width)
    length = ragged(driver_length(64))
    n, kg, kp = ragged(width), ragged(width), ragged(2 * hidden)
    m = length * length
    kw = dict(device=dev(), dtype=BF16)
    norm, x = torch.randn(m, kp, **kw), torch.randn(m, kg, **kw)
    wp = torch.randn(n, kp, **kw) / kp**.5
    wg = torch.randn(kg, n, **kw) / kg**.5
    residual = torch.randn(m, n, **kw)
    dropscale = (torch.rand(length, n, device=dev()) > .25).to(BF16) / .75
    return norm, x, wp, wg, residual, dropscale, length


def output_f567_train():
    from miniworld_engine.kernels.trimul_inproj.triton.output_fused import output_f567_train as launch
    launch(*operands())
