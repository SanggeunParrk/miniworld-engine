"""``modules.runtime``: the controls FoldForge / MiniWorld call around the modules instead of importing kernel packages."""
import torch

from miniworld_engine.kernels import _capture
from miniworld_engine.modules import runtime


def test_clear_weight_caches_empties_the_shared_store() -> None:
    _capture._FORMS[("probe", 0)] = object()
    runtime.clear_weight_caches()
    assert not _capture._FORMS


def test_parallel_residual_off_the_kernel_is_the_one_rounding_sum() -> None:
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    ys = [x + torch.randn_like(x) for _ in range(3)]
    want = (x.float() + sum(y.float() - x.float() for y in ys)).to(torch.bfloat16)
    assert torch.equal(runtime.parallel_residual(x, ys), want)


def test_the_capture_contexts_open_and_close() -> None:
    with runtime.static_weights(), runtime.static_inputs():
        pass


def test_trimul_batch_limit_per_card() -> None:
    from miniworld_engine.kernels.integrations.trimul_b200 import MAX_BATCH

    assert runtime.trimul_batch_limit((9, 0)) > 10**6            # the H100 module splits any batch itself
    assert runtime.trimul_batch_limit((10, 0)) == MAX_BATCH
    assert runtime.trimul_batch_limit((8, 6)) == 1
