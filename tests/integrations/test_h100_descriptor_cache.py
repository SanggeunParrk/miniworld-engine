"""Descriptor cache identity must cover addresses, layouts and CUDA contexts."""
from miniworld_engine.kernels.trimul_inproj.cuda._h100_launch import _encoded_tensor_map


def test_descriptor_cache_key_and_bound():
    class Driver:
        def __init__(self):
            self.calls = 0
        def tensor_map_encode_tiled(self, *args):
            self.calls += 1
            return (self.calls, args)
    drv = Driver()
    key = (drv, 7, 9, 2, 4096, (128, 384), (256,), (64, 64), (1, 1), 0, 3, 2, 0)
    first = _encoded_tensor_map(*key)
    assert _encoded_tensor_map(*key) is first
    assert drv.calls == 1
    for index, value in [(1, 8), (4, 8192), (5, (128, 768)), (6, (512,)),
                         (7, (64, 32)), (10, 2)]:
        other = list(key)
        other[index] = value
        assert _encoded_tensor_map(*other) != first
    assert drv.calls == 7
    assert _encoded_tensor_map.cache_info().maxsize == 8192
    _encoded_tensor_map.cache_clear()
