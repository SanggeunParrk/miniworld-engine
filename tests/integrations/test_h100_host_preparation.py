import ctypes
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_launch as L
from miniworld_engine.kernels.trimul_inproj.cuda import _h100_runtime as T


def reference(fields):
    blob, align = bytearray(), 1
    for f in fields:
        if isinstance(f, L.Struct):
            b, a = reference(f.fields)
        else:
            b, a = L._pack_one(f)
        blob += b'\0' * ((-len(blob)) % a)
        blob += b
        align = max(align, a)
    blob += b'\0' * ((-len(blob)) % align)
    return bytes(blob), align


def test_layout_alignment_and_fresh_values():
    tensor = torch.empty(7)
    tm = L.TensorMap(bytes(range(128)))
    inner = L.Struct([3, tensor, False, L.f64(.7), b'ab'])
    fields = [1, tm, inner, None, L.u64(2**40), bytearray(b'xyz'), 2.5]
    s = L.Struct(fields)
    assert s.layout() == reference(fields)
    inner.fields[0] = -99
    inner.fields[1] = torch.empty(11)
    assert s.layout() == reference(fields)
    with pytest.raises(TypeError):
        L.Struct([2**32]).pack()


def test_live_packs_have_exclusive_storage():
    a = L._Packed([L.Struct([17, None])])
    b = L._Packed([L.Struct([31, None])])
    assert a.array[0] != b.array[0]
    assert ctypes.c_int32.from_address(a.array[0]).value == 17
    del b
    c = L._Packed([L.Struct([-83, None])])
    assert ctypes.c_int32.from_address(a.array[0]).value == 17
    assert ctypes.c_int32.from_address(c.array[0]).value == -83
    assert a.array[0] % 64 == 0


def test_thread_local_host_storage():
    def pack(n):
        for i in range(100):
            a = L._Packed([L.Struct([n+i, None])])
            assert ctypes.c_int32.from_address(a.array[0]).value == n+i
        return True
    with ThreadPoolExecutor(4) as pool:
        assert all(pool.map(pack, range(4)))


def test_config_read_once(tmp_path, monkeypatch):
    monkeypatch.setattr(T, 'SOURCES', tmp_path)
    T.read_config.cache_clear()
    p = tmp_path/'fixed.json'
    p.write_text('{"tile":64}')
    assert T.read_config('fixed.json')['tile'] == 64
    p.unlink()
    assert T.read_config('fixed.json')['tile'] == 64
    T.read_config.cache_clear()


def test_multiple_arguments_and_empty_pack():
    args = [L.i32(91), L.Struct([L.TensorMap(bytes(128)), None, -5]), L.f64(1.25)]
    packed = L._Packed(args)
    for i, arg in enumerate(args):
        blob, alignment = L._pack_one(arg)
        assert packed.array[i] % alignment == 0
        assert ctypes.string_at(packed.array[i], len(blob)) == blob
    assert L._Packed([]).array is None


def test_fixed_abi_updates_pointers_and_nested_values():
    key='test_fixed_abi_updates_pointers'
    inner=L.Struct.fixed('test_fixed_inner',[None,19])
    first=L.Struct.fixed(key,[L.TensorMap(bytes(128)),inner,None,1.25])
    assert first.layout()==reference(first.fields)
    t=torch.empty(11)
    inner.fields[:]=[t,-4]
    second=L.Struct.fixed(key,[L.TensorMap(bytes([7])*128),inner,t,3.5])
    assert second.layout()==reference(second.fields)
    with pytest.raises(ValueError):
        L.Struct.fixed(key,[None]).pack()


def test_map_spec_validation_and_context_scope(monkeypatch):
    class Driver:
        def __init__(self,ctx):self.ctx=ctx;self.calls=0
        def ctx_get_current(self):return self.ctx
        def tensor_map_encode_tiled(self,*args):
            self.calls+=1
            return bytes([self.ctx])*128
    a,b=Driver(1),Driver(2)
    monkeypatch.setattr(L,'driver',lambda:a)
    t=torch.empty(4,4)
    L._encoded_tensor_map.cache_clear()
    with L.tensor_map_scope(a):
        x=L.tensor_map(t,[4,4])
        assert L.tensor_map(t,[4,4]).raw==x.raw
        with L.tensor_map_scope(b):
            assert L.tensor_map(t,[4,4]).raw==bytes([2])*128
        assert L.tensor_map(t,[4,4]).raw==bytes([1])*128
    assert L._map_scope.active is None
    assert a.calls==1 and b.calls==1
    with pytest.raises(ValueError):
        L.tensor_map(t,[4,4],dims=[4,4],strides_bytes=[4])
    with pytest.raises(ValueError):
        L.tensor_map(t,[3,4])
    with pytest.raises(RuntimeError):
        with L.tensor_map_scope(a):
            raise RuntimeError('scope cleanup')
    assert L._map_scope.active is None
    L._encoded_tensor_map.cache_clear()
