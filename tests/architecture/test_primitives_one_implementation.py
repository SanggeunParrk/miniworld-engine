"""The architecture primitives are the engine modules' (one implementation), and ``convert_linears_to_mp`` still converts only the
Linears architecture code builds."""

from torch import nn

from miniworld_engine.architecture import primitives as arch
from miniworld_engine.modules import primitives as engine


def test_one_implementation():
    assert arch.Dropout is engine.Dropout
    assert arch.InitType is engine.InitType
    assert arch.magnitude_normalize is engine.magnitude_normalize
    assert issubclass(arch.Linear, engine.Linear)
    assert issubclass(arch.MPLinear, engine.MPLinear)
    assert issubclass(arch.MPLinear, arch.Linear)
    for cls in (arch.Linear, arch.MPLinear):
        assert not {"__init__", "forward"} & set(vars(cls)), f"{cls.__name__} must add no behaviour of its own"


def test_convert_linears_to_mp_leaves_engine_linears_alone():
    root = nn.Sequential(arch.Linear(8, 4), engine.Linear(8, 4), arch.MPLinear(8, 4))
    assert arch.convert_linears_to_mp(root) == 1
    assert type(root[0]) is arch.MPLinear
    assert type(root[1]) is engine.Linear
    assert type(root[2]) is arch.MPLinear
