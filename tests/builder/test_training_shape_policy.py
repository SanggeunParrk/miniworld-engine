"""Training gets two lengths; inference and shared forward coverage survive."""
import csv
from pathlib import Path
from collections import defaultdict
from unittest.mock import patch

import pytest

from miniworld_engine.autotune import builder, derive, module_registry as registry


def test_training_module_units_use_only_the_two_real_lengths():
    units = derive.units(registry.module_rows(), arch='sm90')
    actual = defaultdict(set)
    for unit in units:
        if unit.mode == 'train':
            actual[unit.stream].add(unit.length)
            # Shape pruning must not turn off training augmentation or dropout.
            if unit.option and unit.option[0] == 'p_drop':
                assert float(unit.option[1]) == .25
    assert dict(actual) == {s:set(v) for s,v in registry.TRAIN_STREAM_LADDERS.items()}
    assert any(u.mode == 'train' and u.augmentation == 48 for u in units)
    assert any(u.mode == 'train' and u.option == ('p_drop', '0.25') for u in units)


def test_inference_row_lengths_are_not_filtered():
    for row in registry.module_rows():
        if 'eval' in row.modes:
            assert row.lengths_for('eval') == row.lengths
    assert registry.STREAM_LADDERS['token_pair'] == (128,256,384,512,640,768)
    assert registry.STREAM_LADDERS['atom_single'] == tuple(range(1024,8193,1024))


def test_builder_and_derivation_agree_on_training_shapes():
    with patch.object(builder, 'device_sm', return_value='sm90'):
        for case in builder.cases():
            for unit in builder.units([case]):
                assert unit.length in case.lengths_for(unit.dim_index, train=unit.train)
                if unit.train:
                    assert unit.length in registry.TRAIN_STREAM_LADDERS[case.stream_for(unit.dim_index)]


def test_legacy_case_path_honors_the_training_policy():
    # Synthetic cases do not have ModuleRows; they still must not widen training.
    from dataclasses import replace
    case = next(c for c in builder.cases() if c.name == 'triangle_multiplication')
    case = replace(case, rows=())
    with patch.object(builder, 'device_sm', return_value='sm90'):
        units = builder.units([case])
    assert {u.length for u in units if u.train} == {384,768}
    assert {u.length for u in units if not u.train} == set(registry.STREAM_LADDERS['token_pair'])


def test_backward_driver_units_do_not_reintroduce_extra_buckets():
    path = Path(builder.__file__).resolve().parents[1]/'kernels/registry.csv'
    rows = {r['kernel']:r for r in csv.DictReader(path.open())}
    actual = defaultdict(set)
    for unit in builder.op_units():
        modes = set(rows[unit.op]['build_modes'].split('|'))
        assert modes and modes <= {'train','eval'}
        if modes == {'train'}:
            expected = registry.TRAIN_ATOM_LENGTHS if unit.side == 'atom' else registry.TRAIN_TOKEN_LENGTHS
            assert unit.length in expected, unit
            actual[(unit.op,unit.side)].add(unit.length)
    assert actual
    assert all(v == ({4096,8192} if side=='atom' else {384,768}) for (_,side),v in actual.items())


@pytest.mark.parametrize('op', ['layernorm_fwd_saveact_triton', 'trimul_output_f567_train_triton'])
def test_forward_helpers_keep_inference_buckets_even_if_named_train(op):
    # Autograd Function.forward may run under no_grad, so names are not modes.
    units = builder.op_units(only={op})
    assert {u.length for u in units if u.side != 'atom'} >= {128,256,384,512,640,768}


def test_noise_is_not_a_token_length_and_bad_mode_fails():
    assert registry.lengths_for_mode('noise',(1,),'train') == (1,)
    with pytest.raises(ValueError, match='unknown build mode'):
        registry.lengths_for_mode('token_pair',(384,768),'typo')
