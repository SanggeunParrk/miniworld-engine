"""A precompile round is only as fast as its slowest chunk.

Measured on the A6000 rebuild (461 rounds, 15 workers each): aggregate worker occupancy was 50%,
429 CPU-hours of workers waiting on a tail. The chunks were contiguous slices of the config list,
so the configs that run the full 60 s compile budget before being SIGKILLed -- the deep-pipeline,
big-tile end of the grid, which the grid lists together -- landed in the same few chunks. One
worker drew thirteen of them and fifteen waited ~800 s.

Dealing the chunks round-robin over a cost-ordered list puts one expensive config in each chunk
instead of thirty-two in one. These tests pin the property that matters (no chunk collects the
expensive tail), not the exact deal.
"""
from __future__ import annotations

from miniworld_engine.autotune import capture


class _Cfg:
    def __init__(self, block, stages=2, warps=4):
        self.kwargs = {"BLOCK_M": block}
        self.num_stages = stages
        self.num_warps = warps


def test_the_expensive_configs_are_spread_across_chunks():
    # 96 configs, the 12 most expensive contiguous at the end -- the shape a swept grid has.
    items = list(range(96))
    costs = [1] * 84 + [1000] * 12
    chunks = capture._balanced_chunks(items, costs, n_chunks=12)
    loads = [sum(costs[i] for i in c) for c in chunks]
    assert max(loads) - min(loads) <= 1000, (
        f"one chunk drew more than its share of the expensive tail: {sorted(loads)}")


def test_a_contiguous_deal_is_what_this_replaces():
    """Guards the claim above rather than the code: sliced contiguously, the same 96 configs put
    every expensive one in the last chunk."""
    costs = [1] * 84 + [1000] * 12
    sliced = [costs[i:i + 8] for i in range(0, 96, 8)]
    loads = [sum(c) for c in sliced]
    assert max(loads) >= 8000
    assert min(loads) == 8


def test_every_config_is_dealt_exactly_once():
    items = list(range(101))
    chunks = capture._balanced_chunks(items, [i % 7 for i in items], n_chunks=8)
    assert sorted(i for c in chunks for i in c) == items


def test_a_round_smaller_than_the_chunk_count_yields_no_empty_chunks():
    """`map_async` over empty chunks is a fork per nothing."""
    chunks = capture._balanced_chunks([1, 2, 3], [1, 1, 1], n_chunks=16)
    assert chunks
    assert all(chunks)


def test_deeper_pipelines_and_bigger_tiles_sort_first():
    """The cost hint only has to ORDER; it is never reported as a time."""
    cheap, wide, deep = _Cfg(32), _Cfg(256), _Cfg(32, stages=6)
    assert capture._chunk_cost(wide) > capture._chunk_cost(cheap)
    assert capture._chunk_cost(deep) > capture._chunk_cost(cheap)


class _Autotuner:
    arg_names = ("x", "n", "BLOCK")
    keys = ("n",)

    def __init__(self, n, dtype):
        self.nargs = {"x": _Tensor(dtype), "n": n, "BLOCK": 64}


class _Tensor:
    def __init__(self, dtype):
        self.dtype = dtype


def test_two_autotune_keys_of_one_kernel_are_two_rounds():
    assert capture._round_id(_Autotuner(512, "bf16"), {}) != \
           capture._round_id(_Autotuner(1024, "bf16"), {})
    assert capture._round_id(_Autotuner(512, "bf16"), {}) != \
           capture._round_id(_Autotuner(512, "fp32"), {})


def test_the_same_key_is_the_same_round():
    assert capture._round_id(_Autotuner(512, "bf16"), {}) == \
           capture._round_id(_Autotuner(512, "bf16"), {})


def test_an_autotuner_that_answers_nothing_still_yields_an_id():
    """The round id is an identity, not a correctness gate: if triton's shape changes under us the
    worst case is two rounds sharing one id, which is the behaviour we already had."""
    assert capture._round_id(object(), {}) == ""
