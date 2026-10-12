"""Transition LN's current key distinguishes every production channel width."""
from miniworld_engine.autotune import builder, width_evidence


def test_transition_ln_production_widths_are_not_collapsed():
    op = "layernorm_bwd_foldstats_triton"
    widths = (128, 256, 384, 512, 768)
    # The former evidence file mapped all five to the same old key, silently
    # dropping the K=128 model launch from the cache build.
    assert width_evidence.distinct_widths(op, widths) == widths
    units = builder.op_units({op})
    pair_widths = {u.width for u in units if u.side == "pair" and u.length == 384}
    assert {128, 256, 384, 768} <= pair_widths
