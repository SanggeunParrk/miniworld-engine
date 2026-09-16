"""Default build dimensions belong to released model configurations."""
from miniworld_engine.autotune.module_registry import module_rows


def test_every_row_names_a_model_contract():
    for row in module_rows():
        assert row.source.strip()
        assert "headroom" not in row.source.lower()
        assert "kernel probe" not in row.source.lower()


def test_real_wide_pairs_remain_without_hypothetical_pair768():
    rows = module_rows()
    assert {r.dims["d_pair"] for r in rows if r.module == "triangle_multiplication"} == {64, 128, 256, 384}
    assert not any(r.stream == "token_pair" and r.dims.get("d_norm") == 768 for r in rows)
