"""``dev bundle-kernels``' coverage driver reaches every extension a first call can build: one case per kind, each kind handled."""
import inspect

from miniworld_engine.build import bundle_coverage


def test_every_listed_kind_has_a_runner():
    kinds = {c["kind"] for c in bundle_coverage.cases() if c["kind"] != "case"}
    source = inspect.getsource(bundle_coverage._run_block)
    for kind in kinds:
        assert f'"{kind}"' in source, f"bundle_coverage lists kind {kind!r} but _run_block has no branch for it"


def test_the_extensions_without_shape_keys_are_covered():
    kinds = {c["kind"] for c in bundle_coverage.cases()}
    # model variants, sampler glue, the swa_dit stage extensions, and the Anthropic OPM epilogue (JIT-built as `miniworld_opm_epilogue`)
    assert {"model_variants", "sampler_ops", "swa_stages", "anthropic_ext"} <= kinds
