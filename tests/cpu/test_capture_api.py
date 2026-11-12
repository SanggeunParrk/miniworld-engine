"""R1 (FoldForge): the public capture / weight-cache API ``miniworld_engine.capture``.

- ``static_weights`` / ``static_inputs`` are the context managers of ``kernels._capture`` (same flags, restored on exit);
- ``clear_weight_caches`` empties every registered store (a weight replaced without a version bump is picked up after it) and the
  per-instance cache attributes of the modules it is given;
- every module-level dict in the package's sources is either a registered weight cache (``_capture.weight_cache``) or listed below as
  not weight-derived (so a new cache cannot be forgotten by ``clear_weight_caches``).
"""

from __future__ import annotations

import ast
import gc
import importlib
import weakref
from pathlib import Path

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from miniworld_engine import capture
from miniworld_engine.kernels import _capture

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "miniworld_engine"

#: module-level dicts that are NOT derived from weights or per-item inputs ("<path under the package>:<name>" -> what they hold)
NOT_WEIGHT_DERIVED = {
    "kernels/_compile.py:_REGISTERED": "opaque op registry",
    "kernels/_codegraph.py:_FILE_CACHE": "parsed source files",
    "kernels/_codegraph.py:_OVERLAY": "source overlays",
    "kernels/_pretune.py:_AXES": "tuning axes per shape key",
    "kernels/_pretune.py:_BASE_NAME": "tuning bucket names",
    "kernels/_pretune.py:_SQUEUE": "tuning queue",
    "kernels/_capture.py:_PINNED": "pinned keys of the registered stores (cleared with them)",
    "kernels/_capture.py:_INPUT_STORES": "the per-input stores lookup_inputs filled (registered ones) and their alive tests",
    "kernels/_capture.py:_HELD_BY_CAPTURES": "the forms capture_reads_eager captures read (held for the process: what live graphs read)",
    "kernels/_capture.py:_HELD_UNOWNED": "cache entries captures read with no static_inputs() handle open (cleared with the stores)",
    "kernels/_capture.py:_WEIGHT_CACHES": "the registry itself",
    "kernels/_capture.py:_FORMS": 'registered below its definition: weight_cache("capture.forms", _FORMS)',
    "kernels/_capture.py:_LINEAGES": 'registered below its definition: weight_cache("capture.input_lineages", _LINEAGES)',
    "kernels/_nvcc.py:_KEY_OF": "extension build keys",
    "kernels/_nvcc.py:_LOADED": "loaded extensions",
    "kernels/augmented_attention/cuda/sm100/__init__.py:_apb_runs": "TMA launches keyed by buffer addresses (hold no tensor)",
    "kernels/augmented_attention/cuda/sm100_atom/__init__.py:_TMAPS32": "TMA descriptors keyed by address and geometry",
    "kernels/augmented_attention/cuda/sm100_atom_local/__init__.py:_TMAPS": "TMA descriptors keyed by address and geometry",
    "kernels/bias_only_dit/cuda/bwd_fused.py:_MID": "kernel objects per device / shape",
    "kernels/bias_only_dit/cuda/bwd_fused.py:_OPS": "kernel objects per device / shape",
    "kernels/bias_only_dit/cuda/tf32.py:_GLU_OPS": "kernel objects per device",
    "kernels/bias_only_dit/cuda/tf32.py:_MAXC": "occupancy per device",
    "kernels/integrations/pwa_wide.py:_EXT": "extensions",
    "kernels/integrations/opm_train.py:_EXT": "extensions",
    "kernels/integrations/pwa_train.py:_EXT": "extensions",
    "kernels/swa_dit/cuda/loader.py:_EXT": "extensions",
    "kernels/swa_dit/cuda/loader.py:ERRORS": "extension build errors",
    "kernels/integrations/adaln_sm80.py:_SIDE": "side streams",
    "kernels/integrations/attention_pair_bias_sm80.py:_SIDE": "side streams",
    "kernels/integrations/pwa_wide_sm80.py:_SIDE": "side streams",
    "kernels/trimul_inproj/cuda/sm80.py:_SIDE": "side streams",
    "kernels/transition/cuda/fused_generic_sm90a.py:_SIDE_STREAMS": "side streams",
    "kernels/transition/cuda/fused_fwd_sm80.py:_SMS": "SM count per device",
    "kernels/transition/cuda/fused_wide_sm80.py:_weight_streams": "side streams",
    "kernels/conditioned_transition/cuda/__init__.py:_TOKEN_STREAMS": "side streams",
    "kernels/conditioned_transition/cuda/__init__.py:_COND_STREAMS": "side streams",
    "kernels/conditioned_transition/cuda/__init__.py:_COND_PIPE": "per-input (conditioning) events and grouped GEMM outputs, capture-scoped",
    "kernels/integrations/local_dit_h100.py:_CONST": "constant tensors per shape / device",
    "kernels/integrations/local_dit_esm.py:_CONST": "constant vectors per value / width / device",
    "kernels/integrations/token_dit.py:_CONST": "constant tensors per shape / device",
    "kernels/integrations/augattn_sm80.py:_ZEROS": "zero tensors per shape / device",
    "kernels/integrations/attention_pair_bias_sm80.py:_zeros": "zero tensors per shape / dtype / device",
    "kernels/integrations/bias_only_dit.py:_ALL_TRUE": "all-true masks per length / device",
    "kernels/integrations/token_dit.py:_ALL_TRUE": "all-true masks per length / device",
    "kernels/integrations/token_dit_train.py:_GSW": "GEMM objects per device",
    "kernels/integrations/bias_only_dit_train.py:_OPS": "kernel objects per device / shape",
    "kernels/integrations/bias_only_dit_train.py:_PLANS": "gradient split plans per parameter shapes",
    "kernels/integrations/bias_only_dit_train.py:_OPS32": "kernel objects per device / shape",
    "kernels/integrations/attention_pair_bias_b200.py:_cores": "kernel objects per device / shape",
    "kernels/trimul_inproj/cuda/_h100_launch.py:_fixed_layouts": "struct layouts",
    "kernels/trimul_inproj/cuda/_h100_launch.py:_UNITS": "loaded kernel units",
    "kernels/trimul_inproj/cuda/_h100_launch.py:_BLOCK_DRIVERS": "drivers per device",
    "modules/norm.py:_FASTLN_STATE": "fast LayerNorm availability",
    "modules/_cueq.py:_STATE": "cuEquivariance availability",
    "modules/triangle.py:_PAIR_FUSED_REFUSED": "refused library cells",
}

#: directories whose sources are not engine runtime caches (measurement state, the judge)
SKIP = ("judge/", "measure/")


def _empty_dict(node: ast.expr | None) -> bool:
    if isinstance(node, ast.Dict):
        return not node.keys
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict" and not node.args and not node.keywords


def _module_level_dicts() -> set[str]:
    found = set()
    for path in PACKAGE.rglob("*.py"):
        rel = path.relative_to(PACKAGE).as_posix()
        if rel.startswith(SKIP):
            continue
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and _empty_dict(node.value):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and _empty_dict(node.value) and isinstance(node.target, ast.Name):
                names = [node.target.id]
            else:
                continue
            found.update(f"{rel}:{n}" for n in names)
    return found


def test_every_module_level_dict_is_registered_or_listed():
    unlisted = sorted(_module_level_dicts() - set(NOT_WEIGHT_DERIVED))
    assert not unlisted, (
        "module-level dicts that are neither registered weight caches (``_capture.weight_cache(name)``) nor listed in "
        f"NOT_WEIGHT_DERIVED: {unlisted}"
    )
    stale = sorted(set(NOT_WEIGHT_DERIVED) - _module_level_dicts())
    assert not stale, f"NOT_WEIGHT_DERIVED names dicts that no longer exist: {stale}"


def test_public_names():
    assert capture.static_weights is _capture.static_weights
    assert capture.static_inputs is _capture.static_inputs
    assert capture.clear_weight_caches is _capture.clear_weight_caches
    assert capture.refresh_static_inputs is _capture.refresh_static_inputs
    assert set(capture.__all__) == {
        "StaleStaticInputs",
        "StaticTables",
        "clear_weight_caches",
        "refresh_static_inputs",
        "release_static_inputs",
        "signature",
        "static_inputs",
        "static_weights",
        "weight_caches",
    }


def test_static_contexts_set_and_restore():
    assert not _capture._STATIC[0] and not _capture._STATIC_INPUTS[0]
    with capture.static_weights():
        assert _capture._STATIC[0]
        with capture.static_inputs():
            assert _capture._STATIC_INPUTS[0]
        assert not _capture._STATIC_INPUTS[0]
    assert not _capture._STATIC[0]


def test_registered_stores_include_the_known_ones():
    for name in ("miniworld_engine.kernels._tf32", "miniworld_engine.kernels.integrations.token_dit"):
        importlib.import_module(name)  # a store registers itself at import
    names = capture.weight_caches()
    assert names["capture.forms"] is _capture._FORMS
    assert "tf32.operands" in names
    assert "token_dit.runners" in names


def test_param_form_replaced_weight_picked_up_after_clear():
    """A weight rewritten without a version bump (``.data``: a new tensor at the old address is the same case) is served the old form
    until ``clear_weight_caches``; after it the new values are."""
    w = torch.ones(4)
    form = lambda: _capture.param_forms("test_capture_api", (w,), lambda: w * 2)
    assert torch.equal(form(), torch.full((4,), 2.0))
    v = w._version
    w.data.fill_(3.0)  # bypasses the version counter
    assert w._version == v
    assert torch.equal(form(), torch.full((4,), 2.0))  # the cached form: stale
    capture.clear_weight_caches()
    assert torch.equal(form(), torch.full((4,), 6.0))


def test_lookup_store_cleared():
    store = _capture.weight_cache("test_capture_api.store")
    try:
        w = torch.ones(2)
        key = (w.data_ptr(), w._version)
        assert torch.equal(_capture.lookup(store, key, lambda: w + 1), torch.full((2,), 2.0))
        w.data.fill_(5.0)
        assert torch.equal(_capture.lookup(store, key, lambda: w + 1), torch.full((2,), 2.0))  # stale hit
        capture.clear_weight_caches()
        assert not store
        assert torch.equal(_capture.lookup(store, key, lambda: w + 1), torch.full((2,), 6.0))
    finally:
        _capture._WEIGHT_CACHES.pop("test_capture_api.store", None)


def test_tf32_operand_cache_cleared():
    """The TF32 operand cache (``_tf32.tf32_cached``: CUDA only) is a registered store: ``clear_weight_caches`` empties it."""
    from miniworld_engine.kernels import _tf32

    assert capture.weight_caches()["tf32.operands"] is _tf32._CACHE
    _tf32._CACHE[("test_capture_api",)] = (torch.ones(1),)
    capture.clear_weight_caches()
    assert not _tf32._CACHE


def test_instance_cache_attributes_cleared():
    from miniworld_engine.modules import TokenDiT

    block = TokenDiT("protenix", 64, 32, 16, 4, 2)
    vars(block.attention)["_anthropic_pack"] = ("key", None, None)
    vars(block.attention)["_anthropic_params"] = ("key", None, None)
    capture.clear_weight_caches(block)
    assert "_anthropic_pack" not in block.attention.__dict__
    assert "_anthropic_params" not in block.attention.__dict__


def test_refresh_records_a_capture_read_and_rewrites_in_place(monkeypatch):
    """R26 on the CPU (the capture faked): a hit served to a capture under ``static_inputs()`` is recorded with its build; a refresh writes the
    build from the input's current contents into the recorded table (same storage); a changed weight raises ``StaleStaticInputs``."""
    store = _capture.weight_cache("test_capture_api.refresh")
    try:
        owner = torch.nn.Linear(2, 2)
        x = torch.ones(4)
        build = lambda: (x * 2, 7)
        key = (x.data_ptr(), x._version)
        with capture.static_weights(), capture.static_inputs() as tables:
            table = _capture.lookup_inputs(store, key, build, inputs=(x,))[0]  # the eager warm-up
            monkeypatch.setattr(_capture, "_graph_capture_id", lambda: 1)  # "inside a capture"
            _capture.CALLER[0] = owner
            assert _capture.lookup_inputs(store, key, build, inputs=(x,))[0] is table
            _capture.CALLER[0] = None
            monkeypatch.undo()
        assert len(tables) == 1
        ptr = table.data_ptr()
        x.copy_(torch.arange(4.0))  # the next item, written in place
        assert capture.refresh_static_inputs(torch.nn.Sequential(torch.nn.Identity())) == 0  # another root: not its table
        assert capture.refresh_static_inputs(owner) == 1
        assert table.data_ptr() == ptr and torch.equal(table, torch.arange(4.0) * 2)
        assert capture.signature(owner, x) == capture.signature(owner, x.clone())
        with torch.no_grad():
            owner.weight.add_(1)
        try:
            capture.refresh_static_inputs(owner)
            raise AssertionError("a changed weight must raise")
        except capture.StaleStaticInputs:
            pass
        assert tables.release() == 1 and len(tables) == 0
    finally:
        _capture._WEIGHT_CACHES.pop("test_capture_api.refresh", None)
        _capture.CALLER[0] = None


def _record_one(store, monkeypatch, x: torch.Tensor, owner: torch.nn.Module, keep: bool = True):
    """One refresh record of ``store`` over the input ``x`` (an eager warm-up, then a hit inside a faked capture), as a sampler warm-up +
    capture under ``static_inputs()`` makes it; returns the table and the handle (None when ``keep`` is off: the handle is dropped)."""
    key = (x.data_ptr(), x._version)

    def build():
        return (x * 2, 7, StorageWeakRef(x.untyped_storage()))

    with capture.static_weights(), capture.static_inputs() as tables:
        table = _capture.lookup_inputs(store, key, build, inputs=(x,), alive=lambda e: not e[2].expired())[0]
        monkeypatch.setattr(_capture, "_graph_capture_id", lambda: 1)  # "inside a capture"
        _capture.CALLER[0] = owner
        assert _capture.lookup_inputs(store, key, build, inputs=(x,), alive=lambda e: not e[2].expired())[0] is table
        _capture.CALLER[0] = None
        monkeypatch.undo()
    assert len(tables) == 1
    return table, (tables if keep else None)


def test_refresh_records_go_with_their_handle(monkeypatch):
    """FoldForge 5I28 OpenDDE n = 16: a refresh record holds its input (``inputs``, the build closure) and its table. The records belong to
    the ``static_inputs()`` handle: a caller that captures a new graph per item and drops the handle (FoldForge) leaves nothing that keeps the
    item alive once it drops the input itself (the table entry and the input's storage go at ``release_dead_inputs``); while the handle is
    kept a refresh rewrites the table in place; a refresh of a module whose records went with a dropped handle raises; a record made with no
    handle open (``set_static_inputs``) is held until released."""
    store = _capture.weight_cache("test_capture_api.records")
    owner = torch.nn.Linear(2, 2)
    try:
        # the handle dropped, the caller still holding its input (the graph is replayed): the table entry stays, the record is gone
        x = torch.ones(4)
        held = StorageWeakRef(x.untyped_storage())
        table, _ = _record_one(store, monkeypatch, x, owner, keep=False)
        assert not _capture._READS or all(r() is None for r in _capture._READS.values())
        _capture.release_dead_inputs()
        assert len(store) == 1 and store[next(iter(store))][0] is table
        # a refresh would rewrite nothing: it raises
        try:
            capture.refresh_static_inputs(owner)
            raise AssertionError("a refresh of tables whose records went with their handle must raise")
        except capture.StaleStaticInputs:
            pass
        # the caller drops the item: the entry, the table and the input's storage go
        del x, table
        _capture.release_dead_inputs()
        assert held.expired() and not store

        # the handle kept: a new capture of the module clears the mark; the refresh rewrites in place
        x = torch.ones(4)
        table, tables = _record_one(store, monkeypatch, x, owner)
        x.copy_(torch.arange(4.0))
        assert capture.refresh_static_inputs(owner) == 1 and torch.equal(table, torch.arange(4.0) * 2)
        assert tables is not None and tables.release() == 1 and len(tables) == 0
        del tables

        # no handle open (set_static_inputs alone): the record is held until released
        x2 = torch.ones(4)
        prev_w, prev_i = _capture.set_static_weights(True), _capture.set_static_inputs(True)
        try:
            key = (x2.data_ptr(), x2._version)
            build = lambda: (x2 * 3, 7, StorageWeakRef(x2.untyped_storage()))
            t2 = _capture.lookup_inputs(store, key, build, inputs=(x2,))[0]
            monkeypatch.setattr(_capture, "_graph_capture_id", lambda: 1)
            _capture.CALLER[0] = owner
            assert _capture.lookup_inputs(store, key, build, inputs=(x2,))[0] is t2
            _capture.CALLER[0] = None
            monkeypatch.undo()
        finally:
            _capture.set_static_weights(prev_w)
            _capture.set_static_inputs(prev_i)
        x2.fill_(2.0)
        assert capture.refresh_static_inputs(owner) == 1 and torch.equal(t2, torch.full((4,), 6.0))
        assert capture.release_static_inputs(owner) == 1 and not _capture._UNOWNED
    finally:
        _capture._WEIGHT_CACHES.pop("test_capture_api.records", None)
        _capture.CALLER[0] = None


def test_lineage_replaces_the_entry_of_an_earlier_version(monkeypatch):
    """An input written in place (a new version) misses its old key; with ``lineage`` the new entry replaces the old one instead of both
    staying while the input lives (FoldForge 5I28 OpenDDE n = 16: two versions' 24 token-DiT pair biases, 2 x 11.3 GiB). An entry a capture
    under ``static_inputs()`` read stays (that graph may replay it)."""
    store = _capture.weight_cache("test_capture_api.lineage")
    try:
        x = torch.ones(4)

        def call():
            key, lineage = (x.data_ptr(), x._version), (x.data_ptr(),)
            return _capture.lookup_inputs(
                store, key, lambda: (x * 2, StorageWeakRef(x.untyped_storage())), alive=lambda e: not e[1].expired(), lineage=lineage
            )[0]

        first = call()
        assert call() is first and len(store) == 1
        x.add_(1)  # in place: a new version
        second = call()
        assert second is not first and torch.equal(second, torch.full((4,), 4.0)) and len(store) == 1

        # read by a capture under static_inputs(): kept when the input moves on
        with capture.static_weights(), capture.static_inputs() as tables:
            monkeypatch.setattr(_capture, "_graph_capture_id", lambda: 1)
            assert call() is second
            monkeypatch.undo()
        x.add_(1)
        third = call()
        assert third is not second and len(store) == 2
        del tables
    finally:
        _capture._WEIGHT_CACHES.pop("test_capture_api.lineage", None)


def test_capture_reads_are_held_by_the_handle(monkeypatch):
    """What a capture reads stays allocated while the graph can be replayed (``_capture.hold``): an entry served to a capture under a
    ``static_inputs()`` handle is held by the handle -- the store evicting it (another shape's entries: a count limit) frees nothing the
    graph reads -- and goes with the handle's release; with no handle open (``static_weights()`` alone) an eager entry is held
    process-wide until ``clear_weight_caches``; outside a capture nothing is held."""
    store = _capture.weight_cache("test_capture_api.holds")
    try:
        w = torch.ones(4)
        build = lambda: w * 2
        with capture.static_weights(), capture.static_inputs() as tables:
            pack = _capture.lookup(store, ("a",), build, limit=1)  # the eager warm-up
            assert not tables.held()
            monkeypatch.setattr(_capture, "_graph_capture_id", lambda: 1)  # "inside a capture"
            assert _capture.lookup(store, ("a",), build, limit=1) is pack
            monkeypatch.undo()
        assert any(h is pack for h in tables.held())
        ref = weakref.ref(pack)
        del pack
        _capture._PINNED.clear()  # (the store's own pin: evictable from here on)
        _capture.lookup(store, ("b",), build, limit=1)  # another shape's entry evicts it
        assert ("a",) not in {k[1] for k in store}
        assert ref() is not None, "the handle must keep what its capture read"
        tables.release()
        assert ref() is None and not tables.held()

        # the handle dropped: its holds go with it
        with capture.static_weights(), capture.static_inputs():
            monkeypatch.setattr(_capture, "_graph_capture_id", lambda: 1)
            ref = weakref.ref(_capture.lookup(store, ("c",), build, limit=8))
            monkeypatch.undo()
        store.clear()
        gc.collect()
        assert ref() is None

        # no static_inputs() handle: held process-wide until clear_weight_caches
        with capture.static_weights():
            entry = _capture.lookup(store, ("d",), build, limit=8)
            monkeypatch.setattr(_capture, "_graph_capture_id", lambda: 1)
            assert _capture.lookup(store, ("d",), build, limit=8) is entry
            monkeypatch.undo()
        assert any(h is entry for h in _capture._HELD_UNOWNED.values())
        ref = weakref.ref(entry)
        del entry
        store.clear()
        assert ref() is not None
        capture.clear_weight_caches()
        assert ref() is None and not _capture._HELD_UNOWNED

        # outside a capture: nothing held
        _capture.lookup(store, ("e",), build)
        assert not _capture._HELD_UNOWNED
    finally:
        _capture._WEIGHT_CACHES.pop("test_capture_api.holds", None)
        _capture._HELD_UNOWNED.clear()
