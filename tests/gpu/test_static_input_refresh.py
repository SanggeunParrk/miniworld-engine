"""FoldForge R26: one sampler CUDA graph per shape, reused across folds (``miniworld_engine.capture.refresh_static_inputs``).

A small sampler step -- a token DiT stack (3 blocks) and an atom DiT stack (2 blocks), conditioning shared by the S = 5 samples (stride 0),
L = 128 tokens -- on static input tensors:

- fold A: inputs written into the static tensors, 2 eager warm-up steps, one capture (``static_weights()`` + ``static_inputs()``);
- fold B: written into the SAME tensors (``copy_``), ``refresh_static_inputs(model)``, replay: bitwise equal to eager on fold B;
- without the refresh the replay is NOT equal (it reads fold A's hoisted tables): the test sees the tables;
- the refresh captured into its own graph (replayed before the step graph) serves fold C the same way.

Token keys protenix (d_cond 384, pair 128), esmfold2 (768 / 256), boltz2 (768 / 128, up-gate), chai1, rosettafold3; atom keys protenix,
af3, boltz2, chai1 (conformer ids), intellifold2 (slid window), esmfold2_pair (rotary tables), each with a padded atom count too.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.gpu]

S, L = 5, 128


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _init(module):
    torch.manual_seed(0)
    with torch.no_grad():
        for p in module.parameters():
            torch.nn.init.normal_(p, std=0.05) if p.dim() > 1 else torch.nn.init.normal_(p, mean=0.3, std=0.2)
    return module


TOKEN = {  # key -> (d_cond, d_pair)
    "protenix": (384, 128),
    "esmfold2": (768, 256),
    "boltz2": (768, 128),
    "chai1": (384, 128),
    "rosettafold3": (384, 128),
}
ATOM_KEYS = ("protenix", "af3", "boltz2", "chai1", "intellifold2", "esmfold2_pair")


class Step(torch.nn.Module):
    """One sampler step on static inputs: the token stack and the atom stack read the fold's conditioning / pair / mask (one object each, as
    FoldForge's stacks pass them), the step's own state (``x_tok`` / ``x_atom``) changes per step."""

    def __init__(self, token_key: str, atom_key: str, n_atoms: int):
        super().__init__()
        from miniworld_engine.modules import AtomDiT, AtomDiTStack, TokenDiTStack

        dc, dp = TOKEN[token_key]
        self.token = _init(TokenDiTStack(token_key, 3, 768, dc, dp, 16, 2, backend="cuda")).float().cuda().eval().requires_grad_(False)
        self.atom_key = atom_key
        self.atom: Any
        if atom_key == "esmfold2_pair":
            blocks = [_init(AtomDiT("esmfold2_pair", backend="cuda")) for _ in range(2)]
            self.atom = torch.nn.ModuleList(blocks).float().cuda().eval().requires_grad_(False)
        else:
            kw: dict[str, Any] = {"key_window": "slide_qblock"} if atom_key == "intellifold2" else {}
            self.atom = _init(AtomDiTStack(atom_key, 2, backend="cuda", **kw)).float().cuda().eval().requires_grad_(False)
        self.dc, self.dp, self.n = dc, dp, n_atoms
        g = torch.Generator(device="cuda").manual_seed(123)
        # the static inputs (FoldForge: batch / embeddings / single_base / pair_cond / key_mask / atom_state)
        self.x_tok = torch.randn(S, 1, L, 768, device="cuda", generator=g).bfloat16()
        self.c_tok = torch.empty(1, 1, L, dc, device="cuda", dtype=torch.bfloat16)
        self.z_tok = torch.empty(1, L, L, dp, device="cuda", dtype=torch.bfloat16)
        self.m_tok = torch.ones(1, L, device="cuda", dtype=torch.bool)
        nwin, keys = (n_atoms + 31) // 32, (192 if atom_key == "esmfold2_pair" else 128)
        self.x_atom = torch.randn(S, 1, n_atoms, 128, device="cuda", generator=g)
        self.c_atom = torch.empty(1, 1, n_atoms, 128, device="cuda")
        self.z_atom = torch.empty(1, nwin, 32, keys, 16, device="cuda")
        self.m_atom = torch.ones(1, n_atoms, device="cuda", dtype=torch.bool)
        self.uid = torch.empty(1, n_atoms, device="cuda", dtype=torch.long)
        self.cos = torch.empty(1, n_atoms, 16, device="cuda")
        self.sin = torch.empty(1, n_atoms, 16, device="cuda")

    def write_fold(self, seed: int) -> None:
        """A new fold's inputs, written in place into the static tensors (``copy_``: same storage, the version counters move)."""
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.c_tok.copy_((torch.randn(self.c_tok.shape, device="cuda", generator=g) * 2 + 0.7).bfloat16())
        self.z_tok.copy_(torch.randn(self.z_tok.shape, device="cuda", generator=g).bfloat16())
        self.m_tok.fill_(True)
        self.m_tok[0, L - 3 - seed % 5 :] = False
        self.c_atom.copy_(torch.randn(self.c_atom.shape, device="cuda", generator=g))
        self.z_atom.copy_(torch.randn(self.z_atom.shape, device="cuda", generator=g))
        self.m_atom.fill_(True)
        self.m_atom[0, self.n - 5 - seed % 7 :] = False
        self.uid.copy_(torch.arange(self.n, device="cuda") // (17 + seed % 3))
        angle = 3 * torch.randn(1, self.n, 16, device="cuda", generator=g)
        self.cos.copy_(angle.cos())
        self.sin.copy_(angle.sin())

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        cond = self.c_tok.expand(S, 1, L, self.dc)  # shared by the samples, stride 0
        tok = self.token(self.x_tok, cond, self.z_tok, self.m_tok)
        acond = self.c_atom.expand(S, 1, self.n, 128)
        if self.atom_key == "esmfold2_pair":
            atom = self.x_atom
            for blk in self.atom:
                atom = blk(atom, acond, self.z_atom, self.m_atom, rope=(self.cos, self.sin))
        elif self.atom_key == "chai1":
            atom = self.atom(self.x_atom, acond, self.z_atom, self.m_atom, uid=self.uid)
        else:
            atom = self.atom(self.x_atom, acond, self.z_atom, self.m_atom)
        return tok, atom


def _capture(step: Step):
    """Fold A's warm-up (2 eager steps) and capture; returns (graph, its outputs, the StaticTables handle)."""
    from miniworld_engine.capture import static_inputs, static_weights

    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), static_weights(), static_inputs() as tables:
        with torch.cuda.stream(side):
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(side)
        with torch.cuda.graph(graph):
            out = step()
    return graph, out, tables


def _eager(step: Step):
    with torch.no_grad():
        tok, atom = step()
    return tok.clone(), atom.clone()


def _same(out, ref) -> bool:
    return all(torch.equal(a, b) for a, b in zip(out, ref, strict=True))


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) path")
@pytest.mark.parametrize(
    ("token_key", "atom_key", "n_atoms"),
    [
        ("protenix", "protenix", 128),
        ("protenix", "af3", 160),
        ("esmfold2", "esmfold2_pair", 256),
        ("boltz2", "boltz2", 128),
        ("chai1", "chai1", 128),
        ("rosettafold3", "intellifold2", 256),
        ("esmfold2", "protenix", 96),
        # N not a multiple of 128 (the kernels pad to 128): the hoisted tables are keyed on the caller's mask / uid, padded in the builds
        ("chai1", "chai1", 160),
        ("rosettafold3", "intellifold2", 224),
        ("esmfold2", "esmfold2_pair", 160),
        ("boltz2", "boltz2", 1000),
    ],
)
def test_refresh_then_replay_is_eager_on_the_new_fold(token_key, atom_key, n_atoms):
    from miniworld_engine.capture import clear_weight_caches, refresh_static_inputs, release_static_inputs

    step = Step(token_key, atom_key, n_atoms)
    try:
        step.write_fold(1)
        graph, out, tables = _capture(step)
        graph.replay()
        torch.cuda.synchronize()
        assert _same(out, _eager(step)), "fold A: replay != eager"
        assert len(tables) > 0, "the capture read no hoisted table from the eager cache"

        # fold B without the refresh: the replay reads fold A's tables
        step.write_fold(2)
        ref_b = _eager(step)
        graph.replay()
        torch.cuda.synchronize()
        assert not _same(out, ref_b), "the replay without a refresh matched fold B: the test does not see the hoisted tables"

        # with the refresh: bitwise eager on fold B (tables rewritten into the same storage)
        ptrs = [t.data_ptr() for t in out]
        n = refresh_static_inputs(step)
        graph.replay()
        torch.cuda.synchronize()
        assert n == len(tables)
        assert [t.data_ptr() for t in out] == ptrs
        assert _same(out, ref_b), "fold B: refresh + replay != eager"

        # the refresh captured into its own graph, replayed in front of the step graph: fold C
        rgraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(rgraph):
            refresh_static_inputs(step)
        step.write_fold(3)
        ref_c = _eager(step)
        rgraph.replay()
        graph.replay()
        torch.cuda.synchronize()
        assert _same(out, ref_c), "fold C: captured refresh + replay != eager"

        # a fresh capture on fold C gives the same outputs
        g2, out2, t2 = _capture(step)
        g2.replay()
        torch.cuda.synchronize()
        assert _same(out2, ref_c)
        del g2, rgraph
        release_static_inputs(tables=t2)
        assert release_static_inputs(tables=tables) == n
    finally:
        clear_weight_caches(step)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) path")
def test_refresh_refuses_changed_weights_and_reports_timing():
    from miniworld_engine.capture import StaleStaticInputs, clear_weight_caches, refresh_static_inputs

    step = Step("protenix", "protenix", 128)
    try:
        step.write_fold(1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        graph, out, _ = _capture(step)
        torch.cuda.synchronize()
        t_capture = time.perf_counter() - t0
        step.write_fold(2)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        refresh_static_inputs(step)
        graph.replay()
        torch.cuda.synchronize()
        t_refresh = time.perf_counter() - t0
        print(f"\nR26 small step: 2 warm-ups + capture {t_capture * 1e3:.1f} ms, refresh + replay {t_refresh * 1e3:.1f} ms")
        with torch.no_grad():
            block: Any = step.token.blocks[0]
            block.attention.to_bias.weight.mul_(1.5)  # an in-place weight update after the capture
        with pytest.raises(StaleStaticInputs):
            refresh_static_inputs(step)
        del graph, out
    finally:
        clear_weight_caches(step)


#: The kernels that make the hoisted atom tables (the windowed pair bias, the slid edge windows' bias): none may run in a replay that is
#: served the tables from the eager cache.
_TABLE_KERNELS = ("pair_bias", "edge_bias_fwd")


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) path")
@pytest.mark.parametrize("atom_key", ATOM_KEYS)
@pytest.mark.parametrize("n_atoms", [160, 1000])
def test_padded_n_replay_builds_no_tables(atom_key, n_atoms):
    """N not a multiple of 128: a second eager call hits the hoisted tables, and a step captured under ``static_inputs()`` reads every one
    from the eager cache (none made inside the capture) and runs none of their kernels at a replay -- bitwise equal to eager.

    The tables were keyed on a padded copy of the key mask / conformer ids made per call: every call missed, and every replay remade them."""
    from torch.profiler import ProfilerActivity, profile

    from miniworld_engine.capture import clear_weight_caches
    from miniworld_engine.kernels import _capture as cap
    from miniworld_engine.kernels.integrations import local_dit_esm, local_dit_h100

    step = Step("protenix", atom_key, n_atoms)
    stores = (
        [local_dit_esm._HOIST_COND, local_dit_esm._HOIST_BIAS]
        if atom_key == "esmfold2_pair"
        else [local_dit_h100._HOIST_COND, local_dit_h100._HOIST_BIAS, local_dit_h100._HOIST_EDGES]
    )
    slid = atom_key in ("af3", "intellifold2")  # af3: AtomDiTOpenFold3's default slid window
    per_block = 3 if slid else 2
    try:
        step.write_fold(1)
        with torch.no_grad():
            ref = step()[1].clone()
            sizes = [len(s) for s in stores]
            again = step()[1]
        assert [len(s) for s in stores] == sizes, "a second eager call made new hoisted tables"
        assert torch.equal(again, ref)
        graph, out, tables = _capture(step)
        built = {k for k in cap._CAPTURE_BUILT if k[0] in {id(s) for s in stores}}
        assert not built, f"{len(built)} atom tables were made inside the capture (every replay remakes them)"
        atom_tables = sum(1 for k in tables._keys if k[0] in {id(s) for s in stores})
        assert atom_tables == 2 * per_block, f"the capture read {atom_tables} atom tables from the eager cache, expected {2 * per_block}"
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            graph.replay()
            torch.cuda.synchronize()
        names = [e.name for e in (prof.events() or []) if str(e.device_type).endswith("CUDA")]
        assert names, "the profiler saw no kernel of the replay"
        made = [n for n in names if any(t in n for t in _TABLE_KERNELS)]
        assert not made, f"a replay ran table kernels: {sorted(set(made))}"
        assert torch.equal(out[1], ref), "replay != eager"
        del graph, out
    finally:
        clear_weight_caches(step)
