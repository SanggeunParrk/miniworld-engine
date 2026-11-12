"""FoldForge R26 memory safety: a graph captured at one shape, kept while graphs of other shapes are captured, then refreshed and replayed.

FoldForge keeps one sampler graph per token shape (``static_weights()`` + ``static_inputs() as tables``) and serves later folds of that
shape by writing them into the graph's static tensors, ``tables.refresh(model)`` and a replay. The graph reads engine cache entries that
its warm-up made eagerly (weight packs, the token DiT's grouped conditioning weights, the hoisted per-item tables, the APB packs). Capturing
another shape makes and evicts entries of the same caches (the conditioning chunks are an LRU of 4, other stores replace or prune), and an
evicted entry's memory went back to the allocator -- and, after ``torch.cuda.empty_cache()``, to the driver -- while the first graph still
read it: an illegal memory access, NaNs or (with the memory reused) silently wrong outputs (engine 507d91366, FoldForge protenix1 /
boltz2 at n = 4). The engine now holds every entry a capture reads for as long as the ``static_inputs()`` handle open at the capture
(``kernels/_capture.hold``).

Per stack (protenix1-like: Protenix token / atom keys; boltz2-like: Boltz-2's up-gate token block and atom key; each with an
AttentionPairBias on the token pair), S = 5 samples:

1. capture at L = 128 (fold A), keep the graph and its handle;
2. capture at L = 256 (and, for the LRU test, 384 / 512 / 640), then ``torch.cuda.empty_cache()``;
3. allocate and fill garbage (NaN) tensors over the freed memory;
4. write fold B into the L = 128 static tensors, refresh, replay: bitwise equal to eager on fold B.
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch

pytestmark = [pytest.mark.gpu]

S = 5
BF = torch.bfloat16
BLOCKS = 24  # the models' depth: the token DiT's conditioning chunks (``_COND_CHUNK_BYTES`` / rows) differ between the shapes

STACKS = {  # name -> (token key, d_cond, d_pair, atom key)
    "protenix1": ("protenix", 384, 128, "protenix"),
    "boltz2": ("boltz2", 768, 128, "boltz2"),
}


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _init(module):
    torch.manual_seed(0)
    with torch.no_grad():
        for p in module.parameters():
            torch.nn.init.normal_(p, std=0.05) if p.dim() > 1 else torch.nn.init.normal_(p, mean=0.3, std=0.2)
    return module.float().cuda().eval().requires_grad_(False)


class Model(torch.nn.Module):
    """The modules of a sampler step, shared by every shape's graph (one set of weights, as FoldForge's model)."""

    def __init__(self, stack: str):
        super().__init__()
        from miniworld_engine.modules import AtomDiTStack, AttentionPairBias, TokenDiTStack

        tkey, dc, dp, akey = STACKS[stack]
        self.dc, self.dp = dc, dp
        self.token = _init(TokenDiTStack(tkey, BLOCKS, 768, dc, dp, 16, 2, backend="cuda"))
        self.atom = _init(AtomDiTStack(akey, 2, backend="cuda"))
        self.apb = _init(AttentionPairBias(384, dp, 16, master_weights=True, backend="cuda"))


class Step(torch.nn.Module):
    """One sampler step at token length ``L`` on its own static tensors (FoldForge's static clone of a fold)."""

    def __init__(self, model: Model, L: int):
        super().__init__()
        self.model, self.L, self.n = model, L, L + 32  # atoms: not a multiple of 128 (the atom tables pad)
        dc, dp, n = model.dc, model.dp, self.n
        dev = "cuda"
        self.x_tok = torch.empty(S, 1, L, 768, device=dev, dtype=BF)
        self.c_tok = torch.empty(1, 1, L, dc, device=dev, dtype=BF)
        self.z_tok = torch.empty(1, L, L, dp, device=dev, dtype=BF)
        self.m_tok = torch.ones(1, L, device=dev, dtype=torch.bool)
        self.s_apb = torch.empty(1, L, 384, device=dev, dtype=BF)
        self.x_atom = torch.empty(S, 1, n, 128, device=dev)
        self.c_atom = torch.empty(1, 1, n, 128, device=dev)
        self.z_atom = torch.empty(1, (n + 31) // 32, 32, 128, 16, device=dev)
        self.m_atom = torch.ones(1, n, device=dev, dtype=torch.bool)

    def write_fold(self, seed: int) -> None:
        """A fold's inputs, written in place (``copy_``: same storage, the versions move)."""
        g = torch.Generator(device="cuda").manual_seed(seed)
        r = lambda t, scale=1.0, shift=0.0: (torch.randn(t.shape, device="cuda", generator=g) * scale + shift).to(t.dtype)
        self.x_tok.copy_(r(self.x_tok))
        self.c_tok.copy_(r(self.c_tok, 2.0, 0.7))
        self.z_tok.copy_(r(self.z_tok))
        self.m_tok.fill_(True)
        self.m_tok[0, self.L - 3 - seed % 5 :] = False
        self.s_apb.copy_(r(self.s_apb))
        self.x_atom.copy_(r(self.x_atom))
        self.c_atom.copy_(r(self.c_atom))
        self.z_atom.copy_(r(self.z_atom))
        self.m_atom.fill_(True)
        self.m_atom[0, self.n - 5 - seed % 7 :] = False

    def forward(self) -> tuple[torch.Tensor, ...]:
        m, L = self.model, self.L
        tok = m.token(self.x_tok, self.c_tok.expand(S, 1, L, m.dc), self.z_tok, self.m_tok)
        atom = m.atom(self.x_atom, self.c_atom.expand(S, 1, self.n, 128), self.z_atom, self.m_atom)
        apb = m.apb(self.s_apb, self.z_tok, self.m_tok)
        return tok, atom, apb


class Captured:
    """A step's graph, its outputs and the ``static_inputs()`` handle (kept together, as FoldForge's per-shape cache entry)."""

    def __init__(self, step: Step, seed: int):
        from miniworld_engine.capture import static_inputs, static_weights

        self.step = step
        step.write_fold(seed)
        self.graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), static_weights(), static_inputs() as tables:
            with torch.cuda.stream(side):
                for _ in range(2):
                    step()
            torch.cuda.current_stream().wait_stream(side)
            with torch.cuda.graph(self.graph):
                self.out = step()
        self.tables = tables
        torch.cuda.synchronize()


def _eager(step: Step) -> tuple[torch.Tensor, ...]:
    with torch.no_grad():
        return tuple(t.clone() for t in step())


def _garbage(keep: int = 6 << 30) -> list[torch.Tensor]:
    """Allocations of mixed sizes (2 - 128 MiB), filled with NaN, over the memory the allocator just released (``empty_cache``): the device
    is filled up to ``keep`` bytes free, so a freed block a graph still reads is mapped again and overwritten (or unmapped: a fault)."""
    held, sizes, i = [], [2 << 20, 6 << 20, 14 << 20, 30 << 20, 62 << 20, 126 << 20], 0
    while torch.cuda.mem_get_info()[0] > keep + sizes[-1]:
        held.append(torch.full((sizes[i % len(sizes)] // 4,), float("nan"), device="cuda"))
        i += 1
    torch.cuda.synchronize()
    return held


def _mismatch(out, ref) -> list[str]:
    names = ("token", "atom", "apb")
    return [
        f"{n}: max |d| {(a.float() - b.float()).abs().max().item()}"
        for n, a, b in zip(names, out, ref, strict=True)
        if not torch.equal(a, b)
    ]


def _replay_new_fold(c: Captured, seed: int) -> tuple[torch.Tensor, ...]:
    """Fold ``seed`` into the graph's static tensors, refresh, replay; returns copies of the outputs."""
    c.step.write_fold(seed)
    assert c.tables.refresh(c.step.model) > 0
    c.graph.replay()
    torch.cuda.synchronize()
    return tuple(t.clone() for t in c.out)


def _run(stack: str, lengths: tuple[int, ...]) -> None:
    from miniworld_engine.capture import clear_weight_caches

    model = Model(stack)
    graphs: list[Captured] = []
    try:
        for i, L in enumerate(lengths):
            graphs.append(Captured(Step(model, L), seed=1 + i))
            graphs[-1].graph.replay()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        junk = _garbage()
        # the first shape's graph -- and the later ones (each was captured with the caches the others then changed)
        got = [_replay_new_fold(c, 7 + i) for i, c in enumerate(graphs)]
        del junk
        for L, c, out in zip(lengths, graphs, got, strict=True):
            bad = _mismatch(out, _eager(c.step))
            assert not bad, f"L = {L}: refresh + replay after the captures of {lengths} != eager: {bad}"
    finally:
        for c in graphs:
            c.tables.release()
        graphs.clear()
        clear_weight_caches(model)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) path")
@pytest.mark.parametrize("stack", sorted(STACKS))
def test_first_shape_replays_after_a_second_capture(stack):
    _run(stack, (128, 256))


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) path")
@pytest.mark.parametrize("stack", sorted(STACKS))
def test_first_shape_replays_after_lru_eviction(stack):
    """Five shapes: the conditioning chunks (an LRU of 4) of the first are evicted several times over."""
    _run(stack, (128, 256, 384, 512, 640))


def test_holds_go_with_the_handle():
    """The holds live exactly as long as the handle: the capture's reads (the conditioning chunks among them) are held by it, nothing is
    held process-wide, and ``release()`` lets them go -- the chunks another shape's capture evicted are then freed."""
    from miniworld_engine.capture import clear_weight_caches
    from miniworld_engine.kernels import _capture
    from miniworld_engine.kernels.conditioned_transition import cuda as ct

    model = Model("boltz2")
    try:
        unowned = len(_capture._HELD_UNOWNED)
        first = Captured(Step(model, 128), seed=1)
        chunks = [weakref.ref(w) for w in ct._COND_CHUNKS.values()]
        assert chunks, "the token stack made no conditioning chunk"
        held = {id(h) for h in first.tables.held()}
        assert all(id(r()) in held for r in chunks), "the handle does not hold the chunks its capture read"
        assert len(_capture._HELD_UNOWNED) == unowned, "a capture under a static_inputs() handle held entries process-wide"
        second = Captured(Step(model, 256), seed=2)
        assert all(r() is not None for r in chunks), "the second capture freed a chunk the first graph reads"
        assert not any(r() is v for r in chunks for v in ct._COND_CHUNKS.values()), "the second shape did not evict the first one's chunks"
        del second
        first.tables.release()
        assert not first.tables.held()
        gc.collect()
        assert all(r() is None for r in chunks), "an evicted chunk outlived the released handle"
    finally:
        clear_weight_caches(model)
