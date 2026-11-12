"""FoldForge R26 on the grouped multi-fold sampler layout (FoldForge R18, ``foldforge.models.grouped.GroupedSampler``): one sampler CUDA
graph per shape, reused across fold sets (``miniworld_engine.capture.refresh_static_inputs``).

The grouped step runs the DiT blocks once over ``[S, F, ...]``: F folds (trunk_seeds) x S samples, each fold with its own pair, its own
conditioning (one per fold, ``[1, F, ., d]`` expanded stride 0 over the S samples) and its own atom pair; one key mask expanded over the
folds. Inputs from the blocks' ``make_inputs`` on the requirement streams ``token_folds_shared_cond`` / ``atom_folds_shared_cond``
(depth = F = 5), bf16 activations over fp32 parameters, L = 128, a 3-block token stack and a 2-block atom stack:

- fold set A: written into the static tensors, 2 eager warm-ups, one capture (``static_weights()`` + ``static_inputs()``);
- fold set B written into the SAME tensors (``copy_``): the replay without a refresh is NOT eager(B) (the grouped tables -- the token
  pair bias ``[H, F, L, L]``, the atom conditioning tables of the F folds, the atom pair bias ``[F, ...]`` -- are the graph's); the refresh
  + replay is eager(B) bitwise, output pointers unchanged; a fresh capture on B gives the same outputs;
- a change of the fold count: ``signature`` differs and the refresh raises ``StaleStaticInputs``.

Token keys: protenix (the requirement rows), esmfold2 and boltz2 (the same H100 token kernel takes B > 1 for every variant; no rows yet).
Atom key protenix (rows ``atom_folds*``), N 128 and a padded N 160.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.gpu]

S, F, L = 5, 5, 128
BF = torch.bfloat16

TOKEN = {  # key -> (d_cond, d_pair)
    "protenix": (384, 128),
    "esmfold2": (768, 256),
    "boltz2": (768, 128),
}


def _h100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def _init(module):
    torch.manual_seed(0)
    with torch.no_grad():
        for p in module.parameters():
            torch.nn.init.normal_(p, std=0.05) if p.dim() > 1 else torch.nn.init.normal_(p, mean=0.3, std=0.2)
    return module.float().cuda().eval().requires_grad_(False)


class GroupedStep(torch.nn.Module):
    """One grouped sampler step on static inputs, laid out as FoldForge's ``GroupedSampler`` calls the blocks: ``x`` [S, F, ., d], the
    per-fold conditioning [1, F, ., d] expanded over S at every call, the folds' pairs [F, ...], one key mask expanded over F (once)."""

    def __init__(self, token_key: str, atom_key: str, n_atoms: int, folds: int = F):
        super().__init__()
        from miniworld_engine.modules import AtomDiTStack, TokenDiTStack

        dc, dp = TOKEN[token_key]
        self.token = _init(TokenDiTStack(token_key, 3, 768, dc, dp, 16, 2, backend="cuda"))
        self.atom = _init(AtomDiTStack(atom_key, 2, backend="cuda"))
        self.f, self.n = folds, n_atoms
        a = self.write_folds(1, fresh=True)
        # the static tensors (FoldForge: the stacked acts, single_cond rows, pair_all, km_all, the atom queries, cond_enc, atom_pair, atom_km)
        self.x_tok, self.c_tok, self.z_tok, self.m_tok = a["x_tok"], a["c_tok"], a["z_tok"], a["m_tok"]
        self.x_atom, self.c_atom, self.z_atom, self.m_atom = a["x_atom"], a["c_atom"], a["z_atom"], a["m_atom"]
        self.km_tok = self.m_tok.expand(folds, L)  # one key mask expanded over the folds, made once (FoldForge ``km_all``)
        self.km_atom = self.m_atom.expand(folds, n_atoms)

    def _draw(self, seed: int, folds: int | None = None) -> dict[str, torch.Tensor]:
        f = self.f if folds is None else folds
        g = torch.Generator(device="cuda").manual_seed(seed)
        tblock: Any = self.token.blocks[0]
        ablock: Any = self.atom.blocks[0]
        tok = tblock.make_inputs(length=L, samples=S, depth=f, dtype=BF, device="cuda", generator=g, stream="token_folds_shared_cond")
        atom = ablock.make_inputs(length=self.n, samples=S, depth=f, dtype=BF, device="cuda", generator=g, stream="atom_folds_shared_cond")
        assert tok["cond"].stride(0) == 0 and atom["cond"].stride(0) == 0, "the shared-cond streams expand the per-fold cond over S"
        out = {
            "x_tok": tok["single"],
            "c_tok": tok["cond"][:1],  # [1, F, L, dc]: one per fold
            "z_tok": tok["pair"],  # [F, L, L, dp]
            "m_tok": tok["mask"][:1].clone(),
            "x_atom": atom["single"],
            "c_atom": atom["cond"][:1],
            "z_atom": atom["pair"],  # [F, nwin, 32, 128, 16]
            "m_atom": atom["mask"][:1].clone(),
        }
        out["m_tok"][0, L - 20 - seed % 5 :] = False  # a mask of this fold set's own
        out["m_atom"][0, self.n - 30 - seed % 7 :] = False
        return out

    def write_folds(self, seed: int, fresh: bool = False) -> dict[str, torch.Tensor]:
        """Fold set ``seed``'s inputs; written in place (``copy_``: same storage, the version counters move) unless ``fresh``."""
        new = self._draw(seed)
        if fresh:
            return {k: v.contiguous().clone() for k, v in new.items()}
        for k, v in new.items():
            getattr(self, k).copy_(v)
        return new

    def static_tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.x_tok, self.c_tok, self.z_tok, self.km_tok, self.x_atom, self.c_atom, self.z_atom, self.km_atom)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        ctok = self.c_tok.expand(S, -1, -1, -1)  # per fold, shared by the samples: stride 0, a new view every call
        tok = self.token(self.x_tok, ctok, self.z_tok, self.km_tok)
        catom = self.c_atom.expand(S, -1, -1, -1)
        atom = self.atom(self.x_atom, catom, self.z_atom, self.km_atom)
        return tok, atom


def _capture(step: GroupedStep):
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


def _eager(step: GroupedStep):
    with torch.no_grad():
        tok, atom = step()
    return tok.clone(), atom.clone()


def _same(out, ref) -> bool:
    return all(torch.equal(a, b) for a, b in zip(out, ref, strict=True))


def _per_fold_differs(out) -> bool:
    """The folds see their own inputs: no two folds of one sample give the same output."""
    return all(not torch.equal(t[:, 0], t[:, 1]) for t in out)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) path")
@pytest.mark.parametrize(
    ("token_key", "atom_key", "n_atoms"),
    [
        ("protenix", "protenix", 128),
        ("protenix", "protenix", 160),  # N not a multiple of 128: the atom tables pad in their builds
        ("esmfold2", "protenix", 128),
        ("boltz2", "protenix", 160),
    ],
)
def test_grouped_refresh_then_replay_is_eager_on_the_new_fold_set(token_key, atom_key, n_atoms):
    from miniworld_engine.capture import clear_weight_caches, refresh_static_inputs, release_static_inputs

    step = GroupedStep(token_key, atom_key, n_atoms)
    try:
        # 1. fold set A: capture
        graph, out, tables = _capture(step)
        graph.replay()
        torch.cuda.synchronize()
        ref_a = _eager(step)
        assert _same(out, ref_a), "fold set A: replay != eager"
        assert _per_fold_differs(ref_a)
        # 3 token blocks (one grouped pair bias each) + 2 atom blocks (the folds' conditioning tables, the grouped pair bias)
        assert len(tables) == 3 + 2 * 2, f"the capture read {len(tables)} hoisted tables from the eager cache, expected 7"

        # 3. fold set B without the refresh: the replay reads fold set A's grouped tables
        step.write_folds(2)
        ref_b = _eager(step)
        graph.replay()
        torch.cuda.synchronize()
        assert not _same(out[:1], ref_b[:1]), "token: the replay without a refresh matched fold set B"
        assert not _same(out[1:], ref_b[1:]), "atom: the replay without a refresh matched fold set B"

        # 2. refresh + replay: bitwise eager on fold set B, the tables rewritten in place
        ptrs = [t.data_ptr() for t in out]
        n = tables.refresh(step)
        graph.replay()
        torch.cuda.synchronize()
        assert n == len(tables)
        assert [t.data_ptr() for t in out] == ptrs
        assert _same(out[:1], ref_b[:1]), "token: fold set B refresh + replay != eager"
        assert _same(out[1:], ref_b[1:]), "atom: fold set B refresh + replay != eager"

        # 4. a fresh capture on fold set B gives the same outputs
        g2, out2, t2 = _capture(step)
        g2.replay()
        torch.cuda.synchronize()
        assert _same(out2, ref_b), "fold set B: a fresh capture != eager"
        assert _same(out2, out)
        del g2
        release_static_inputs(tables=t2)

        # a third fold set through the same graph (the refresh again)
        step.write_folds(3)
        ref_c = _eager(step)
        refresh_static_inputs(step, tables=tables)
        graph.replay()
        torch.cuda.synchronize()
        assert _same(out, ref_c), "fold set C: refresh + replay != eager"
        assert tables.release() == n
    finally:
        clear_weight_caches(step)


@pytest.mark.skipif(not _h100(), reason="H100 (sm_90) path")
def test_grouped_fold_count_change_is_a_new_signature_and_stale():
    """Another fold count is another graph: ``signature`` of the static tensors differs, and a graph captured on 5 folds whose static tensors
    now hold 3 folds' layout (re-pointed in place with ``set_``) refuses the refresh (``StaleStaticInputs``). The graph is not replayed after that: it
    reads the 5-fold storage the tensors no longer hold."""
    from miniworld_engine.capture import StaleStaticInputs, clear_weight_caches, signature

    step = GroupedStep("protenix", "protenix", 160)
    try:
        sig5 = signature(step, *step.static_tensors())
        assert sig5 == signature(step, *step.static_tensors())
        graph, out, tables = _capture(step)
        step.write_folds(2)  # same layout: the signature holds
        assert signature(step, *step.static_tensors()) == sig5
        assert tables.refresh(step) == len(tables) > 0

        new = step._draw(4, folds=3)
        for name in ("x_tok", "c_tok", "z_tok", "x_atom", "c_atom", "z_atom"):
            t = getattr(step, name)
            t.set_(new[name].contiguous().clone())  # the same tensor objects, now holding 3 folds
        assert step.z_tok.shape[0] == 3 and step.z_atom.shape[0] == 3
        assert signature(step, *step.static_tensors()) != sig5
        with pytest.raises(StaleStaticInputs, match="layout changed"):
            tables.refresh(step)
        del graph, out
    finally:
        clear_weight_caches(step)
