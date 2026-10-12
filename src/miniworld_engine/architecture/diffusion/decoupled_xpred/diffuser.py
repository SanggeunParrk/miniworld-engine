"""VE x-prediction diffuser (Decoupled coordinate + rigid-body SE(3))."""

from __future__ import annotations

import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel

from miniworld_engine.architecture.diffusion.base.diffuser import Diffuser
from miniworld_engine.architecture.diffusion.decoupled_xpred.scheduler import (
    DecoupledXPredScheduler,
)
from miniworld_engine.architecture.utils.align import weighted_align
from miniworld_engine.architecture.utils.se3 import apply_chain_rt, sample_rigid


class XPredDecoupledDiffuser(Diffuser):
    """VE x-prediction diffuser for decoupled coordinate + rigid-body noise.

    The network predicts x0/sigma_data (unit-scale output).
    ``get_x0_hat`` multiplies by sigma_data to recover original coordinates.
    """

    class DecoupledXPredConfig(BaseModel):
        """Configuration for XPredDecoupledDiffuser."""

        method: str = "AF3"
        seed: int = 0
        translation_noise: float = 1.0
        max_loss_weight: float = 100.0

    def __init__(
        self,
        config: DecoupledXPredConfig,
        scheduler: DecoupledXPredScheduler,
    ) -> None:
        self.config = config
        self.scheduler: DecoupledXPredScheduler = scheduler
        self._set_seed(config.seed)
        self.dtype = torch.float32

    @property
    def sigma_data(self) -> float:
        """Data standard deviation from scheduler config."""
        return self.scheduler.config.sigma_data

    def loss_weight(self, sigma: Float[torch.Tensor, ...]) -> Float[torch.Tensor, ...]:
        """v-loss weight in sigma space: clamp((sigma + sigma_data)^2 / sigma^2, max)."""
        w = (sigma + self.sigma_data) ** 2 / sigma**2
        return w.clamp(max=self.config.max_loss_weight)

    # ── chain-splitting ─────────────────────────────────────────────────────

    @staticmethod
    def _build_contact_graph(
        chain_atoms: list[torch.Tensor],
        dist_cutoff: float,
    ) -> torch.Tensor:
        """Symmetric bool adjacency (n_chains, n_chains) for chains within dist_cutoff."""
        n = len(chain_atoms)
        nonempty = [(i, ca) for i, ca in enumerate(chain_atoms) if len(ca) > 0]
        if len(nonempty) < 2:
            device = chain_atoms[0].device if chain_atoms else torch.device("cpu")
            return torch.zeros(n, n, dtype=torch.bool, device=device)

        _, atoms = zip(*nonempty, strict=True)
        all_atoms = torch.cat(atoms)
        device = all_atoms.device
        labels = torch.cat(
            [
                torch.full((len(ca),), idx, dtype=torch.long, device=device)
                for idx, ca in nonempty
            ],
        )

        D = torch.cdist(all_atoms, all_atoms)
        pair_idx = labels.unsqueeze(1) * n + labels.unsqueeze(0)
        chain_min_dist = torch.full((n * n,), float("inf"), device=device)
        chain_min_dist.scatter_reduce_(
            0,
            pair_idx.flatten(),
            D.flatten(),
            reduce="amin",
            include_self=True,
        )
        contact = chain_min_dist.view(n, n) <= dist_cutoff
        contact.fill_diagonal_(fill_value=False)
        return contact

    @staticmethod
    def _find(x: int, p: list[int]) -> int:
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    @staticmethod
    def _random_kruskal(
        n: int,
        edges: list[list[int]],
    ) -> list[tuple[int, int]]:
        """Random spanning forest via Kruskal with shuffled edge order."""
        parent = list(range(n))
        forest: list[tuple[int, int]] = []
        for idx in torch.randperm(len(edges)).tolist():
            i, j = edges[idx]
            pi = XPredDecoupledDiffuser._find(i, parent)
            pj = XPredDecoupledDiffuser._find(j, parent)
            if pi != pj:
                parent[pi] = pj
                forest.append((i, j))
        return forest

    @staticmethod
    def _compress_components(n: int, kept_edges: list[tuple[int, int]]) -> list[int]:
        """Union-Find on kept edges; return contiguous group index per node."""
        comp = list(range(n))
        for i, j in kept_edges:
            pi = XPredDecoupledDiffuser._find(i, comp)
            pj = XPredDecoupledDiffuser._find(j, comp)
            if pi != pj:
                comp[pi] = pj
        root_to_group: dict[int, int] = {}
        return [
            root_to_group.setdefault(
                XPredDecoupledDiffuser._find(i, comp),
                len(root_to_group),
            )
            for i in range(n)
        ]

    @staticmethod
    def _spanning_forest_cut(contact: torch.Tensor, max_groups: int = 4) -> list[int]:
        """Random spanning forest cut on the contact graph."""
        n = contact.shape[0]
        edges = contact.triu(diagonal=1).nonzero().tolist()
        forest = XPredDecoupledDiffuser._random_kruskal(n, edges)

        k = n - len(forest)
        min_target_groups = max(2, k)
        max_target_groups = min(n, max_groups)
        if max_target_groups < min_target_groups:
            target_groups = k
        else:
            target_groups = int(
                torch.randint(min_target_groups, max_target_groups + 1, (1,)).item(),
            )

        cut_set: set[int] = set()
        group_count = k
        for idx in torch.randperm(len(forest)).tolist():
            if group_count >= target_groups:
                break
            cut_set.add(idx)
            group_count += 1

        kept = [e for idx, e in enumerate(forest) if idx not in cut_set]
        return XPredDecoupledDiffuser._compress_components(n, kept)

    @torch.no_grad()
    def _randomly_split_chains(
        self,
        x0: Float[torch.Tensor, "B L 3"],
        mask: Bool[torch.Tensor, "B L"] | None,
        atom_to_chain_idx: torch.Tensor,
        dist_cutoff: float = 6.0,
        max_groups: int = 4,
    ) -> torch.Tensor:
        """Randomly split chains into target-sampled rigid body groups."""
        device = atom_to_chain_idx.device
        result = torch.zeros_like(atom_to_chain_idx)

        for b in range(atom_to_chain_idx.shape[0]):
            chain_ids = torch.unique(atom_to_chain_idx[b])
            n_chains = len(chain_ids)
            if n_chains <= 1:
                continue

            valid = (
                mask[b]
                if mask is not None
                else atom_to_chain_idx[b].new_ones(x0.shape[1], dtype=torch.bool)
            )
            chain_atoms = [
                x0[b][(atom_to_chain_idx[b] == cid) & valid] for cid in chain_ids
            ]

            contact = self._build_contact_graph(chain_atoms, dist_cutoff)
            group_map = self._spanning_forest_cut(contact, max_groups=max_groups)

            lut = torch.zeros(int(chain_ids.max()) + 1, dtype=torch.long, device=device)
            lut[chain_ids] = torch.tensor(group_map, dtype=torch.long, device=device)
            result[b] = lut[atom_to_chain_idx[b]]

        return result

    # ── forward (training) ──────────────────────────────────────────────────

    def sample(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor | None,
        atom_to_chain_idx: torch.Tensor,
        num_augment: int = 1,
    ) -> tuple[
        Float[torch.Tensor, "... L 3"],
        Float[torch.Tensor, "... L 3"],
        Bool[torch.Tensor, "... L"] | None,
        Float[torch.Tensor, ...],
        Float[torch.Tensor, ...],
        Float[torch.Tensor, ...],
        Float[torch.Tensor, ...],
        Int[torch.Tensor, ...],
    ]:
        """VE noise + decoupled R/T. Returns 8-tuple for downstream loss/inference."""
        if num_augment < 1:
            msg = "num_augment must be at least 1"
            raise ValueError(msg)
        if atom_to_chain_idx is None:
            msg = "atom_to_chain_idx is required."
            raise ValueError(msg)

        batch_size = x0.shape[0]
        device, dtype = x0.device, self.dtype
        if x0.dtype != dtype:
            x0 = x0.to(device=device, dtype=dtype)

        # -- Expand for augmentation ------------------------------------------
        if len(x0.shape) == 3:
            x0 = x0.unsqueeze(0).expand(num_augment, -1, -1, -1)
            x0 = x0.reshape(num_augment * batch_size, *x0.shape[2:])
            atom_to_chain_idx = atom_to_chain_idx.expand(
                batch_size,
                *atom_to_chain_idx.shape[1:],
            )
            atom_to_chain_idx = atom_to_chain_idx.unsqueeze(0).expand(
                num_augment,
                -1,
                *atom_to_chain_idx.shape[1:],
            )
            atom_to_chain_idx = atom_to_chain_idx.reshape(
                num_augment * batch_size,
                *atom_to_chain_idx.shape[2:],
            )
            if mask is not None:
                mask = mask.unsqueeze(0).expand(num_augment, -1, -1)
                mask = mask.reshape(num_augment * batch_size, *mask.shape[2:])
        elif len(x0.shape) == 4:
            num_expand = num_augment // x0.shape[1]
            num_augment = num_expand * x0.shape[1]
            atom_to_chain_idx = atom_to_chain_idx.unsqueeze(1).expand(
                -1,
                x0.shape[1],
                -1,
            )
            atom_to_chain_idx = atom_to_chain_idx.reshape(
                -1,
                atom_to_chain_idx.shape[-1],
            ).repeat(num_expand, 1)
            x0 = x0.reshape(-1, *x0.shape[2:]).repeat(num_expand, 1, 1)
            if mask is not None:
                mask = mask.reshape(-1, *mask.shape[2:]).repeat(num_expand, 1)

        x0 = self.random_rotation_and_translation(x0)

        total_num = x0.shape[0]
        sigma_shape = (total_num,) + (1,) * (x0.ndim - 1)

        # -- Chain splitting --------------------------------------------------
        atom_to_combine = self._randomly_split_chains(
            x0,
            mask,
            atom_to_chain_idx,
        )
        group_num = int(atom_to_combine.max().item()) + 1

        # -- VE noise ---------------------------------------------------------
        sigma_y, sigma_rotation, sigma_translation = self.scheduler.sample_noise(
            total_num,
        )
        sigma_y = sigma_y.to(device=device, dtype=dtype)
        sigma_rotation = sigma_rotation.to(device=device, dtype=dtype)
        sigma_translation = sigma_translation.to(device=device, dtype=dtype)

        noise = torch.randn_like(x0)
        noisy_x = x0 + noise * sigma_y.view(sigma_shape)

        # -- R/T --------------------------------------------------------------
        rotation_matrix, translation_vector = sample_rigid(
            sigma_rotation,
            sigma_translation,
            C=group_num,
            device=device,
            dtype=dtype,
        )
        noisy_x = apply_chain_rt(
            noisy_x,
            rotation_matrix,
            translation_vector,
            atom_to_combine,
        )

        # -- c_in (EDM) -------------------------------------------------------
        sigma_y = sigma_y.view(sigma_shape)
        sigma_translation = sigma_translation.view(sigma_shape)
        input_scaling = self.scheduler.input_scale(sigma_y, sigma_translation).to(
            device=device,
            dtype=dtype,
        )
        x_input = noisy_x * input_scaling
        t_emb = self.scheduler.noise_condition(sigma_y).to(device=device, dtype=dtype)

        # -- Reshape ----------------------------------------------------------
        x0 = x0.view(num_augment, batch_size, *x0.shape[1:])
        sigma_y = sigma_y.view(num_augment, batch_size, *sigma_y.shape[1:])
        x_input = x_input.view(num_augment, batch_size, *x_input.shape[1:])
        if mask is not None:
            mask = mask.view(num_augment, batch_size, *mask.shape[1:])
        t_emb = t_emb.view(num_augment, batch_size, *t_emb.shape[1:])

        return (
            x0,
            x_input,
            mask,
            t_emb,
            sigma_y,
            rotation_matrix,
            translation_vector,
            atom_to_combine,
        )

    # ── prediction ──────────────────────────────────────────────────────────

    def get_x0_hat(
        self,
        x0: Float[torch.Tensor, "... L 3"],
        x_input: Float[torch.Tensor, "... L 3"],
        x_update: Float[torch.Tensor, "... L 3"],
        sigma_y: Float[torch.Tensor, ...],
        rotation_matrix: Float[torch.Tensor, ...],
        translation_vector: Float[torch.Tensor, ...],
        atom_to_combine: Int[torch.Tensor, "... L"],
        mask: Bool[torch.Tensor, "... L"] | None = None,
    ) -> Float[torch.Tensor, "... L 3"]:
        """Recover original coordinates: x0_hat = F_theta * sigma_data."""
        if x_update.dtype != self.dtype:
            msg = "x_update must be float32, got " + str(x_update.dtype)
            raise ValueError(msg)
        return (x_update * self.sigma_data).reshape(x0.shape)

    # ── loss ────────────────────────────────────────────────────────────────

    def cal_loss(
        self,
        x0: Float[torch.Tensor, "... L 3"],
        x_pred: Float[torch.Tensor, "... L 3"],
        sigma_y: Float[torch.Tensor, ...],
        mask: Bool[torch.Tensor, "... L"] | None = None,
        atom_weight: Float[torch.Tensor, "... L"] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Float[torch.Tensor, 1]:
        """Loss on normalized target: lambda * ||x_pred/sd - x0/sd||^2.

        x_pred is in original space (from get_x0_hat = F_theta * sigma_data).

        ``atom_weight`` is the AF3 per-atom weight w_l (Eq. 4:
        w_l = 1 + is_dna*alpha_dna + is_rna*alpha_rna + is_ligand*alpha_ligand);
        it multiplies the per-atom squared error AND the rigid alignment weights
        (AF3 aligns with the same weights). Normalisation stays the raw valid-atom
        count (3*N), i.e. the AF3 (1/3)*mean_l form, so upweighted atoms genuinely
        contribute more rather than being renormalised away. Defaults to 1 (no-op).
        """
        sd = self.sigma_data
        weight = self.loss_weight(sigma_y).to(dtype=dtype)

        if mask is None:
            mask = torch.ones(x0.shape[:-1], device=x0.device, dtype=torch.bool)
        mask_f = mask.to(dtype=dtype)
        if atom_weight is not None:
            mask_f = mask_f * atom_weight.to(dtype=dtype)  # fold w_l into the mask
        weight = weight * mask_f.unsqueeze(-1)

        x0 = x0.to(dtype=dtype)
        x0_aligned = weighted_align(x0, x_pred, weight=mask_f)
        per_sample_loss = (((x_pred - x0_aligned) / sd).pow(2) * weight).sum(
            dim=(-2, -1)
        )
        n_valid = mask.sum(dim=-1).clamp(min=1) * 3
        return (per_sample_loss / n_valid).mean()
