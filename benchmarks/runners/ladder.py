"""The performance ladder: CUDA < portable (Triton) < PyTorch, for every op, mode, dtype and shape the engine serves.

The invariant (maintainer's rule): *the portable kernels must always be faster than PyTorch, and the hand CUDA kernels must always be
faster than the portable ones*. This file checks it. It adds no timing of its own: it drives `benchmarks/runners/bench.py` (the
module `bench.yaml` mechanism; `ladder_ops` for the ops that have no module) once per (group, mode) with the three rungs

    pytorch   -- ``implementations=pytorch``   the best PyTorch form: ``torch.compile`` + CUDA-graph replay (``pt_compiled``);
                 the eager statements replayed in a graph (``pt_eager_graph``) and plain eager (``pt_eager``) are reported next to it,
                 and the PyTorch rung is the FASTEST of the three that ran (a careful user would run that one)
    portable  -- ``implementations=portable``  the portable Triton family (``settings.force_portable_kernels`` for the row)
    cuda      -- ``implementations=miniworld`` the engine's own routing on this card (hand CUDA on an H100)

and reads the CSVs back. Every number is the median of ``repeats`` (default 3) independent ``do_bench`` medians of the same callable.

    python benchmarks/runners/ladder.py plan   [--tier smoke|core|full] [--group REGEX] [--out DIR]
    python benchmarks/runners/ladder.py run    [--tier ...] [--group ...] [--out DIR] [--repeats 3]   # needs a GPU: run it through Slurm
    python benchmarks/runners/ladder.py report --out DIR [--md FILE] [--csv FILE]                          # no GPU needed
    python benchmarks/runners/ladder.py gate   --csv FILE [--allow FILE] [--strict]                         # exit 1 on a violation

`miniworld-engine ladder {plan,run,report,gate}` is the same thing. Verdict per row, per rung pair (ratio = lower rung / upper rung):

    OK                                  every pair that could be measured has ratio < 1
    VIOLATION(cuda_vs_portable)         CUDA is not faster than portable (ratio >= 1)
    VIOLATION(portable_vs_pytorch)      portable is not faster than the best PyTorch
    VIOLATION(cuda_vs_pytorch)          (ops with no portable rung only) CUDA is not faster than the best PyTorch
    n.a.(rung missing)                  fewer than two rungs have a measurement; the missing rungs are named

A pair whose ratio is within the measured repeat spread of 1 is still a violation if ratio >= 1; it is marked ``noise`` in the
``note`` column so the maintainer can tell a tie from a loss. Nothing in this file reads or writes outside ``--out`` except the
CSVs ``bench.py`` writes under ``benchmarks/modules/<target>/artifacts/``.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "benchmarks" / "runners" / "bench.py"

BOTH = ("inference", "training")
DEFAULT_RUNGS = {"pytorch": "pytorch", "portable": "portable", "cuda": "miniworld"}
NO_PORTABLE = {"pytorch": "pytorch", "cuda": "miniworld"}
TIERS = {"smoke": 0, "core": 1, "full": 2}


@dataclass(frozen=True)
class Group:
    """One row family of the ladder: a bench target at fixed overrides, swept over ``lengths`` (``seq_len``)."""

    id: str
    op: str                       # the op as the maintainer names it (report column)
    target: str                   # bench.py module target
    lengths: tuple[int, ...]
    desc: str                     # the shape other than L, for the table ("D128", "A5/A48", ...)
    overrides: tuple[str, ...] = ()
    modes: tuple[str, ...] = BOTH
    precision: str = "bf16-mixed"  # bench.py precision: bf16-mixed | 32
    weight: int = 2               # production importance: 3 = the shape MiniWorld / FoldForge / La-Proteina run, 2 = served, 1 = exotic
    tier: str = "core"
    rungs: dict = field(default_factory=lambda: dict(DEFAULT_RUNGS))
    #: per-mode extra overrides (n_augment: inference 5 / training 48 is bench.py's own default)
    mode_overrides: dict = field(default_factory=dict)
    note: str = ""
    dtype_label: str = ""         # for the ops whose dtypes are fixed by the op, not by `precision`

    @property
    def dtype(self) -> str:
        return self.dtype_label or ("fp32" if self.precision == "32" else "bf16")


def G(id, op, target, lengths, desc, *over, **kw):
    return Group(id=id, op=op, target=target, lengths=tuple(lengths), desc=desc, overrides=tuple(over), **kw)


def ops(id, op, ladder_op, lengths, desc, *over, **kw):
    kw.setdefault("rungs", dict(DEFAULT_RUNGS))
    return G(id, op, "ladder_ops", lengths, desc, f"+ladder_op={ladder_op}", *over, **kw)


GROUPS: list[Group] = [
    # ---- TriMul (pair width D, hidden = D unless stated; bf16; B = 1) --------------------------------------------------------------
    G("tm_out_D128", "TriMul outgoing", "triangle_multiplication", (128, 384, 768, 1024), "D128", "d_pair=128", weight=3, tier="smoke"),
    G("tm_out_D64", "TriMul outgoing", "triangle_multiplication", (128, 384), "D64", "d_pair=64", weight=2),
    G("tm_out_D256", "TriMul outgoing", "triangle_multiplication", (384, 768), "D256", "d_pair=256", weight=2),
    G("tm_out_D512", "TriMul outgoing", "triangle_multiplication", (384, 768), "D512", "d_pair=512", weight=1),
    G("tm_in_D128", "TriMul incoming", "triangle_multiplication", (384, 768), "D128", "d_pair=128", "trimul_direction=incoming", weight=2),
    G("tm_in_D512", "TriMul incoming", "triangle_multiplication", (384,), "D512", "d_pair=512", "trimul_direction=incoming", weight=1, tier="full"),
    G("tm_out_h128_D64", "TriMul outgoing hidden!=D", "triangle_multiplication", (128, 384), "D64 hidden128", "d_pair=64", "+trimul_d_hidden=128", weight=2),
    G("tm_out_h256_D128", "TriMul outgoing hidden!=D", "triangle_multiplication", (384,), "D128 hidden256", "d_pair=128", "+trimul_d_hidden=256", weight=1, tier="full"),
    G("tmb_D128", "TriMul bidirectional", "triangle_multiplication_bidirectional", (128, 384, 768), "D128", "d_pair=128", weight=3, tier="smoke"),
    G("tmb_D64", "TriMul bidirectional", "triangle_multiplication_bidirectional", (384,), "D64", "d_pair=64", weight=2),
    G("tmb_D256", "TriMul bidirectional", "triangle_multiplication_bidirectional", (384, 768), "D256", "d_pair=256", weight=2),
    G("tmb_D512", "TriMul bidirectional", "triangle_multiplication_bidirectional", (384,), "D512", "d_pair=512", weight=1, tier="full"),
    G("tmb_h128_D64", "TriMul bidirectional hidden!=D", "triangle_multiplication_bidirectional", (384,), "D64 hidden128", "d_pair=64", "+trimul_d_hidden=128", weight=2, tier="full"),
    # ---- TriAttn (starting node; 32-channel heads) ---------------------------------------------------------------------------------
    G("ta_D128", "TriAttn", "triangle_attention", (128, 384, 768), "D128 4x32", "d_pair=128", weight=3, tier="smoke"),
    G("ta_D256", "TriAttn", "triangle_attention", (384,), "D256 8x32", "d_pair=256", "+tri_n_head=8", weight=2),
    G("ta_D64", "TriAttn", "triangle_attention", (384,), "D64 2x32", "d_pair=64", "+tri_n_head=2", weight=1, tier="full"),
    # ---- Transition (n = 4; bf16, and fp32 on TF32) -------------------------------------------------------------------------------------
    G("tr_D128", "Transition", "transition", (128, 384, 768, 1024), "D128", "d_pair=128", weight=3, tier="smoke"),
    G("tr_D256", "Transition", "transition", (384, 768), "D256", "d_pair=256", weight=2),
    G("tr_D384", "Transition", "transition", (384, 768), "D384", "d_pair=384", weight=2),
    G("tr_D512", "Transition", "transition", (384,), "D512", "d_pair=512", weight=1),
    G("tr_D768", "Transition", "transition", (384,), "D768", "d_pair=768", weight=2),
    G("tr_fp32_D128", "Transition", "transition", (384, 768), "D128 TF32", "d_pair=128", precision="32", weight=2),
    G("tr_fp32_D256", "Transition", "transition", (384,), "D256 TF32", "d_pair=256", precision="32", weight=1, tier="full"),
    # ---- MSA module ---------------------------------------------------------------------------------------------------------------------
    G("opm", "OuterProductMean", "outer_product", (256, 384, 768), "S1024 msa64 pair128", "d_pair=128", weight=3, tier="smoke"),
    G("pwa", "MSAPairWeightedAveraging", "msa_pair_weighted_averaging", (256, 384, 768), "S1024 8x32 pair128", "d_pair=128", weight=3, tier="smoke"),
    G("opm_rf3", "OuterProductMean RF3 (bias, divide before)", "outer_product", (256, 384), "S512 msa64 pair128", "d_pair=128", "+n_msa=512",
      "+opm_variant=rf3", weight=2),
    G("opm_rf3_after", "OuterProductMean RF3 (bias, divide after)", "outer_product", (256, 384), "S512 msa64 pair128", "d_pair=128", "+n_msa=512",
      "+opm_variant=rf3_after", weight=1, tier="full"),
    G("opm_chai1", "OuterProductMean Chai-1 (8 groups)", "outer_product", (256, 384), "S512 pair256 8x64", "d_pair=256", "+n_msa=512",
      "+opm_variant=chai1", weight=2),
    G("pwa_v8", "MSAPairWeightedAveraging value_dim 8", "msa_pair_weighted_averaging", (256, 384), "S512 8x8 pair128", "d_pair=128", "+n_msa=512",
      "+msa_d_hidden=8", weight=2),
    # ---- attention with pair bias --------------------------------------------------------------------------------------------------------
    G("apb_h8", "AttentionPairBias", "attention_pair_bias", (384, 768), "8x48 pair128", "d_pair=128", weight=3),
    G("apb_h16", "AttentionPairBias", "attention_pair_bias", (384,), "16x24 pair128", "d_pair=128", "+apb_n_head=16", weight=2),
    G("apb_p256", "AttentionPairBias", "attention_pair_bias", (384,), "8x48 pair256", "d_pair=256", weight=1, tier="full"),
    G("aat_bf16", "AugmentedAttention token", "augmented_attention_token", (384, 768), "d768 A5/A48", weight=2),
    G("aat_fp32", "AugmentedAttention token", "augmented_attention_token", (384,), "d768 A5/A48", precision="32", weight=1, tier="full"),
    G("aaa_bf16", "AugmentedAttention atom", "augmented_attention_atom", (128, 384), "d128 A5/A48", weight=1, tier="full"),
    # ---- diffusion transformers ---------------------------------------------------------------------------------------------------------
    G("dit_bf16", "token DiT (fused+composed)", "dit", (384, 768), "d768 16x48 A5/A48 shared-cond", "+shared_cond=true", weight=3, tier="smoke"),
    G("dit_fp32", "token DiT (fused+composed)", "dit", (384,), "d768 A5/A48 shared-cond", "+shared_cond=true", precision="32", weight=2),
    G("dit_qk", "token DiT (fused+composed)", "dit", (384,), "d768 qk-norm shared-cond", "+shared_cond=true", "+dit_qk_norm=true", weight=2),
    G("dit_xbias_bf16", "token DiT + extra_pair_bias", "dit", (256, 512), "d768 A2 shared-cond +[L,L] bias", "n_augment=2", "+shared_cond=true",
      "+dit_extra_pair_bias=true", weight=2),
    G("dit_xbias_fp32", "token DiT + extra_pair_bias", "dit", (256,), "d768 A2 shared-cond +[L,L] bias", "n_augment=2", "+shared_cond=true",
      "+dit_extra_pair_bias=true", precision="32", weight=1, tier="full"),
    G("dit_boltz2", "DiT variant stack", "dit", (384,), "Boltz-2 token", "+dit_variant=boltz2", weight=2),
    G("dit_rf3", "DiT variant stack", "dit", (384,), "RoseTTAFold3 token", "+dit_variant=rosettafold3", weight=2),
    G("dit_chai1", "DiT variant stack", "dit", (384,), "Chai-1 token", "+dit_variant=chai1", weight=2),
    G("dit_boltz2_fp32", "DiT variant stack", "dit", (384,), "Boltz-2 token", "+dit_variant=boltz2", precision="32", weight=1),
    G("dit_rf3_fp32", "DiT variant stack", "dit", (384,), "RoseTTAFold3 token", "+dit_variant=rosettafold3", precision="32", weight=1),
    G("dit_chai1_fp32", "DiT variant stack", "dit", (384,), "Chai-1 token", "+dit_variant=chai1", precision="32", weight=1),
    G("local_bf16", "LocalDiT atom block", "local_dit", (512, 1024), "d128 32x128 (atoms=8L)", weight=3, tier="smoke"),
    G("local_fp32", "LocalDiT atom block", "local_dit", (512,), "d128 32x128 (atoms=8L)", precision="32", weight=2),
    G("local_esm", "LocalDiT ESMFold2 atom block", "local_dit", (512, 1024), "d128 192 keys rotary (atoms=8L)", "+local_dit_preset=esmfold2", "+shared_cond=true",
      weight=2),
    G("local_boltz2", "DiT variant stack (atom)", "local_dit", (512,), "Boltz-2 atom", "+dit_variant=boltz2", weight=2),
    G("local_rf3", "DiT variant stack (atom)", "local_dit", (512,), "RoseTTAFold3 atom", "+dit_variant=rosettafold3", weight=2),
    G("local_chai1", "DiT variant stack (atom)", "local_dit", (512,), "Chai-1 atom", "+dit_variant=chai1", weight=2),
    G("local_boltz2_fp32", "DiT variant stack (atom)", "local_dit", (512,), "Boltz-2 atom", "+dit_variant=boltz2", precision="32", weight=1),
    G("local_rf3_fp32", "DiT variant stack (atom)", "local_dit", (512,), "RoseTTAFold3 atom", "+dit_variant=rosettafold3", precision="32", weight=1),
    G("local_chai1_fp32", "DiT variant stack (atom)", "local_dit", (512,), "Chai-1 atom", "+dit_variant=chai1", precision="32", weight=1),
    G("bodit_bf16", "bias-only DiT", "bias_only_dit", (256, 512, 768), "d768 16x48 A5/A48", weight=3, tier="smoke"),
    G("bodit_psb", "bias-only DiT", "bias_only_dit", (256, 512), "d768 16x48 per-sample pair A4", "n_augment=4", "+per_sample_pair=true", modes=("training",),
      weight=2),
    G("bodit_fp32", "bias-only DiT", "bias_only_dit", (256, 512), "d768 16x48 A5/A48", precision="32", weight=2),
    G("bodit_h24", "bias-only DiT", "bias_only_dit", (512,), "24x32", "+n_head=24", weight=1, tier="full"),
    G("bodit_h12", "bias-only DiT", "bias_only_dit", (512,), "12x64", "+n_head=12", weight=1, tier="full"),
    G("swa_bf16", "SWA DiT (atom)", "swa_dit", (512, 1024), "d128 atoms=8L shared-cond", "+shared_cond=true", weight=3),
    G("swa_fp32", "SWA DiT (atom)", "swa_dit", (512,), "atoms=8L shared-cond", "+shared_cond=true", precision="32", weight=2),
    G("swa_attn", "SWA atom attention (global)", "swa_atom_attention", (512, 1024), "atoms=8L", weight=2, rungs=dict(NO_PORTABLE)),
    # ---- adaptive norms and the conditioned transition ---------------------------------------------------------------------------
    G("adaln_bf16", "AdaLN", "adaptive_layernorm", (384, 768, 1024), "d768 cond384 A5/A48", weight=2),
    G("adaln_fp32", "AdaLN", "adaptive_layernorm", (384,), "d768 cond384 A5/A48", precision="32", weight=2),
    G("ct_bf16", "ConditionedTransition", "conditioned_transition", (384, 768, 1024), "d768 A5/A48", weight=2),
    G("ct_fp32", "ConditionedTransition", "conditioned_transition", (384,), "d768 A5/A48", precision="32", weight=2),
    # ---- norms --------------------------------------------------------------------------------------------------------------------------
    ops("ln_pair_bf16", "LayerNorm (pair)", "layernorm", (384, 768), "x[1,L,L,128]", "d_pair=128", weight=3, tier="smoke"),
    ops("ln_pair_fp32", "LayerNorm (pair)", "layernorm", (384,), "x[1,L,L,128]", "d_pair=128", precision="32", weight=1),
    ops("ln_pair_D256", "LayerNorm (pair)", "layernorm", (384,), "x[1,L,L,256]", "d_pair=256", weight=2),
    ops("rms_bf16", "RMSNorm", "rmsnorm", (384, 1024), "x[48,L,768]", "n_augment=48", weight=2),
    # ---- La-Proteina pair builders and the token-pair initialisation (B = samples) ------------------------------------------------
    ops("geom_B16", "pair_embed_geom", "geom", (128, 256, 384), "B16", "n_augment=16", weight=3, tier="smoke", dtype_label="bf16 out, fp32 geometry"),
    ops("geom_B48", "pair_embed_geom", "geom", (128, 256), "B48", "n_augment=48", weight=2, dtype_label="bf16 out, fp32 geometry"),
    ops("ca_B16", "pair_embed_ca", "ca", (128, 256, 384), "B16", "n_augment=16", weight=3, tier="smoke", dtype_label="bf16 out, fp32 geometry"),
    ops("ca_B48", "pair_embed_ca", "ca", (128, 256), "B48", "n_augment=48", weight=2, dtype_label="bf16 out, fp32 geometry"),
    ops("adalnpair_B16", "pair_embed_adaln", "adaln", (128, 256, 384), "B16", "n_augment=16", weight=3, tier="smoke", dtype_label="bf16 out, fp32 geometry"),
    ops("adalnpair_B48", "pair_embed_adaln", "adaln", (128, 256), "B48", "n_augment=48", weight=2, dtype_label="bf16 out, fp32 geometry"),
    ops("tpi_B16", "token_pair_init", "tpi", (128, 256, 384), "B16", "n_augment=16", weight=3, tier="smoke", dtype_label="fp32 in, fp32 out"),
    ops("tpi_B48", "token_pair_init", "tpi", (128, 256), "B48", "n_augment=48", weight=2, dtype_label="fp32 in, fp32 out"),
    # ---- the diffusion loss and the sampler (atoms) -------------------------------------------------------------------------------------
    ops("slddt_A48", "smooth_lddt", "smooth_lddt", (1024, 2048), "A48 B1", "n_augment=48", weight=3, tier="smoke", dtype_label="fp32"),
    ops("slddt_A16", "smooth_lddt", "smooth_lddt", (4096,), "A16 B1", "n_augment=16", weight=3, dtype_label="fp32",
        note="the PyTorch reference cannot run A48 x N4096 (cdist indices overflow int32: illegal memory access)"),
    ops("samp_aug", "sampler augment_noise", "sampler_augment", (4096, 12288, 24576), "B5", "n_augment=5", weight=3, dtype_label="fp32", modes=("inference",)),
    ops("samp_align", "sampler align_euler", "sampler_align", (4096, 12288, 24576), "B5", "n_augment=5", weight=3, dtype_label="fp32", modes=("inference",)),
    ops("samp_pool", "sampler pool_atoms", "sampler_pool", (4096, 12288, 24576), "B5 c768", "n_augment=5", weight=2, dtype_label="bf16", modes=("inference",)),
]
BY_ID = {g.id: g for g in GROUPS}
PARAM_VARIANT = "@bf16p"


def group_of(gid: str) -> Group | None:
    """A group by id; ``<id>@bf16p`` is the same group re-run with bf16 parameters (``+param_master=false``): the fallback for a bf16-mixed
    (fp32 master) cell whose portable rung cannot take fp32 parameters under bf16 activations."""
    base, sep, _ = gid.partition("@")
    g = BY_ID.get(base)
    if g is None or not sep:
        return g
    return replace(g, id=gid, desc=g.desc + ", bf16 params", overrides=(*g.overrides, "+param_master=false"))


def select(args) -> list[Group]:
    rank = TIERS[args.tier]
    chosen = [g for g in GROUPS if TIERS[g.tier] <= rank]
    if args.group:
        rx = re.compile(args.group)
        chosen = [g for g in GROUPS if rx.search(g.id)]
    return chosen


# --------------------------------------------------------------------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------------------------------------------------------------------
def commands(g: Group, mode: str, repeats: int, out: Path) -> list[dict]:
    """The bench.py invocations of one (group, mode): the three rungs (compiled + graph), then the eager PyTorch references."""
    base = [
        sys.executable, str(BENCH), "level=module", f"target={g.target}", f"mode={mode}", "metric=time",
        f"precision={g.precision}", "sweep_axis=seq_len", f"+seq_len_values=[{','.join(map(str, g.lengths))}]",
        f"+repeats={repeats}", "+kernel_probe=true", "+fp32_check_limit=0.01", "hydra.job.chdir=false", "hydra.output_subdir=null",
        f"hydra.run.dir={out / 'hydra'}", *g.overrides, *g.mode_overrides.get(mode, ()),
    ]
    impls = ",".join(g.rungs[r] for r in ("pytorch", "portable", "cuda") if r in g.rungs)
    main = {"kind": "rungs", "regime": "compile+graph",
            "cmd": [*base, f"implementations=[{impls}]", "compile=true", "cudagraph=manual", f"name_suffix=ladder_{g.id}_rungs"]}
    # the engine rungs again WITHOUT torch.compile (CUDA-graph replay of the eager module: how the CUDA / Triton blocks are deployed and
    # measured in docs/gpus/h100); the ladder takes the faster of the two regimes for them. The PyTorch rung has its own eager_graph row.
    engine = ",".join(g.rungs[r] for r in ("portable", "cuda") if r in g.rungs)
    engine_nc = {"kind": "engine_nc", "regime": "no compile+graph",
                 "cmd": [*base, f"implementations=[{engine}]", "compile=false", "cudagraph=manual", f"name_suffix=ladder_{g.id}_nc"]}
    eager_graph = {"kind": "eager_graph", "regime": "eager+graph",
                   "cmd": [*base, "implementations=[pytorch]", "compile=false", "cudagraph=manual", f"name_suffix=ladder_{g.id}_eg"]}
    eager = {"kind": "eager", "regime": "eager",
             "cmd": [*base, "implementations=[pytorch]", "compile=false", "cudagraph=disabled", "+allow_eager=true",
                     f"name_suffix=ladder_{g.id}_e"]}
    return [main, engine_nc, eager_graph, eager]


def provenance() -> dict:
    def git(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()

    return {
        "commit": os.environ.get("LADDER_COMMIT") or git("rev-parse", "HEAD"),
        "branch": os.environ.get("LADDER_BRANCH") or git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": False if os.environ.get("LADDER_COMMIT") else bool(git("status", "--porcelain", "--untracked-files=no")),
        "date": _dt.datetime.now().isoformat(timespec="seconds"), "host": os.uname().nodename,
        "slurm_job": os.environ.get("SLURM_JOB_ID", ""), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="ascii") as f:
        return list(csv.DictReader(f))


def run_one(spec: dict, g: Group, mode: str, out: Path, log) -> dict:
    """Run one bench.py process, copy its CSV into ``out``, return the index record. A row that failed under CUDA-graph capture is
    re-measured once without the graph (``regime`` says so); an ``unsupported`` row is a fact about the rung, not retried."""
    cmd = spec["cmd"]
    print("+", " ".join(cmd[2:]), flush=True)
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": f"{REPO}/src:{REPO}"})
    log.write(f"\n##### {g.id} {mode} {spec['kind']}\n$ {' '.join(cmd)}\n{proc.stdout[-6000:]}\n{proc.stderr[-3000:]}\n")
    log.flush()
    m = re.findall(r"^wrote (.+\.csv)\s*$", proc.stdout, re.M)
    rec = {"group": g.id, "mode": mode, "kind": spec["kind"], "regime": spec["regime"], "returncode": proc.returncode, "csv": ""}
    if not m:
        errs = [l for l in (proc.stderr + proc.stdout).splitlines() if re.search(r"Error|Exception|Traceback", l)]
        rec["error"] = " | ".join(errs[-3:])[-500:] or (proc.stderr or proc.stdout)[-300:].strip().replace("\n", " | ")
        return rec
    dst = out / "csv" / f"{g.id}__{mode}__{spec['kind']}.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(m[-1], dst)
    rec["csv"] = str(dst.relative_to(out))
    rows = read_csv(dst)
    #: failures that a CUDA graph (or the compiled graph around it) can cause and the same step without the graph survives; any other failure
    #: (a dtype contract, a refused call, an OOM) is a fact about the rung and is not retried
    graph_related = re.compile(r"timed execution|capture|graph|Graph|stream|Higher Order|Dynamo|torch\._check|data-dependent", re.I)
    failed = [r for r in rows if r["status"] == "failed" and graph_related.search(r["error"])]
    if failed and spec["kind"] in ("rungs", "engine_nc") and "cudagraph=manual" in cmd:
        # retry the failed (impl, L) cells without the graph: some training steps cannot be captured (host-side autograd work)
        retry = [c for c in cmd if not c.startswith(("cudagraph=", "implementations=", "+seq_len_values=", "name_suffix="))]
        impls = ",".join(sorted({r["implementation"] for r in failed}))
        lens = ",".join(sorted({r["seq_len"] for r in failed}, key=int))
        retry += [f"implementations=[{impls}]", f"+seq_len_values=[{lens}]", "cudagraph=disabled", f"name_suffix=ladder_{g.id}_{spec['kind']}_retry"]
        if "compile=false" in cmd:
            retry.append("+allow_eager=true")
        sub = run_one({"kind": "retry", "regime": "compile, no graph", "cmd": retry}, g, mode, out, log)
        rec["retry"] = sub
    # a CUDA error (illegal address, invalid argument, ...) of a row that shared its process with other rungs may be state left by an earlier rung
    # (a portable kernel's cached launch parameters read by the CUDA path, a poisoned context): measure that rung alone in a fresh process, with the
    # kernel probe off (the profiler is the other thing that shares the process). `isolated` records say so in the report.
    cuda_err = re.compile(r"CUDA error|AcceleratorError|illegal memory|invalid argument|cudaError", re.I)
    shared = re.search(r"implementations=\[([^\]]*)\]", " ".join(cmd))
    n_impl = len(shared.group(1).split(",")) if shared else 1
    bad = sorted({r["implementation"] for r in rows if r["status"] == "failed" and cuda_err.search(r["error"])})
    if bad and n_impl > 1 and spec["kind"] in ("rungs", "engine_nc"):
        rec["isolated"] = []
        for impl in bad:
            lens = ",".join(sorted({r["seq_len"] for r in rows if r["implementation"] == impl and r["status"] == "failed"}, key=int))
            iso = [c for c in cmd if not c.startswith(("implementations=", "+seq_len_values=", "name_suffix=", "+kernel_probe="))]
            iso += [f"implementations=[{impl}]", f"+seq_len_values=[{lens}]", "+kernel_probe=false", f"name_suffix=ladder_{g.id}_{spec['kind']}_iso_{impl}"]
            rec["isolated"].append(run_one({"kind": "iso", "regime": "own process, no probe", "cmd": iso}, g, mode, out, log))
    return rec


def cmd_plan(args) -> int:
    out = Path(args.out)
    n = 0
    for g in select(args):
        for mode in g.modes:
            for spec in commands(g, mode, args.repeats, out):
                n += 1
            print(f"{g.id:18s} {mode:9s} {g.target:38s} {g.dtype} L={list(g.lengths)} {g.desc}")
    rows = sum(len(g.lengths) * len(g.modes) * (len(g.rungs) + 2) for g in select(args))
    print(f"\n{len(select(args))} groups, {n} bench.py processes, {rows} measured rows (rungs + 2 eager references), repeats={args.repeats}")
    return 0


def cmd_run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.jsonl"
    meta_path = out / "provenance.json"
    meta = provenance()
    meta["args"] = {"tier": args.tier, "group": args.group, "repeats": args.repeats}
    meta_path.write_text(json.dumps(meta, indent=2))
    log = (out / "run.log").open("a")
    done = set()
    if args.resume and index_path.exists():
        done = {(json.loads(l)["group"], json.loads(l)["mode"], json.loads(l)["kind"]) for l in index_path.read_text().splitlines() if l.strip()}
    bad = 0
    for g in select(args):
        for mode in g.modes:
            if args.modes and mode not in args.modes.split(","):
                continue
            recs = {}
            for spec in commands(g, mode, args.repeats, out):
                if args.kinds and spec["kind"] not in args.kinds.split(","):
                    continue
                if (g.id, mode, spec["kind"]) in done:
                    continue
                rec = run_one(spec, g, mode, out, log)
                recs[spec["kind"]] = rec
                with index_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                bad += int(not rec["csv"])
                if spec["kind"] == "engine_nc" and g.precision != "32" and PARAM_VARIANT not in g.id and "portable" in g.rungs:
                    # a bf16-mixed (fp32 master parameters) cell whose portable rung fails in BOTH regimes at some L: re-run the whole group
                    # with bf16 parameters, the layout the portable kernels take
                    ok, dtype_fail = set(), False
                    for k in ("rungs", "engine_nc"):
                        if recs.get(k, {}).get("csv"):
                            for r in read_csv(out / recs[k]["csv"]):
                                if r["implementation"] == "portable":
                                    if r["status"] == "ok":
                                        ok.add(r["seq_len"])
                                    elif re.search(r"same dtype|scalar type|dtype", r["error"]):
                                        dtype_fail = True
                    if dtype_fail and {str(x) for x in g.lengths} - ok:
                        g2 = group_of(g.id + PARAM_VARIANT)
                        for spec2 in commands(g2, mode, args.repeats, out)[:3]:
                            rec2 = run_one(spec2, g2, mode, out, log)
                            with index_path.open("a") as f:
                                f.write(json.dumps(rec2) + "\n")
    print(f"done; {bad} bench.py processes produced no CSV; index: {index_path}")
    return 0


# --------------------------------------------------------------------------------------------------------------------------------------
# reading back: the ladder table
# --------------------------------------------------------------------------------------------------------------------------------------
REF_PATH = re.compile(r"reference|\.torch|torch\.nn|pytorch", re.I)


def _cell(rows: list[dict]) -> dict:
    """One (impl, L) measurement: value ms, spread, regime, execution path, top kernels."""
    r = rows[-1]
    val = float(r["value"]) if r["status"] == "ok" and r["value"] not in ("", None) else None
    ev = {}
    try:
        ev = json.loads(r.get("execution_validation") or "{}")
    except json.JSONDecodeError:
        pass
    rep = ev.get("repeat_ms") or ([val] if val is not None else [])
    spread = (max(rep) - min(rep)) / val if (val and len(rep) > 1) else 0.0
    return {"ms": val, "spread": spread, "status": r["status"], "error": r.get("error", ""), "path": r.get("execution_path", ""),
            "graph": r.get("cudagraph", ""), "compiled": r.get("compiled", ""), "kernels": ev.get("top_kernels", []),
            "dev": r.get("device", ""), "torch": r.get("torch_version", ""), "cos": r.get("output_cosine", "")}


def load(outs) -> tuple[list[dict], dict]:
    """The ladder rows of one or several run directories (``index.jsonl`` + ``csv/``). Later directories replace earlier ones for the same
    (group, mode, kind): a re-run of a fixed group goes in its own directory and wins."""
    outs = [Path(o) for o in ([outs] if isinstance(outs, (str, Path)) else outs)]
    meta: dict = {}
    recs: dict[tuple, tuple[dict, Path]] = {}
    for out in outs:
        m = json.loads((out / "provenance.json").read_text()) if (out / "provenance.json").exists() else {}
        meta = {**m, "sources": [*meta.get("sources", []), {"dir": str(out), **{k: m.get(k) for k in ("commit", "date", "slurm_job")}}]}
        here = [json.loads(line) for line in (out / "index.jsonl").read_text().splitlines() if line.strip()]
        rerun = {(r["group"].partition("@")[0], r["mode"]) for r in here}
        for k in [k for k in recs if (k[0].partition("@")[0], k[1]) in rerun]:
            del recs[k]                      # a group re-run in a later directory replaces ALL its earlier records, bf16-params variants too
        for rec in here:
            recs[(rec["group"], rec["mode"], rec["kind"])] = (rec, out)
    cells: dict[tuple, dict] = {}   # (group, mode, L, rung) -> cell
    for rec, out in recs.values():
        g = group_of(rec["group"])
        if g is None:
            continue
        for sub, is_retry in ((rec, False), (rec.get("retry"), True), *[(i, True) for i in rec.get("isolated", [])]):
            if not sub or not sub.get("csv"):
                continue
            by: dict[tuple, list[dict]] = {}
            for r in read_csv(out / sub["csv"]):
                by.setdefault((int(r["seq_len"]), r["implementation"]), []).append(r)
            for (L, impl), rows in by.items():
                if rec["kind"] in ("eager_graph", "eager"):
                    rung = "pt_" + rec["kind"]
                else:
                    rung = {"pytorch": "pt_compiled", "portable": "portable", "miniworld": "cuda"}.get(impl, impl)
                c = _cell(rows)
                c["regime"] = (sub.get("regime") if is_retry else rec["regime"]) or rec["regime"]
                key = (g.id, rec["mode"], L, rung)
                old_c = cells.get(key)
                if old_c is None or old_c["ms"] is None or (c["ms"] is not None and c["ms"] < old_c["ms"]):
                    cells[key] = c          # the faster regime of a rung wins (compiled vs not compiled for the engine rungs)
    table = []
    keys = sorted({(k[0], k[1], k[2]) for k in cells},
                  key=lambda k: ([x.id for x in GROUPS].index(k[0].partition("@")[0]), k[0], k[1], k[2]))
    for gid, mode, L in keys:
        g = group_of(gid)
        c = {r: cells.get((gid, mode, L, r)) for r in ("pt_compiled", "pt_eager_graph", "pt_eager", "portable", "cuda")}
        ms = {r: (v["ms"] if v else None) for r, v in c.items()}
        pts = {r: ms[r] for r in ("pt_compiled", "pt_eager_graph", "pt_eager") if ms[r] is not None}
        pt_best_name = min(pts, key=pts.get) if pts else None
        pt = pts[pt_best_name] if pt_best_name else None
        row = {
            "group": gid, "op": g.op, "mode": mode, "dtype": g.dtype, "L": L, "shape": g.desc, "weight": g.weight, "target": g.target,
            "pt_compiled_ms": ms["pt_compiled"], "pt_eager_graph_ms": ms["pt_eager_graph"], "pt_eager_ms": ms["pt_eager"],
            "pt_best": pt_best_name or "", "pytorch_ms": pt, "portable_ms": ms["portable"], "cuda_ms": ms["cuda"],
        }
        row["cuda_over_portable"] = ms["cuda"] / ms["portable"] if ms["cuda"] and ms["portable"] else None
        row["portable_over_pytorch"] = ms["portable"] / pt if ms["portable"] and pt else None
        row["cuda_over_pytorch"] = ms["cuda"] / pt if ms["cuda"] and pt else None
        spreads = [v["spread"] for r, v in c.items() if v and r in ("portable", "cuda", "pt_compiled", "pt_eager_graph", "pt_eager")]
        row["noise"] = max(spreads) if spreads else 0.0
        row["paths"] = {r: (v["path"] if v else "") for r, v in c.items()}
        row["regimes"] = {r: (v["regime"] if v else "") for r, v in c.items()}
        row["kernels"] = {r: v["kernels"] for r, v in c.items() if v}
        row["errors"] = {r: v["error"] for r, v in c.items() if v and v["ms"] is None}
        row["dev"] = next((v["dev"] for v in c.values() if v and v["dev"]), "")
        row["torch"] = next((v["torch"] for v in c.values() if v and v["torch"]), "")
        row.update(verdict(row, g))
        table.append(row)
    return table, meta


def _short(err: str) -> str:
    err = " ".join((err or "").split())
    m = re.search(r"(Both operands must be same dtype[^.]*|expected scalar type[^|]*|OutOfMemoryError[^|]*|timed execution differs[^|]*)", err)
    return (m.group(1) if m else err)[:90]


def verdict(row: dict, g: Group) -> dict:
    """Per-pair verdicts + the combined string. ratio < 1 passes; the rungs a group lacks by design are not 'missing'.

    A pair is *not measured* (n.a.) when a rung has no number (the bench failed or the op refused: the reason is in ``note``), or when two
    rungs launched the very same kernels (the 'cuda' row of an op with no distinct CUDA path, or a 'portable' row that fell back to the
    PyTorch statements): that is not a measurement of the rung it names."""
    notes: list[str] = []
    has_portable = "portable" in g.rungs
    cp, pp, cpt = row["cuda_over_portable"], row["portable_over_pytorch"], row["cuda_over_pytorch"]
    noise = row["noise"]
    why: dict[str, str] = {}

    def names(rung):
        return {k[0] for k in row["kernels"].get(rung, []) if not k[0].startswith(("Memcpy", "Memset", "probe failed"))}

    kc, kp, kt = names("cuda"), names("portable"), names("pt_compiled") | names("pt_eager_graph")

    def overlap(a, b):
        return len(a & b) / len(a | b) if a and b else 0.0

    def same(a, b):
        """The two rungs launched (nearly) the same kernels: Jaccard overlap of their top-kernel names >= 0.7."""
        return overlap(a, b) >= 0.7

    #: the 'cuda' row ran the PyTorch statements: the engine has no CUDA path for the call and H100 / B200 route `miniworld` to the reference there.
    #: Evidence: the same kernels as the PyTorch rung, or (three-rung ops) a time within 3% of the PyTorch rung while the portable rung differs
    cuda_is_pytorch = False
    if has_portable and row["cuda_ms"] and row["pytorch_ms"]:
        t = row["cuda_ms"] / row["pytorch_ms"]
        declined = bool(re.search(r"refused|declined|neither the H100|not taken|reference composition|\[statements", row.get("paths", {}).get("cuda", "")))
        cuda_is_pytorch = (declined or same(kc, kt) or (abs(t - 1.0) <= 0.03 and overlap(kc, kt) >= 0.3)) and not same(kc, kp)
    if has_portable and same(kc, kp):
        cp = None
        why["cuda_vs_portable"] = "cuda rung launched the portable kernels (no distinct CUDA path)"
    elif cuda_is_pytorch:
        cp = None
        gap = row["portable_ms"] / row["cuda_ms"] if row["portable_ms"] and row["cuda_ms"] else None
        why["cuda_vs_portable"] = ("no CUDA path serves this call: `miniworld` ran the PyTorch reference"
                                   + (f" (the portable kernels are {1 / gap:.2f}x faster than what H100 runs)" if gap and gap < 1 else ""))
    if has_portable and same(kp, kt) and not cuda_is_pytorch:
        pp = None
        why["portable_vs_pytorch"] = "portable rung launched the PyTorch kernels (fell back to the PyTorch statements)"
    if not has_portable and same(kc, kt):
        cpt = None
        why["cuda_vs_pytorch"] = "cuda rung launched the PyTorch kernels (no CUDA path served the call)"
    no_cuda_path = cuda_is_pytorch
    pairs = [("cuda_vs_portable", cp), ("portable_vs_pytorch", pp)] if has_portable else [("cuda_vs_pytorch", cpt)]
    rung_ms = {"cuda": row["cuda_ms"], "portable": row["portable_ms"], "pytorch": row["pytorch_ms"]}
    rung_err = {"cuda": row["errors"].get("cuda", ""), "portable": row["errors"].get("portable", ""),
                "pytorch": row["errors"].get("pt_compiled", "")}
    parts = {"cuda_vs_portable": ("cuda", "portable"), "portable_vs_pytorch": ("portable", "pytorch"), "cuda_vs_pytorch": ("cuda", "pytorch")}
    v, na, causes = [], [], {}
    for name, r in pairs:
        if r is None:
            reason = why.get(name)
            if not reason:
                gone = [x for x in parts[name] if rung_ms[x] is None]
                reason = "; ".join(f"{x}: {_short(rung_err[x]) or 'no measurement'}" for x in gone)
            na.append(f"{name}: {reason}")
            continue
        if abs(r - 1.0) <= max(noise, 0.02):
            notes.append(f"{name} within noise (ratio {r:.3f}, spread {noise:.3f})")
        if r >= 1.0:
            v.append(f"VIOLATION({name})")
            causes[name] = ("no CUDA path: `miniworld` runs the PyTorch reference" if (name == "cuda_vs_portable" and no_cuda_path)
                            else {"cuda_vs_portable": "CUDA kernel slower than the portable kernel",
                                  "portable_vs_pytorch": "portable path slower than PyTorch",
                                  "cuda_vs_pytorch": "CUDA kernel slower than PyTorch"}[name])
    if v:
        out = "+".join(v)
    elif not na:
        out = "OK"
    elif cuda_is_pytorch and len(na) == 1:
        out = "n.a.(cuda rung missing)"  # the other pair was measured and holds; the routing gap is in the note and in the routing-gap table
    else:
        out = "n.a." if len(na) == len(pairs) else "OK"
    if na:
        notes.append("n.a.: " + " | ".join(na))
    if not has_portable:
        notes.append("no portable rung by design")
    return {"verdict": out, "note": "; ".join(notes), "causes": causes, "routing_gap": cuda_is_pytorch}


def fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if x < 100 else f"{x:.1f}")


def violations(table, kinds=("cuda_vs_portable", "portable_vs_pytorch", "cuda_vs_pytorch")):
    out = []
    for r in table:
        for k, ratio in (("cuda_vs_portable", r["cuda_over_portable"]), ("portable_vs_pytorch", r["portable_over_pytorch"]),
                         ("cuda_vs_pytorch", r["cuda_over_pytorch"])):
            if f"VIOLATION({k})" in r["verdict"] and k in kinds and ratio is not None:
                out.append({**r, "kind": k, "ratio": ratio, "cause": r.get("causes", {}).get(k, "")})
    return out


FIELDS = ["group", "op", "mode", "dtype", "L", "shape", "weight", "pt_compiled_ms", "pt_eager_graph_ms", "pt_eager_ms", "pt_best",
          "portable_ms", "cuda_ms", "cuda_over_portable", "portable_over_pytorch", "cuda_over_pytorch", "noise", "verdict", "routing_gap", "cause", "note",
          "dev", "torch", "target"]


def write_csv(table, path: Path, meta: dict) -> None:
    with path.open("w", newline="") as f:
        f.write("# " + json.dumps({k: meta.get(k) for k in ("commit", "branch", "dirty", "date", "host")}) + "\n")
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in table:
            r = {**r, "cause": "; ".join(f"{k}: {v}" for k, v in r.get("causes", {}).items())}
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k in FIELDS})


def read_ladder_csv(path: Path) -> list[dict]:
    lines = [l for l in path.read_text().splitlines() if not l.startswith("#")]
    rows = list(csv.DictReader(lines))
    for r in rows:
        for k in ("cuda_over_portable", "portable_over_pytorch", "cuda_over_pytorch", "noise", "pt_compiled_ms", "pt_eager_graph_ms",
                  "pt_eager_ms", "portable_ms", "cuda_ms"):
            r[k] = float(r[k]) if r.get(k) not in (None, "", "-") else None
        r["weight"] = int(r["weight"])
    return rows


def render_md(table, meta) -> str:
    L = []
    dev = next((r["dev"] for r in table if r["dev"]), "?")
    tv = next((r["torch"] for r in table if r["torch"]), "?")
    L.append(f"Provenance: commit `{meta.get('commit', '?')[:8]}` (branch `{meta.get('branch', '?')}`, dirty={meta.get('dirty')}), card **{dev}**, "
             f"torch {tv}, run {meta.get('date', '?')} on {meta.get('host', '?')}, Slurm job {meta.get('slurm_job', '?')}.\n")
    n = len(table)
    ok = sum(r["verdict"] == "OK" for r in table)
    na = sum(r["verdict"].startswith("n.a.") and not r.get("routing_gap") for r in table)
    gaps = [r for r in table if r.get("routing_gap")]
    cvp = [r for r in table if "VIOLATION(cuda_vs_portable)" in r["verdict"]]
    pvp = [r for r in table if "VIOLATION(portable_vs_pytorch)" in r["verdict"]]
    cvt = [r for r in table if "VIOLATION(cuda_vs_pytorch)" in r["verdict"]]
    L.append("## Summary\n")
    L.append("| rows | OK | VIOLATION(cuda_vs_portable) | VIOLATION(portable_vs_pytorch) | VIOLATION(cuda_vs_pytorch) | n.a.(cuda rung missing: no CUDA path, `miniworld` runs PyTorch) | n.a.(other rung missing) |\n|---|---|---|---|---|---|---|")
    L.append(f"| {n} | {ok} | {len(cvp)} | {len(pvp)} | {len(cvt)} | {len(gaps)} | {na} |\n")
    # per op counts
    by: dict[str, list[dict]] = {}
    for r in table:
        by.setdefault(r["op"], []).append(r)
    L.append("| op | rows | OK | cuda>=portable | portable>=pytorch | cuda>=pytorch (no portable) | no CUDA path (routing gap) | other n.a. |\n|---|---|---|---|---|---|---|---|")
    for op, rs in by.items():
        L.append(f"| {op} | {len(rs)} | {sum(r['verdict'] == 'OK' for r in rs)} | "
                 f"{sum('VIOLATION(cuda_vs_portable)' in r['verdict'] for r in rs)} | {sum('VIOLATION(portable_vs_pytorch)' in r['verdict'] for r in rs)} | "
                 f"{sum('VIOLATION(cuda_vs_pytorch)' in r['verdict'] for r in rs)} | {sum(bool(r.get('routing_gap')) for r in rs)} | "
                 f"{sum(r['verdict'].startswith('n.a.') and not r.get('routing_gap') for r in rs)} |")
    L.append("")
    v = violations(table)
    L.append("## Routing gaps: the CUDA rung does not exist, `miniworld` on this card runs the PyTorch reference\n")
    L.append("Rows where the hand CUDA path does not serve the call (a dtype, a shape, a mode, an option it does not take) and the engine's portable kernels would be faster than "
             "what `miniworld` runs. The ladder invariant is not testable for them (no CUDA rung); the routing is what a user sees. `speed-up` = PyTorch / portable.\n")
    L.append("| op | mode | dtype | shape | L | miniworld (= PyTorch) ms | portable ms | speed-up lost | weight |\n|---|---|---|---|---|---|---|---|---|")
    for r in sorted([r for r in table if r.get("routing_gap") and r["portable_ms"] and r["cuda_ms"] > r["portable_ms"] * 1.01],
                    key=lambda r: -(r["cuda_ms"] / r["portable_ms"])):
        L.append(f"| {r['op']} | {r['mode']} | {r['dtype']} | {r['shape']} | {r['L']} | {fmt(r['cuda_ms'])} | {fmt(r['portable_ms'])} | {r['cuda_ms'] / r['portable_ms']:.2f}x | {r['weight']} |")
    L.append("")
    L.append("## Violations ranked by size (ratio = lower rung / upper rung; must be < 1)\n")
    L.append("| # | op | mode | dtype | shape | L | pair | ratio | CUDA ms | portable ms | PyTorch best ms (form) | weight | cause / note |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(sorted(v, key=lambda r: -r["ratio"]), 1):
        L.append(f"| {i} | {r['op']} | {r['mode']} | {r['dtype']} | {r['shape']} | {r['L']} | {r['kind']} | {r['ratio']:.3f} | {fmt(r['cuda_ms'])} | "
                 f"{fmt(r['portable_ms'])} | {fmt(r['pytorch_ms'])} ({r['pt_best']}) | {r['weight']} | {r['cause']}{'; ' + r['note'] if r['note'] else ''} |")
    L.append("")
    L.append("## Violations ranked by importance (production weight, then size)\n")
    L.append("| # | weight | op | mode | dtype | shape | L | pair | ratio |\n|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(sorted(v, key=lambda r: (-r["weight"], -r["ratio"])), 1):
        L.append(f"| {i} | {r['weight']} | {r['op']} | {r['mode']} | {r['dtype']} | {r['shape']} | {r['L']} | {r['kind']} | {r['ratio']:.3f} |")
    L.append("")
    L.append("## Violation detail: where the time goes (top GPU kernels of one call, microseconds x calls)\n")
    L.append("`torch.profiler` over one call of the timed step, in the regime of the row. A rung that launches many small kernels, or spends its time in "
             "a cuBLAS GEMM the other rung fuses, shows it here.\n")
    seen = set()
    for r in sorted(v, key=lambda r: (-r["weight"], -r["ratio"])):
        tag = (r["group"], r["mode"], r["L"], r["kind"])
        if tag in seen:
            continue
        seen.add(tag)
        a, b = {"cuda_vs_portable": ("cuda", "portable"), "portable_vs_pytorch": ("portable", "pt_compiled"),
                "cuda_vs_pytorch": ("cuda", "pt_compiled")}[r["kind"]]
        L.append(f"* **{r['op']} {r['mode']} {r['dtype']} {r['shape']} L={r['L']}**, {r['kind']} ratio {r['ratio']:.3f}")
        for rung in (a, b):
            ks = r["kernels"].get(rung, [])
            L.append(f"  * {rung}: " + (", ".join(f"`{k[0][:40]}` {k[1]:.0f}us x{k[2]}" for k in ks[:5]) or "(no profile)"))
    L.append("")
    L.append("## Full ladder table (ms; CUDA-graph replay of the compiled step, median of 3)\n")
    L.append("`pt_c` = PyTorch compiled + graph, `pt_eg` = eager statements in a graph, `pt_e` = plain eager; the PyTorch rung is the fastest of the three. "
             "`c/p` = CUDA / portable, `p/t` = portable / PyTorch, `c/t` = CUDA / PyTorch.\n")
    L.append("| op | mode | dtype | shape | L | pt_c | pt_eg | pt_e | portable | CUDA | c/p | p/t | c/t | verdict | note |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in table:
        L.append(f"| {r['op']} | {r['mode']} | {r['dtype']} | {r['shape']} | {r['L']} | {fmt(r['pt_compiled_ms'])} | {fmt(r['pt_eager_graph_ms'])} | "
                 f"{fmt(r['pt_eager_ms'])} | {fmt(r['portable_ms'])} | {fmt(r['cuda_ms'])} | {fmt(r['cuda_over_portable'])} | "
                 f"{fmt(r['portable_over_pytorch'])} | {fmt(r['cuda_over_pytorch'])} | {r['verdict']} | {r['note']} |")
    return "\n".join(L) + "\n"


def cmd_report(args) -> int:
    outs = [Path(o) for o in args.out]
    table, meta = load(outs)
    csv_path = Path(args.csv) if args.csv else outs[-1] / "ladder.csv"
    write_csv(table, csv_path, meta)
    md = render_md(table, meta)
    Path(args.md).write_text(md) if args.md else (outs[-1] / "ladder.md").write_text(md)
    n = len(table)
    print(f"{n} ladder rows -> {csv_path}")
    for key in ("OK", "VIOLATION(cuda_vs_portable)", "VIOLATION(portable_vs_pytorch)", "VIOLATION(cuda_vs_pytorch)", "n.a."):
        print(f"  {key:34s} {sum(key in r['verdict'] for r in table)}")
    print(f"  of which routing gaps (no CUDA path)  {sum(bool(r.get('routing_gap')) for r in table)}")
    return 0


def cmd_gate(args) -> int:
    """Exit 1 if any row is a violation that is not on the allow list (one ``group:mode:L:pair`` per line); ``--strict`` ignores the list."""
    rows = read_ladder_csv(Path(args.csv))
    allow = set()
    if args.allow and not args.strict:
        allow = {l.strip() for l in Path(args.allow).read_text().splitlines() if l.strip() and not l.startswith("#")}
    bad = []
    if args.routing:
        for r in rows:
            if r.get("routing_gap") == "True" and r["cuda_ms"] and r["portable_ms"]:
                tag = f"{r['group']}:{r['mode']}:{r['L']}:routing_gap"
                if tag not in allow:
                    bad.append((tag, r["cuda_ms"] / r["portable_ms"]))
    for r in violations(rows):
        tag = f"{r['group']}:{r['mode']}:{r['L']}:{r['kind']}"
        if tag not in allow:
            bad.append((tag, r["ratio"]))
    for tag, ratio in sorted(bad, key=lambda x: -x[1]):
        print(f"VIOLATION {tag} ratio {ratio:.3f}")
    print(f"{len(rows)} rows, {len(bad)} violation(s) not allowed")
    return 1 if bad else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ladder", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("plan", cmd_plan), ("run", cmd_run), ("report", cmd_report), ("gate", cmd_gate)):
        s = sub.add_parser(name)
        s.set_defaults(func=fn)
        if name in ("plan", "run"):
            s.add_argument("--tier", default="core", choices=list(TIERS), help="smoke: one cell per op; core: production shapes; full: all served")
            s.add_argument("--group", default="", help="regex over group ids (overrides --tier)")
            s.add_argument("--repeats", type=int, default=3)
            s.add_argument("--modes", default="", help="comma list of modes to run (default: every mode of the group)")
            s.add_argument("--kinds", default="", help="comma list of command kinds to run: rungs, engine_nc, eager_graph, eager (default: all four)")
            s.add_argument("--out", default=str(Path.home() / "miniworld-engine-scratch" / "ladder" / "out"))
            if name == "run":
                s.add_argument("--resume", action="store_true")
        if name == "report":
            s.add_argument("--out", required=True, nargs="+", help="run directories; a later one replaces an earlier one per (group, mode, kind)")
            s.add_argument("--md", default="")
            s.add_argument("--csv", default="")
        if name == "gate":
            s.add_argument("--csv", required=True)
            s.add_argument("--allow", default="")
            s.add_argument("--strict", action="store_true")
            s.add_argument("--routing", action="store_true", help="also fail on a routing gap: no CUDA path and a faster portable path (tag group:mode:L:routing_gap)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
