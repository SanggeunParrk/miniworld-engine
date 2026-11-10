"""The A100 (sm_80) port's switches, at the defaults the pre-redesign A100 port shipped (2.4.1 + sm80, ``settings.py``).

The sm_80 kernels and integrations were ported as they were; they read their A/B switches through ``settings.current()``. The redesign
has no run-time switches: nothing here is configured by the engine, the measurers or a module, so every sm_80 path runs at these defaults.
``configure`` remains for kernel experiments only.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

AutotuneKernel = Literal["transition", "tri_attention", "adaln"]
DispatchMode = Literal["auto", "off", "force"]


@dataclasses.dataclass(frozen=True)
class Settings:
    engine_backend: Literal["auto", "triton"] = "auto"   # the pre-redesign engine-wide Triton switch
    autotune_kernels: frozenset[AutotuneKernel] = frozenset()
    run_autotune: bool = False
    capture: bool = False
    fill_gaps: bool = False
    compile_jobs: int | None = None
    predict_unusable: bool = False
    bench_clear_mb: int = 0
    bench_rep_ms: int = 0
    bench_lock: str = ""
    autotune_miss_cap: int = 24
    autotune_on_miss_shards: str = ""
    compile_wrap: Literal["disable", "custom_op"] = "custom_op"
    biasonly_dispatch: DispatchMode = "auto"
    layernorm_dispatch: DispatchMode = "auto"
    pin_gate_backend: Literal["fused", "split"] | None = None
    pin_infer_concat: bool | None = None
    force_portable_kernels: bool = False
    portable_on_hopper_blackwell: bool = False
    trimul_h100_training_widths: tuple[int, ...] = (64, 128, 256, 384, 512)
    transition_cuda_b2b: bool = True
    augmented_attention_bf16_sm100: bool = True
    transition_fuse_stats: bool = False
    transition_lnbwd_privatize: bool = True
    layernorm_bwd_path: Literal["persistent", "atomic", "cuda"] | None = None
    layernorm_out_bwd_path: str | None = None
    trimul_impl: str | None = None
    swa_dit_fused: bool = True
    b200_portable_kernels: bool = False
    rope_angles_bf16_compat: bool = False
    swa_dit_rows_impl: Literal["auto", "slab", "wgmma"] = "auto"
    swa_dit_ffn_dw: Literal["mat", "fused"] = "mat"
    swa_dit_dq1: Literal["bf16", "fp32"] = "bf16"
    typecheck: bool = False
    swa_check_front_packed: bool = False
    swa_flash_saves_lse: bool = True
    bias_only_dit: bool = True
    bias_only_dit_train: bool = True
    bias_only_dit_train_pack1: bool = True
    bias_only_dit_bwd_dg: bool = True
    bias_only_dit_bwd_fused: bool = True
    bias_only_dit_bwd_pbfin: bool = True
    bias_only_dit_bwd_mid: bool = False
    bias_only_dit_bwd_pvdpb: bool = True
    bias_only_dit_pair_exact: bool = False
    bias_only_dit_core: bool = True
    bias_only_dit_inf3: bool = True
    bias_only_dit_inf3_2cta: bool = True
    bias_only_dit_inf3_bf16: bool = True
    bias_only_dit_inf3_p_early: bool = True
    bias_only_dit_pv_batch: bool = True
    bias_only_dit_tf32: bool = True
    bias_only_dit_cond: bool = False
    bias_only_dit_resln: bool = False
    local_dit_hoist: bool = True
    local_dit_tf32_fused: bool = True
    local_dit_tf32: bool = True
    local_dit_sm100: bool = True
    opm_train_save_o: bool = True
    token_dit_train: bool = True
    swa_dit_sm100: bool = True
    swa_dit_tf32: bool = True
    swa_global_bf16: bool = False
    transition_fused_sm100a: bool = True
    transition_fused_sm80: bool = True
    transition_wide_save_h: bool = False
    transition_generic_small: bool = True
    bias_only_dit_tf32_glu: bool = False
    bias_only_dit_tf32_glu_bwd: bool = True
    static_weights: bool = False
    static_conditioning: bool = False
    bias_only_dit_pv_vb: int | None = None
    bias_only_dit_sg: int | None = None
    bias_only_dit_inf3_cl: int | None = None
    bias_only_dit_inf3_tail_cl: int | None = None
    bias_only_dit_wgrad_split: int | None = None
    bias_only_dit_dpb_pair: bool | None = None
    bias_only_dit_tf32_dpb_pair: bool | None = None
    bias_only_dit_bwd_tail: str = "split3"
    tf32_fused_units: str = "auto"
    transition_sm100_repl: int = 0
    transition_wide_small_tiles: int = 0
    transition_wide_small_fwd_tiles: int = 0
    transition_wide_repl: int = 0
    plan_cache_dir: str = ""
    transition_sm100_nvcc: str = ""
    engine_jit_root: str = ""
    jit_build_log: bool = True
    prebuilt_kernels: bool = True
    jit_fallback: bool = False   # the redesign never falls back
    opm_train: bool | None = None
    pwa_train: bool | None = None
    pwa_infer: bool | None = None
    token_pair_init_fused: bool = True
    apb_h100_exact_dwb: bool = False
    swa_dit_h100_native: bool | None = None
    a100_cleanup: bool = True
    augattn_sm80: bool = True
    augattn_sm80_fused: bool = True
    augattn_sm80_pbgen: bool = True
    augattn_sm80_pbgen_rows: int = 2
    augattn_sm80_flags: str = ""
    augattn_sm80_ops: str = "auto"
    augattn_sm80_shared_core: bool = False
    augattn_sm80_shared_score: bool = False
    augattn_sm80_shared_uniform: bool | None = None
    augattn_sm80_atom_qkvg: bool | None = None
    augattn_sm80_atom_qkvg_tiled: bool | None = None
    augattn_sm80_atom_output_fused: bool | None = None
    augattn_sm80_atom_output_tiled: bool | None = None
    dense_core_dq_order: int = 0
    dense_core_dkv_order: int = 0
    dense_e274_dq_order: int = 0
    dense_e274_dkv_order: int = 0
    apb_sm80: bool = True
    apb_sm80_train16: bool = True
    apb_sm80_typed_grad: bool = True
    apb_sm80_save_natural: bool = True
    apb_sm80_overlap: bool = True
    apb_sm80_cpp_step: bool = True
    apb_sm80_attn_bn: int = 32
    apb_sm80_bwd_mode: int = 1
    apb_sm80_bwd_pair_grid: int = 0
    apb_sm80_attn_warps: int = 0
    bias_only_attn_sm80: str = "1"
    bias_only_dit_prep: bool = True
    bias_only_dit_sm80_stages: int = 2
    bias_only_dit_sm80_sg: int | None = None
    bias_only_dit_sm80_dpb: int | None = None
    opm_sm80: bool = True
    opm_sm80_save_o: bool = True
    opm_sm80_pipeline: bool = False
    opm_sm80_native128_pipeline: bool = True
    opm_sm80_typed_grad: bool = False
    opm_sm80_mask_count_native: bool | None = None
    opm_sm80_mask_epilogue: bool | None = None
    opm_sm80_pl: int = 0
    opm_sm80_epi: int = 0
    opm_sm80_dg: int = 0
    opm_sm80_pb: int = 0
    pwa_sm80: bool = True
    pwa_sm80_broadcast_native: bool = True
    pwa_sm80_broadcast_pipeline: str = "0"
    pwa_sm80_contract_gate_fused: bool | None = None
    pwa_sm80_dw_short: bool = False
    pwa_sm80_dw_native: bool | None = None
    pwa_sm80_typed_grad: bool | None = None
    pwa_sm80_glue_tuned: bool | None = None
    pwa_sm80_glue_pipeline: bool = True
    pwa_sm80_lt: bool = True
    pwa_sm80_grad_reduce: bool | None = None
    pwa_sm80_pair_partial: bool = True
    pwa_sm80_split: int | None = None
    pwa_sm80_pb: int = 0
    triattn_sm80: bool = True
    triattn_sm80_compact: bool = True
    triattn_sm80_internal: bool = True
    triattn_sm80_tuned: bool = True
    triattn_sm80_softmax: bool = True
    triattn_sm80_gemm_pad: bool = True
    triattn_sm80_out_lt: bool = True
    triattn_sm80_native_rng: bool = True
    triattn_sm80_train_capture_aware: bool = True
    triattn_sm80_train_inner: bool = False
    triattn_sm80_rows: bool = False
    triattn_sm80_wide: bool = False
    triattn_sm80_dq_rows: bool | None = None
    triattn_sm80_native_db: bool | None = None
    triattn_sm80_native_wo: bool | None = None
    triattn_sm80_native_out: bool | None = None
    triattn_sm80_exact_gamma: str = "2"
    triattn_sm80_flags: str = ""
    trimul_sm80: bool = True
    trimul_sm80_wide: bool = True
    trimul_sm80_wide512: bool = True
    trimul_sm80_row_mask: bool = True
    trimul_sm80_tuned: bool = True
    trimul_sm80_flags: str = ""
    trimul_sm80_wide_flags: str = ""
    adaln_atom: bool = True
    adaln_atom_cfg: int = -1
    adaln_atom_tf32: bool = True
    adaln_branch: bool = True
    adaln_fp32_atom_min_rows: int = 8192
    adaln_sm80: bool = True
    adaln_sm80_atom_cond_wgrad_fused: bool = False
    adaln_sm80_flags: str = ""
    adaln_sm80_native_param_cast: bool = False
    adaln_sm80_pack_param_cast: bool = False
    adaln_sm80_save_param_cast: bool = False
    adaln_warp_blocks: int = 4
    condtrans_atom: bool = True
    condtrans_atom_tf32: bool = True
    condtrans_sm80: bool = True
    condtrans_sm80_atom_cond_wgrad_fused: bool = False
    condtrans_sm80_atom_narrow: bool | None = None
    condtrans_sm80_atom_small: bool | None = None
    condtrans_sm80_atom_split: bool | None = None
    condtrans_sm80_atom_split_cfg: int | None = None
    condtrans_sm80_atom_wgrad_split: bool = False
    condtrans_sm80_atom_wgrad_wide: bool = False
    condtrans_sm80_flags: str = ""
    condtrans_sm80_native_param_cast: bool = False
    condtrans_sm80_pack_param_cast: bool = False
    condtrans_sm80_save_param_cast: bool = False
    dropout_scale_sm80: bool = False
    gated_sm80: bool = True
    lnlinear_sm80: bool = True
    norms_sm80: str = "1"
    norms_sm80_flags: str = ""
    token_dit_prep: bool = True
    transition_bwd_sm80_flags: str = ""
    transition_bwd_sm80_nrep: int = 0
    transition_fused_bwd_sm80: bool = True
    transition_fused_fwd_sm80: bool = True
    transition_fwd_sm80_flags: str = ""
    transition_wide_grad_cast: bool = False
    transition_wide_grad_fp32: bool = True
    transition_wide_save_ab: bool = True
    transition_wide_sm80: bool = True
    transition_wide_sm80_flags: str = ""
    local_dit_sm80: bool = True
    local_dit_sm80_rows: bool = True
    mpnn_edge_flags: str = ""
    mpnn_edge_ptxas: bool = False
    mpnn_edge_sm80: bool = True
    mpnn_msg_sm80: bool = True
    mpnn_msg_sm80_bwd: str = "fused"
    mpnn_msg_sm80_flags: str = ""
    mpnn_node_sm80_flags: str = ""
    sigmoid_gate_sm80: bool = True
    swa_dit_sm80: bool = True
    swa_dit_sm80_ffn_tuned: bool = True
    swa_dit_sm80_flags: str = ""
    swa_dit_sm80_infer_tuned: bool = True
    swa_dit_sm80_native_dwd: bool = True
    swa_dit_sm80_native_grad_cast: bool | None = None
    swa_dit_sm80_native_grouped5_dw: bool | None = None
    swa_dit_sm80_native_grouped8_alloc: bool = False
    swa_dit_sm80_native_grouped8_dw: bool = False
    swa_dit_sm80_native_grouped_dw: bool | None = None
    swa_dit_sm80_native_qkvg_dw: bool = False
    token_dit_sm80_alias_train_pack: bool = False
    token_dit_sm80_direct_grad: bool = True
    token_dit_sm80_native_train_pack: bool = False
    token_dit_typed_grad: bool = False

_ACTIVE = Settings()

def current() -> Settings:
    """The active settings."""
    return _ACTIVE

def configure(**kwargs) -> Settings:
    """Replace the active settings (kernel experiments only); returns the previous value."""
    global _ACTIVE
    previous = _ACTIVE
    unknown = set(kwargs) - {f.name for f in dataclasses.fields(Settings)}
    if unknown:
        raise TypeError(f"unknown setting(s): {', '.join(sorted(unknown))}")
    _ACTIVE = dataclasses.replace(previous, **kwargs)
    return previous

def reset() -> None:
    """Restore the defaults."""
    global _ACTIVE
    _ACTIVE = Settings()

def engine_jit_root():
    """The JIT build cache (``kernels._nvcc.engine_jit_root``)."""
    from miniworld_engine.kernels._nvcc import engine_jit_root as root

    return root()
