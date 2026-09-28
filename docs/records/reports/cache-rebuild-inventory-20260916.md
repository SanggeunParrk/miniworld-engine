# 캐시 재생성 목록 — 2026-09-16

재생성 전 현재 소스와 저장된 캐시 237개를 대조한 스냅샷. GPU에서 검사했으며 도구 버전 불일치는 없었다. 기존 파일 검사이므로 누락된 GPU·shape 캐시 전체를 증명하는 목록은 아니다.

## 진행 중인 빌드

- A6000: Slurm `1705443`, 소스/의존성 변경 6개 커널, 576개 기존 등록 조합만 재생성. 이전 잡 `1705081`은 중단했고, 임의로 추가했던 atom 3072/4096 전용 캡처 및 벤치 단계는 실행 전에 제거했다.
- A5000: Slurm `1705092`, 같은 6개 커널, 576개 등록 조합.
- A100: 해당 GPU가 이 클러스터에 없어 실행하지 못함.
- H100: 이 체크아웃에 해당 GPU 캐시 파일이 없어 이번 기존 파일 검사에 포함되지 않음. H100에서 별도 생성·검증 필요.
- 제출/실행은 완료를 뜻하지 않는다. 최종 상태는 각 작업의 로그와 종료 코드를 확인해야 한다.

## 1. 기존 측정을 다시 해야 하는 캐시

총 36개 GPU별 파일: NVIDIA A100 80GB PCIe (sm80): 24, NVIDIA RTX A5000 (sm86): 6, NVIDIA RTX A6000 (sm86): 6.

| 커널 | GPU | 사유 |
|---|---|---|
| `adaln_bwd_dx_dlnw_triton` | NVIDIA A100 80GB PCIe (sm80) | build_rev declared 2 (registry.csv) |
| `adaln_bwd_pre_dx_triton` | NVIDIA A100 80GB PCIe (sm80) | build_rev declared 2 (registry.csv) |
| `adaln_epilogue_saveact_triton` | NVIDIA A100 80GB PCIe (sm80) | build_rev declared 2 (registry.csv) |
| `adaln_epilogue_triton` | NVIDIA A100 80GB PCIe (sm80) | build_rev declared 2 (registry.csv) |
| `adaln_fwd_gate_triton` | NVIDIA A100 80GB PCIe (sm80) | build_rev declared 2 (registry.csv) |
| `adaln_fwd_triton` | NVIDIA A100 80GB PCIe (sm80) | build_rev declared 2 (registry.csv) |
| `adaln_gemm_gate_triton` | NVIDIA A100 80GB PCIe (sm80) | build_rev declared 2 (registry.csv) |
| `cond_transition_bwd_gemm_swiglu_triton` | NVIDIA A100 80GB PCIe (sm80) | build_rev declared 2 (registry.csv) |
| `cond_transition_bwd_swiglu_flat_triton` | NVIDIA A100 80GB PCIe (sm80) | build_rev declared 2 (registry.csv) |
| `cond_transition_expand_swiglu_saveact_triton` | NVIDIA A100 80GB PCIe (sm80) | build_rev declared 2 (registry.csv) |
| `cond_transition_expand_swiglu_triton` | NVIDIA A100 80GB PCIe (sm80) | build_rev declared 2 (registry.csv) |
| `cond_transition_fwd_b2b_saveact_triton` | NVIDIA A100 80GB PCIe (sm80) | build_rev declared 2 (registry.csv) |
| `cond_transition_fwd_b2b_triton` | NVIDIA A100 80GB PCIe (sm80) | build_rev declared 2 (registry.csv) |
| `cond_transition_squeeze_gate_saveact_triton` | NVIDIA A100 80GB PCIe (sm80) | build_rev declared 2 (registry.csv) |
| `cond_transition_squeeze_gate_triton` | NVIDIA A100 80GB PCIe (sm80) | build_rev declared 2 (registry.csv) |
| `cond_transition_swiglu_triton` | NVIDIA A100 80GB PCIe (sm80) | build_rev declared 2 (registry.csv) |
| `layernorm_fwd_strided_triton` | NVIDIA A100 80GB PCIe (sm80) | build_rev declared 2 (registry.csv) |
| `rmsnorm_adamod_bwd_triton` | NVIDIA A100 80GB PCIe (sm80) | kernel source/key changed (op_identity) |
| `rmsnorm_adamod_bwd_triton` | NVIDIA RTX A5000 (sm86) | kernel source/key changed (op_identity) |
| `rmsnorm_adamod_bwd_triton` | NVIDIA RTX A6000 (sm86) | kernel source/key changed (op_identity) |
| `rmsnorm_adamod_fwd_triton` | NVIDIA A100 80GB PCIe (sm80) | kernel source/key changed (op_identity) |
| `rmsnorm_adamod_fwd_triton` | NVIDIA RTX A5000 (sm86) | kernel source/key changed (op_identity) |
| `rmsnorm_adamod_fwd_triton` | NVIDIA RTX A6000 (sm86) | kernel source/key changed (op_identity) |
| `rmsnorm_bwd_triton` | NVIDIA RTX A5000 (sm86) | profiled kernel dependency changed (implementation) |
| `rmsnorm_bwd_triton` | NVIDIA RTX A6000 (sm86) | profiled kernel dependency changed (implementation) |
| `rmsnorm_fwd_triton` | NVIDIA RTX A5000 (sm86) | profiled kernel dependency changed (implementation) |
| `rmsnorm_fwd_triton` | NVIDIA RTX A6000 (sm86) | profiled kernel dependency changed (implementation) |
| `transition_bwd_swiglu_recompute_triton` | NVIDIA A100 80GB PCIe (sm80) | kernel source/key changed (op_identity) |
| `transition_bwd_swiglu_recompute_triton` | NVIDIA RTX A5000 (sm86) | kernel source/key changed (op_identity) |
| `transition_bwd_swiglu_recompute_triton` | NVIDIA RTX A6000 (sm86) | kernel source/key changed (op_identity) |
| `transition_expand_swiglu_triton` | NVIDIA A100 80GB PCIe (sm80) | kernel source/key changed (op_identity) |
| `transition_expand_swiglu_triton` | NVIDIA RTX A5000 (sm86) | kernel source/key changed (op_identity) |
| `transition_expand_swiglu_triton` | NVIDIA RTX A6000 (sm86) | kernel source/key changed (op_identity) |
| `trimul_bwd_gate_packed_triton` | NVIDIA A100 80GB PCIe (sm80) | kernel source/key changed (op_identity) |
| `trimul_gemm_gate_mmajor_triton` | NVIDIA A100 80GB PCIe (sm80) | kernel source/key changed (op_identity) |
| `trimul_outproj_layernorm_gemm_gate_triton` | NVIDIA A100 80GB PCIe (sm80) | kernel source/key changed (op_identity) |

## 2. 후보 설정 변경 — 증분 빌드 필요

12개 파일. 스캐너는 OK로 표시하지만 런타임은 config_space_hash 불일치 시 fallback하므로 갱신 전 튜닝 캐시 적중을 가정하면 안 된다.

| 커널 | GPU | 사유 |
|---|---|---|
| `gated_projection_bwd_dx_triton` | NVIDIA A100 80GB PCIe (sm80) | config grid changed -- incremental build pending |
| `gated_projection_bwd_gate_dropres_triton` | NVIDIA A100 80GB PCIe (sm80) | config grid changed -- incremental build pending |
| `gated_projection_bwd_gate_flat_triton` | NVIDIA A100 80GB PCIe (sm80) | config grid changed -- incremental build pending |
| `gated_projection_bwd_gate_recompute_flat_triton` | NVIDIA RTX A5000 (sm86) | config grid changed -- incremental build pending |
| `layernorm_fwd_rowscale_triton` | NVIDIA A100 80GB PCIe (sm80) | config grid changed -- incremental build pending |
| `layernorm_linear_fwd_triton` | NVIDIA A100 80GB PCIe (sm80) | config grid changed -- incremental build pending |
| `transition_bwd_epilogue_triton` | NVIDIA RTX A5000 (sm86) | config grid changed -- incremental build pending |
| `transition_bwd_transpose_packed_triton` | NVIDIA A100 80GB PCIe (sm80) | config grid changed -- incremental build pending |
| `transition_fold_triton` | NVIDIA RTX A5000 (sm86) | config grid changed -- incremental build pending |
| `transition_fwd_b2b_triton` | NVIDIA A100 80GB PCIe (sm80) | config grid changed -- incremental build pending |
| `trimul_bwd_gate_recompute_triton` | NVIDIA A100 80GB PCIe (sm80) | config grid changed -- incremental build pending |
| `trimul_gemm_gate_triton` | NVIDIA A100 80GB PCIe (sm80) | config grid changed -- incremental build pending |

## 3. 빌드 입력 변경 — coverage 확인 필요

42개 파일. 기존 커널 측정은 유지할 수 있으나 새 shape/dtype/레이아웃 조합을 커버하는지 검사하고 빈 항목만 보충해야 한다. 무조건 전체 재튜닝 대상은 아니다.

| 커널 | GPU | 사유 |
|---|---|---|
| `augmented_attention_bwd_atomic_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `augmented_attention_bwd_atomic_triton` | NVIDIA RTX A6000 (sm86) | build driver changed -- coverage may differ; rebuild the op |
| `augmented_attention_bwd_pre_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `augmented_attention_bwd_reduce_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `augmented_attention_bwd_split_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `augmented_attention_fwd_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `gated_projection_bwd_gate_flat_triton` | NVIDIA RTX A6000 (sm86) | build driver changed -- coverage may differ; rebuild the op |
| `gated_projection_bwd_gate_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `gated_projection_bwd_gate_triton` | NVIDIA RTX A6000 (sm86) | build driver changed -- coverage may differ; rebuild the op |
| `gated_projection_gate_dropres_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `gated_projection_gate_flat_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `gated_projection_gate_flat_triton` | NVIDIA RTX A6000 (sm86) | build driver changed -- coverage may differ; rebuild the op |
| `gated_projection_gate_gemm_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `gated_projection_gate_inplace_flat_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `gated_projection_gate_packed_flat_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `gated_projection_gate_res_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `gated_projection_gate_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `gated_projection_gate_triton` | NVIDIA RTX A6000 (sm86) | build driver changed -- coverage may differ; rebuild the op |
| `layernorm_bwd_atomic_strided_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `layernorm_bwd_foldstats_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `layernorm_fwd_recompute_foldstats_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `layernorm_linear_bwd_fp32_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `layernorm_linear_fwd_fp32_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `rope_fwd_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `transition_bwd_epilogue_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `transition_fold_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `transition_fwd_b2b_ktiled_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `transition_layernorm_expand_swiglu_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `triangle_attention_bwd_atomic_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `triangle_attention_bwd_atomic_triton` | NVIDIA RTX A6000 (sm86) | build driver changed -- coverage may differ; rebuild the op |
| `triangle_attention_bwd_dkdv_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `triangle_attention_bwd_dq_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `triangle_attention_bwd_pre_contig_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `triangle_attention_bwd_pre_contig_triton` | NVIDIA RTX A6000 (sm86) | build driver changed -- coverage may differ; rebuild the op |
| `triangle_attention_bwd_pre_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `triangle_attention_fwd_contig_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `triangle_attention_fwd_contig_triton` | NVIDIA RTX A6000 (sm86) | build driver changed -- coverage may differ; rebuild the op |
| `triangle_attention_fwd_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `trimul_bwd_gate_packed_recompute_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `trimul_bwd_gate_packed_recompute_triton` | NVIDIA RTX A6000 (sm86) | build driver changed -- coverage may differ; rebuild the op |
| `trimul_outproj_bwd_gate_recompute_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |
| `trimul_outproj_gemm_gate_triton` | NVIDIA A100 80GB PCIe (sm80) | build driver changed -- coverage may differ; rebuild the op |

## 4. 별도 형식의 캐시

6개 dispatch 캐시는 그리드 캐시가 아니어서 UNKNOWN으로 분류된다. 이것만으로 재생성이 필요하다고 판단하지 않았다.

## 로그

- A6000: `benchmarks/modules/swa_dit/artifacts/cache_rebuild_20260916/`
- A5000: `benchmarks/modules/swa_dit/artifacts/cache_rebuild_a5000_20260916/`
- 전체 기계 판독 스냅샷: `cache-rebuild-inventory-20260916.json`

## 추가 예약

- A5000 후보 설정 변경 3개: `gated_projection_bwd_gate_recompute_flat_triton`, `transition_bwd_epilogue_triton`, `transition_fold_triton`.
- Slurm `1705118`, `afterok:1705092` 의존성으로 순서대로 병합한다.
- 나머지 후보 설정 변경 9개는 A100 대상이며 해당 GPU가 없어 미실행 상태다.
- 재현용 실행 스크립트는 각 재생성 artifact 디렉터리의 `run.sh`에 보관했다.

## GPU 병렬 재배치

- 단일 GPU 작업 `1705443`, `1705092` 및 후속 대기 `1705118`을 취소했다.
- A6000: `1705724`, GPU 8장 / CPU 64개. 기존 shard 디렉터리에서 완료 결과를 재사용하고 중단된 claim을 회수하여 재개한다.
- A5000: `1705725`, GPU 8장 / CPU 64개. 무효 캐시 6개와 후보 설정 변경 3개를 같은 작업 큐에서 처리한다.
- GPU당 unit 1개, 컴파일 worker 8개. GPU 모델별로 별도 병합한다. 임의의 3072/4096 전용 캡처는 포함하지 않는다.

### 병렬 잡 수정

- A5000 `1705725`는 비활성 커널 2개를 지정해 preflight에서 실패했다. 측정은 시작하지 않았다.
- `gated_projection_bwd_gate_recompute_flat_triton`, `transition_bwd_epilogue_triton`은 registry.csv의 `developed=no` 항목이다. 기존 파일의 grid 불일치만으로 현행 빌드 대상이라 판단한 것을 정정한다. 활성 경로에서 필요한지 확인 전 재생성 대상에서 제외한다.
- A5000 최종 작업은 `1705729`: 8 GPU, 무효화된 6개와 활성 증분 대상 `transition_fold_triton`.

## token 길이 범위 정정

- 사용자 지정 token_single 길이 128~768을 적용했다. builder.op_units의 token 길이에 전체 모듈 CASE_LENGTHS를 합치는 동작을 제거했다.
- 생성된 전체 per-op 목록과 HTML에서 token 길이가 128,256,384,512,640,768 이내임을 확인했다.
- 6개 재생성 커널은 576개에서 400개 작업으로 감소했다. A5000은 transition_fold 추가 7개 포함 407개이다.
- 기존 잡 1705724/1705729 중단. 재개 잡: A6000 1706622, A5000 1706623, 각각 8 GPU. 먼저 기존 shard를 병합한 후 남은 후보를 측정한다.
