# H100 모듈 비교 — 2026-09-23

Engine 2.0.0은 공개 v2.0.0 태그 뒤에 로컬에서 추가한 production 배선을 포함한 wheel이다. 아직 공개 태그나 실행 중인 학습 환경에 이 배선이 반영된 것은 아니다.
Engine 1.0.0은 실제 v1.0.0 태그 `850e160c`이다. 버전 문자열이 1.0.0인 9월 설치본 `1bc0803e` 및 과거 개별 커널 표와 구분한다.

조건: node01 H100 80GB, B=1, BF16 activation/projection, FP32 LN affine, static torch.compile + CUDA graph replay. 각 경우 7회 × 50 replay의 회당 시간 중앙값(ms). 컴파일·첫 튜닝·CPU 실행·optimizer 제외. 학습은 fwd+bwd, 입력 및 전체 parameter gradient, 실제 dropout RNG 포함. mask 적용. 양방향 TriMul 방향별 hidden=128, D=128, dropout=25%; Transition expansion=4. OPM/PWA는 MSA depth=256, MSA64/pair128, PWA head8×32/dropout15%. Token DiT는 sample1/single768/condition384/pair128/head16/expansion1536.

cuEquivariance 양방향 TriMul은 동일한 shared 2H output LN 수식을 유지하는 vendor primitive 조합이며, 단방향 public TMU 두 개의 합이 아니다. MiniPairformer 1블록은 양방향 TriMul + Transition이며 cuEq 열에서는 Transition을 PyTorch로 실행한다. 전용 cuEq 구현 없는 모듈은 `—`. v1.0.0에는 DiTBlock이 없어 해당 셀도 `—`. 오류는 다른 backend 숫자로 대체하지 않는다. 신규 전체 config 튜닝은 수행하지 않았으며 기본 경로의 첫 사용 튜닝 정책을 유지했다.


## 결론과 적용 범위

- 기본 `miniworld` / `auto`에서 7모듈 × 2길이 × 2모드 = 28건의 실행·유한값·커널 trace 검사를 완료했다. [검사 결과](dispatch-audit.json).
- 양방향 TriMul, Transition, OPM, PWA는 새 학습·추론 경로가 실행된다. 단방향 TriMul과 Token DiT는 새 추론 경로가 실행되며 학습은 기존 경로이다.
- 양방향 TriMul 학습은 v1.0.0 태그 대비 L384 1.75배, L768 1.72배 빠르다. MiniPairformer는 cuEq TriMul + PyTorch Transition 대비 학습 2.13배/2.10배 빠르다.
- Token DiT sample1에서는 새 추론 경로가 PyTorch보다 느리다. 전체 shape의 최적 backend 선택이나 튜닝이 끝났다는 결론은 아니다. 단방향 TriMul 및 DiT 학습에서도 실제 cache miss가 관찰됐다.
- Transition의 native 빌드 확인이 Dynamo 안으로 들어가는 문제를 수정했다. narrow/wide CPU 회귀 검사 4건과 수정 후 GPU forward/backward 실행을 확인했다.
- v1.0.0 Transition/블록은 이 PyTorch 2.10 환경에서 CUDA assertion 또는 custom-op alias 제약으로 실패했다. 단방향 학습은 원래 parameter 일부의 gradient가 연결되지 않아 실패했다. 다른 revision/커널의 값으로 채우지 않았다.
- 아래는 pair D128 중심의 제한된 shape 비교다. D64/256/384/512의 추가 최적화 및 전체 cache 재튜닝 완료를 뜻하지 않는다. OPM은 pair residual을 포함한다.


## 추론 forward

| 모듈 | L | PyTorch | cuEquivariance | Engine 1.0.0 | Engine 2.0.0 |
|---|---:|---:|---:|---:|---:|
| 양방향 TriMul | 384 | 1.303 | 0.553 | 0.685 | 0.255 |
| 양방향 TriMul | 768 | 11.663 | 2.111 | 2.628 | 1.040 |
| Transition | 384 | 0.434 | — | 오류 | 0.127 |
| Transition | 768 | 1.651 | — | 오류 | 0.467 |
| MiniPairformer 1블록 | 384 | 1.712 | 0.983 | 오류 | 0.377 |
| MiniPairformer 1블록 | 768 | 13.314 | 3.739 | 오류 | 1.510 |
| 단방향 TriMul (outgoing) | 384 | 0.763 | 0.305 | 0.377 | 0.151 |
| 단방향 TriMul (outgoing) | 768 | 6.203 | 1.188 | 1.462 | 0.587 |
| OPM | 384 | 0.545 | — | 0.544 | 0.362 |
| OPM | 768 | 2.125 | — | 2.135 | 1.283 |
| PWA | 384 | 0.323 | — | 0.320 | 0.125 |
| PWA | 768 | 0.729 | — | 0.734 | 0.376 |
| Token DiT | 384 | 0.150 | — | — | 0.268 |
| Token DiT | 768 | 0.303 | — | — | 0.363 |

## 학습 forward + backward

| 모듈 | L | PyTorch | cuEquivariance | Engine 1.0.0 | Engine 2.0.0 |
|---|---:|---:|---:|---:|---:|
| 양방향 TriMul | 384 | 4.018 | 2.261 | 1.857 | 1.060 |
| 양방향 TriMul | 768 | 35.351 | 8.568 | 7.206 | 4.190 |
| Transition | 384 | 1.191 | — | 오류 | 0.562 |
| Transition | 768 | 4.498 | — | 오류 | 2.140 |
| MiniPairformer 1블록 | 384 | 5.274 | 3.515 | 오류 | 1.648 |
| MiniPairformer 1블록 | 768 | 40.077 | 13.333 | 오류 | 6.360 |
| 단방향 TriMul (outgoing) | 384 | 2.415 | 1.397 | 오류 | 1.097 |
| 단방향 TriMul (outgoing) | 768 | 18.936 | 5.386 | 오류 | 4.145 |
| OPM | 384 | 1.355 | — | 1.392 | 0.997 |
| OPM | 768 | 5.058 | — | 5.160 | 3.495 |
| PWA | 384 | 0.877 | — | 0.975 | 0.562 |
| PWA | 768 | 1.944 | — | 2.303 | 1.322 |
| Token DiT | 384 | 0.449 | — | — | 0.466 |
| Token DiT | 768 | 0.891 | — | — | 0.940 |

## 실제 엔진 2 경로 (profiler kernel 이름)

- 양방향 TriMul inference: triton_poi_fused__to_copy_bitwise_and_trimul_h100_infer_unsqueeze_0; tmn_k1_z128_h256_b_t3x64_s8k2_m1_l2_v0; tmn_k3_z128_h256_b_t2x64_s8a1_l1
- 양방향 TriMul training: triton_poi_fused__to_copy_div_gt_rand_0; triton_poi_fused__to_copy_bitwise_and_unsqueeze_1; triton_poi_fused_cat_copy__stack_view_0; infer_k1; save_k3; b1_fused; b7_joint
- Transition inference: triton_poi_fused_clone_t_transition_fused_fwd_sm90a_view_0; transition_fwd_fused
- Transition training: triton_poi_fused_clone_t_transition_fused_fwd_sm90a_view_0; transition_fwd_fused; transition_bwd_fused
- MiniPairformer 1블록 inference: triton_poi_fused__to_copy_bitwise_and_trimul_h100_infer_unsqueeze_0; tmn_k1_z128_h256_b_t3x64_s8k2_m1_l2_v0; tmn_k3_z128_h256_b_t2x64_s8a1_l1; triton_poi_fused_clone_t_transition_fused_fwd_sm90a_view_1; transition_fwd_fused
- MiniPairformer 1블록 training: triton_poi_fused__to_copy_div_gt_rand_0; triton_poi_fused__to_copy_bitwise_and_unsqueeze_1; triton_poi_fused_cat_copy__stack_view_0; infer_k1; save_k3; triton_poi_fused_clone_t_transition_fused_fwd_sm90a_view_2; transition_fwd_fused; transition_bwd_fused; b1_fused; b7_joint
- 단방향 TriMul (outgoing) inference: triton_poi_fused__to_copy_bitwise_and_trimul_h100_infer_unsqueeze_0; tmn_k1_z128_h128_b_t6x32_s8k2_m1_l2_v0; tmn_k3_z128_h128_b_t3x64_s8a1_l1
- 단방향 TriMul (outgoing) training: triton_poi_fused__to_copy_div_gt_rand_0; triton_poi_fused_cat_copy_slice_t_1; triton_poi_fused_bitwise_and_trimul_inproj_masked_sm90_cute_unsqueeze_view_2; triton_poi_fused__to_copy_bitwise_and_trimul_front_bwd_dconcat_unsqueeze_view_0; triton_poi_fused_cat_t_1; triton_poi_fused_add_view_2; triton_poi_fused_clone_slice_t_3; triton_poi_fused_clone_slice_t_4; triton_poi_fused_clone_slice_t_5; triton_poi_fused_clone_slice_t_6
- OPM inference: _opm_prologue_kernel; (anonymous namespace)::opm_epilogue_kernel(__nv_bfloat16 const*, float const*, __nv_bfloat16 const*, float const*, int, int, long, CUtensorMap_st); triton_poi_fused_add_0
- OPM training: _opm_prologue_kernel; (anonymous namespace)::opm_epilogue_kernel(__nv_bfloat16 const*, float const*, __nv_bfloat16 const*, float const*, int, int, long, CUtensorMap_st); triton_poi_fused_add_0; (anonymous namespace)::opm_dgrad_kernel(__nv_bfloat16 const*, float const*, __nv_bfloat16 const*, __nv_bfloat16*, float*, int, int, long, CUtensorMap_st); (anonymous namespace)::opm_reduce_partials_chunked(float const*, float*, int, int, int, long); (anonymous namespace)::opm_dwo_kernel(__nv_bfloat16 const*, __nv_bfloat16 const*, float*, int, int, long, int); (anonymous namespace)::opm_prologue_bwd_kernel(__nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16 const*, float const*, __nv_bfloat16 const*, float const*, float const*, float, __nv_bfloat16 const*, __nv_bfloat16*, float*, float*, int, int)
- PWA inference: void (anonymous namespace)::pwa_fwd2_kernel<3, 1>(int, int, int, int, CUtensorMap_st, CUtensorMap_st, __nv_bfloat16 const*, __nv_bfloat16 const*, float, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)
- PWA training: triton_poi_fused__to_copy_clone_expand_select_unsqueeze_0; void (anonymous namespace)::pwa_fwd2_kernel<3, 1>(int, int, int, int, CUtensorMap_st, CUtensorMap_st, __nv_bfloat16 const*, __nv_bfloat16 const*, float, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st); void (anonymous namespace)::pwa_glue3_kernel<2>(int, int, int, float*, CUtensorMap_st, CUtensorMap_st, __nv_bfloat16 const*, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st); void (anonymous namespace)::pwa_fwd2_kernel<4, 2>(int, int, int, int, __nv_bfloat16 const*, float, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, CUtensorMap_st)
- Token DiT inference: 원본 JSON 참조
- Token DiT training: triton_poi_fused_adaln_fwd_gate_clone_t_view_1; triton_poi_fused_div_2; triton_poi_fused_layernorm_fwd_view_zeros_3; triton_poi_fused__unsafe_view_clone_permute_4; triton_poi_fused__unsafe_view_mul_sigmoid_view_5; triton_poi_fused__unsafe_view_add_mul_sigmoid_view_6; triton_poi_fused_layernorm_linear_materialize_new_zeros_view_0; triton_poi_fused_add_view_7; triton_red_fused_sum_0; triton_per_fused_sum_1; triton_poi_fused_cat_2; triton_poi_fused_cat_3

## 실패 기록

- ('block', 768, 'engine1') inference: `torch._dynamo.exc.InternalTorchDynamoError: AcceleratorError: CUDA error: device-side assert triggered`
- ('block', 768, 'engine1') training: `torch.AcceleratorError: CUDA error: device-side assert triggered`
- ('block', 384, 'engine1') inference: `torch._dynamo.exc.InternalTorchDynamoError: AcceleratorError: CUDA error: device-side assert triggered`
- ('block', 384, 'engine1') training: `torch.AcceleratorError: CUDA error: device-side assert triggered`
- ('transition', 384, 'engine1') inference: `torch._dynamo.exc.InternalTorchDynamoError: AcceleratorError: CUDA error: device-side assert triggered`
- ('transition', 384, 'engine1') training: `torch.AcceleratorError: CUDA error: device-side assert triggered`
- ('transition', 768, 'engine1') inference: `torch._dynamo.exc.InternalTorchDynamoError: AcceleratorError: CUDA error: device-side assert triggered`
- ('transition', 768, 'engine1') training: `torch.AcceleratorError: CUDA error: device-side assert triggered`
- ('single', 384, 'engine1') training: `RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.`
- ('single', 768, 'engine1') training: `RuntimeError: One of the differentiated Tensors appears to not have been used in the graph. Set allow_unused=True if this is the desired behavior.`
