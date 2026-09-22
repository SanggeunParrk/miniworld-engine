# MiniPairformer 1블록: PyTorch·cuEquivariance 비교

> **후속 해결:** 이 문서는 수정 전 측정 이력이다. 이후 gate 미분 곱셈 순서를 복원하여 L384 엄격한 LN gradient 검증을 통과했다. [수정·검증 결과](../trimul_ln_gradient_20260922/README.md). 원래 실패 기록과 당시 시간은 보존한다.

job15592 · 단위 ms

| 구성 | 추론 | 학습 forward | 학습 fwd+bwd |
|---|---:|---:|---:|
| PyTorch eager | 3.274 | 3.313 | 9.029 |
| PyTorch compile | 1.722 | 1.736 | 5.211 |
| cuEquivariance TriMul + PyTorch Transition (compile) | 0.977 | 0.979 | 3.461 |
| cuEquivariance TriMul + 새 CUDA Transition (compile) | 0.664 | 0.671 | 2.750 |
| 최신 TriMul + 새 CUDA Transition | 0.386 | 0.412 | 1.520 |

- PyTorch eager 대비: 추론 8.48배, 학습 forward 8.05배, 학습 fwd+bwd 5.94배.
- PyTorch compile 대비: 추론 4.46배, 학습 forward 4.22배, 학습 fwd+bwd 3.43배.
- cuEquivariance TriMul + PyTorch Transition (compile) 대비: 추론 2.53배, 학습 forward 2.38배, 학습 fwd+bwd 2.28배.
- cuEquivariance TriMul + 새 CUDA Transition (compile) 대비: 추론 1.72배, 학습 forward 1.63배, 학습 fwd+bwd 1.81배.

## 비교 대상

블록 하나는 양방향 TriMul → Transition이며, 전체를 실제 연결해 측정했다. 파라미터와 입력은 모든 조합에서 동일하다.

- **PyTorch eager:** 엔진 PyTorch 경로의 수식을 plain PyTorch 연산으로 작성. FP32 LayerNorm → BF16 출력, projection/gate, 두 방향 contraction, 공유 256채널 output LN, dropout/residual, SwiGLU Transition/residual이다.
- **PyTorch compile:** 위 전체 블록을 static/fullgraph `torch.compile`한다. 수동 CUDA graph를 사용하므로 Inductor의 자체 cudagraphs는 껐다.
- **cuEquivariance + PyTorch Transition:** cuEq input/output LN과 fused sigmoid-gated dual GEMM을 사용하고 contraction/output projection은 PyTorch로 구성한다. Transition은 PyTorch다. 전체 블록을 compile한다.
- **cuEquivariance + 새 CUDA Transition:** 동일한 cuEq TriMul에 최신 native CUDA Transition을 붙인다. Transition을 공통으로 두어 TriMul 변경 효과를 비교한다.
- **최신 조합:** Anthropic 유래 TriMul forward + cache-policy B1 + K128 단일 B7, 최신 native CUDA Transition forward/backward다.

설치된 cuEquivariance에는 이 블록에 대응하는 Transition API가 없다. cuEq 공개 TMU는 단방향이어서 두 번 호출하면 공유 output LN을 가진 MiniWorld 양방향 TriMul과 수식이 달라진다. 따라서 이 표는 **같은 수식을 가진 cuEq primitives 조합**과의 비교이며, NVIDIA 공개 TMU 모듈 또는 전체 Pairformer 제품의 벤치라고 해석하면 안 된다. cuEq 조합은 이전 비교에서 사용한 `compare_cueq_training.py:cueq_forward`와 동일하며, 추론에서는 dropout 곱셈을 생략한다.

## 조건

node01 H10080GB, B=1/L=384/C=128, BF16 projection과 activation, FP32 LN params, TriMul hidden128 per direction, Transition expansion4.
학습 고정 row dropout25%, 추론 dropout0. pair mask, 두 residual, live weight packing, 입력 및 15개 파라미터 gradient를 포함한다.
모드마다 5×250회 교차 CUDA graph; optimizer, RNG 생성, CPU dispatch, 컴파일은 제외한다. PyTorch eager도 CUDA graph를 쓰므로 eager CPU launch overhead는 포함하지 않는다.
cuEq는 설치된 버전의 init_triton_cache 및 기본 tuning 정책을 사용한다. 버전과 환경값은 results.json에 기록했다.

## 검증과 한계

PyTorch compile 대비 forward 상대L2 0.5%, gradient 1% 한도로 모든 출력과 gradient를 확인한다. 입력 및 squeeze 가중치를 바꾼 graph/eager 비교도 수행한다. LN gradient는 5e-6, 나머지는 bit-exact를 요구한다.
**이전에 남은 최신 TriMul 입력 LN gradient 엄격 검사(최대9.535e-6 >5e-6)는 해결되지 않았다.** 이번 BF16 경로 간 비교가 그 검사를 대체하지 않는다. 최신 학습 시간은 개발 후보의 진단 결과다. production 기본 배선을 바꾸지 않았다.

## 자료

- [원본 결과·시간표본·kernel trace 목록](results.json) · [요약](summary.json)
- [벤치 코드](bench.py) · [Slurm 제출](bench.sbatch)
- [Transition 원본 경로와 SHA-256](transition_sources.json)
- [이전 엔진과 비교](../minipairformer_block_cuda_20260922/index.html)

재현: `sbatch runs/minipairformer_block_baselines_20260922/bench.sbatch`
