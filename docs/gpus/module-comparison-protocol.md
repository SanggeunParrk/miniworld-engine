# 모듈별 추론과 학습 성능 비교 기준

2026-10-05 사용자 합의 기준. B200에서 실제 측정한 모듈별 길이와 샘플 수를 아래 표로 정리했으며, 이후 A100을 포함한 모듈 비교도 이 shape를 따른다. 추론은 Anthropic, 학습은 해당 모듈의 cuEquivariance 구현을 기준으로 비교한다. 대응하는 cuEquivariance 구현이 없으면 PyTorch를 사용한다.

이 문서는 비교 조건을 정한다. 기존 B200 수치는 각 원문에 남겨두며, 새로운 GPU에서의 성능은 해당 GPU에서 다시 측정한다.

이 기준으로 실행한 결과: [A100 모듈별 추론과 학습 비교 2026-10-05](../records/a100-module-comparison-20261005.md).

## 추론과 학습 shape

`L`은 토큰 수, `N`은 실제 원자 수, `A`는 augmentation 또는 diffusion sample 수, `B`는 구조 batch 수다. `128–768, 간격 128`은 `{128, 256, 384, 512, 640, 768}`을 뜻한다. 아래 기본 구조 batch는 `B=1`이다.

| 모듈과 B200 근거 | 추론 길이와 샘플 수 | 학습 길이와 샘플 수 |
|---|---|---|
| [Token DiT](b200/token_dit/token_dit.md) | `L=128–768`, 간격 128; 추가 `L=200,450,700`; `A=5` | `L=384,768`; `A=48` |
| [Bias-only Token DiT](b200/bias_only_dit/bias_only_dit.md) | `L=128–768`, 간격 128; `A=5` | `L=384,768`; `A=48` |
| [Dense Atom DiT](b200/atom_dit/atom_dit.md) | `N=1024,2048,4096` × `A=1,5` | `N=4096,8192`; `A=48` |
| [AF3 local 32×128 Atom DiT](b200/local_dit/local_dit.md) | block 기준 `(A,N)=(5,2048),(5,4096),(1,4096),(48,4096)` | `N=4096,8192`; `A=48` |
| [SWA Atom DiT](b200/swa_atom_dit/swa_atom_dit.md) | `N=1024,2048,4096` × `A=1,5` | `N=4096,8192`; `A=48` |
| [TriMul 단방향 및 양방향](b200/trimul/trimul.md) | `L=128–768`, 간격 128; `B=1` | 동일 길이; `B=1` |
| [TriangleAttention](b200/triattn/triattn.md) | `L=128–768`, 간격 128; `B=1` | 동일 길이; `B=1` |
| [AttentionPairBias](b200/attention_pair_bias/attention_pair_bias.md) | `L=128–768`, 간격 128; `B=1` | 동일 길이; `B=1` |
| [OPM](b200/opm/opm.md) / [PWA](b200/pwa/pwa.md) | `L=128–768`, 간격 128 × MSA depth `1024,2048,4096` | 동일 길이; MSA depth `1024` |
| [Pair Transition](b200/transition/transition.md) | 대표 길이 `L=384,768` | 대표 길이 `L=384,768` |

모든 모듈을 `L=128, A=2`로 통일하지 않는다. shape registry의 지원 범위와 실제 측정 조건은 구분한다. 위 표에 없는 독립 모듈의 전체 sweep은 아직 이 합의로 정해지지 않았다. LayerNorm, AdaLN, ConditionedTransition 등 하위 연산을 분해 측정할 때는 대응하는 상위 block의 입력 shape를 사용하고, 전체 block 결과와 구분해 기록한다.

### Atom DiT의 길이와 변형

Atom DiT의 `N`을 토큰 길이로 해석하면 안 된다. 공통 `bench.py`의 atom target은 `seq_len`을 원자 수로 변환할 때 8을 곱한다. 따라서 `N=4096,8192`는 그 경로에서 `seq_len=512,1024`에 해당한다. 별도 비교 스크립트의 `--length`가 실제 원자 수를 받는 경우에는 `4096,8192`를 직접 지정한다. 실행 전 생성된 tensor shape를 확인하고 결과에 실제 `N`을 기록한다.

AF3 local은 query 32개 × key 128개 window를 유지한다. `W=ceil(N/32)`일 때 pair 입력은 `[B,W,32,128,16]`이다. Dense Atom DiT와 SWA Atom DiT는 각각 별도 모듈로 비교한다.

B200 local 문서의 block 측정은 shared-AdaLN 변형이다. 현재 AF3like 비교 대상인 `cross_attention=True`는 별도 KV AdaLN이 있으므로 변형을 결과에 명시하고 같은 변형끼리 비교한다. 기존 shared-AdaLN 시간을 cross-attention 성능으로 옮겨 적지 않는다. 또한 attention core만 측정한 `(A,N)` 목록을 전체 block의 측정 목록과 혼합하지 않는다.

## Compile과 CUDA graph

추론 표에는 PyTorch compiled 비교를 포함하고, 각 backend의 실제 compile 여부와 CUDA graph 여부를 별도로 기록한다. 일반 모듈의 B200 측정은 compiled 경로에서 추론 CUDA graph, 학습 CUDA graph OFF/ON을 나누어 측정했다. 단, 다음 이력과 예외를 유지한다.

| 대상 | B200에서 실제 측정한 실행 방식 | 이후 표기 규칙 |
|---|---|---|
| Token DiT 및 AttentionPairBias의 Anthropic 경로 | `compile=false` + CUDA graph; upstream 호출에 실행 가능한 compiled graph가 없음 | Anthropic을 compiled로 표기하지 않는다 |
| AF3 local Atom DiT의 PyTorch baseline | eager 연산을 CUDA graph로 replay | 과거 수치는 eager + graph로 유지하고, 새 compiled 측정은 별도 행으로 추가한다 |
| Dense 및 SWA Atom DiT의 capsule 측정 | 각 원문에 기재된 block 및 graph 측정 | compile 지원 테스트 통과를 compiled 성능 측정으로 간주하지 않는다 |

학습은 forward + backward를 비교하며, CUDA graph OFF/ON 결과를 분리한다. 특정 모듈이나 backend에서 한 방식만 측정했으면 나머지는 미측정으로 둔다. compile 실패 또는 지원 불가 시 원인을 기록하고, eager fallback을 compiled 결과로 표시하지 않는다.

## Baseline과 결과 기록

- 추론의 주 비교 대상은 Anthropic이며 PyTorch compiled도 함께 기록한다. Anthropic 대응 구현이 없거나 실행할 수 없으면 해당 칸에 사유를 표시한다.
- 학습은 동등한 모듈의 cuEquivariance 경로가 있으면 사용하고, 없으면 PyTorch compiled를 사용한다. AttentionPairBias에는 공통 benchmark의 cuEquivariance whole-module 경로가 있으므로 누락하지 않는다. Pairformer처럼 cuEquivariance와 PyTorch를 조합한 경로는 hybrid로 명시한다.
- 동일 GPU에서 같은 shape, 연산 범위, dtype, parameter dtype, TF32, mask, dropout, QK-norm, conditioning 공유 여부를 맞춘다. 세부 폭과 head 구성은 연결된 B200 원문의 측정 조건을 따른다. BF16 mixed와 BF16 parameter, FP32와 TF32 설정을 구분한다.
- conditioning 또는 pair bias를 사전 계산하는 경우, block 내부 계산과 사전 계산 결과를 구분하고 양쪽 timing에 포함한 작업을 명시한다.
- 실제 로드한 Anthropic 소스 revision과 실행 경로를 기록한다. upstream 구현 대신 다른 backend로 fallback한 결과를 Anthropic 수치로 표시하지 않는다.
- 시간은 ms, 속도비는 `baseline 시간 / ours 시간`으로 기록한다. 추론 Anthropic 대비, 학습 cuEquivariance 또는 PyTorch 대비를 구분하고, 기존 ours 대비 개선율은 별도 열로 둔다.
- OOM, 미지원, compile 실패, 미측정은 그대로 남긴다. 길이 또는 `A`를 낮춘 결과로 원래 조건의 칸을 채우지 않는다.

기존 A100 [2026-10-05 CSV](a100/module-inference-training-20261005.csv)와 [JSON](a100/module-inference-training-20261005.json)은 작은 shape의 탐색 측정이다. 이 문서의 조건을 충족하는 정식 비교 결과로 사용하지 않는다.
