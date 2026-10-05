# A100 모듈별 추론과 학습 비교

측정일 2026-10-05. [합의한 비교 기준](../gpus/module-comparison-protocol.md)의 길이와 샘플 수로 실행한 현재 코드의 결과다. 완료 task 118/118.

NVIDIA A100 80GB PCIe, 전력 한도 300 W. 공통 `benchmarks/runners/bench.py`를 Slurm의 전용 GPU allocation에서 실행했다. 각 task의 backend는 같은 GPU에서 순서대로 측정했다. 서로 다른 shape와 graph 조건은 다른 GPU에서 실행될 수 있으며 node·job·실제 tensor shape·소스 hash는 raw 결과에 기록했다.

BF16 activation, `precision=bf16-mixed`, TF32 허용. 실제 parameter dtype은 CSV에 기록했다. B=1, Token/Bias-only DiT 추론 A5 및 shared conditioning, 학습 A48 및 per-sample conditioning, mask 20%; 나머지는 mask 없음. TriMul/TriangleAttention 학습 dropout 0.25, PWA 0.15. Atom DiT는 conditioning을 block 내부에서 계산하며 local은 `cross_attention=True`다.

Token DiT는 16×48 heads, QK-norm OFF; atom은 4×32 heads, single/cond 폭 128이다. Pair 계열은 폭 64/128/256/384/512, APB는 8×48, 16×24, 12×32, 24×16, 16×32 구성을 측정했다. Pair Transition은 n=4다. OPM/PWA의 MSA 폭은 64, pair 폭은 128이다. FP32 및 그 밖의 head 변형은 이번 결과에 포함하지 않았다.

시간은 warmup 이후 median ms다. Token/Bias-only DiT 추론은 고정된 pair를 반복 사용한다. 지원되는 fused 경로에서는 Token DiT의 pair-bias cache와 Bias-only DiT의 softmax cache를 사용하며, PyTorch baseline은 해당 연산을 다시 계산한다. 이는 매 호출마다 pair가 바뀌는 workload의 속도비가 아니다.

추론은 CUDA graph ON, 학습은 OFF/ON 별도 측정이다. PyTorch와 MiniWorld에는 compile을 요청하고 실제 실행 증거를 검증했다. Anthropic Token DiT/APB는 graph-only이며, 다른 Anthropic compile 실패에 대한 graph-only 재측정도 구분했다. 아래 †는 compile=false + CUDA graph를 뜻한다. compile 실패 기록은 CSV/JSON에 보존했다.

Anthropic upstream revision: `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`. 공통 harness의 named upstream 경로를 사용했다. Token DiT는 `fpf_apb` composition, APB는 `apb_attn` composition, TriangleAttention은 strict `block:triattn_native`다. 모든 upstream variant 중 최솟값을 탐색한 결과는 아니다. Ours 측정 시 Anthropic override를 비활성화했다.

## 모듈별 요약

속도비는 baseline ms / ours ms로 1보다 크면 ours가 빠르다. 각 칸은 **기하평균 (최소–최대; 유효 비교 수)**이며 해당 모듈의 측정 폭과 길이를 포함한다. 실패한 shape는 평균에서 제외하되 아래 상세 표에 남겼다. 서로 다른 조건의 기하평균만으로 모든 shape가 더 빠르다고 판단하지 않는다.

| 모듈 | 추론 Anthropic 대비 | 추론 PyTorch compiled 대비 | 학습 baseline | 학습 graph OFF 대비 | 학습 graph ON 대비 |
|---|---|---|---|---|---|
| Token DiT | 0.96× (0.67–1.40; 9개) | 1.20× (0.72–1.96; 9개) | pytorch | 1.63× (1.44–1.83; 2개) | 1.60× (1.43–1.80; 2개) |
| Bias-only Token DiT | 1.18× (0.85–1.65; 6개) | 0.88× (0.70–1.00; 6개) | pytorch | 1.12× (1.11–1.12; 2개) | 1.09× (1.09–1.09; 2개) |
| Dense Atom DiT | 3.45× (2.19–5.25; 6개) | 3.44× (2.02–5.12; 6개) | pytorch | 4.35× (4.35–4.35; 1개) | 4.34× (4.34–4.34; 1개) |
| AF3 local 32×128 Atom DiT | 1.15× (1.07–1.24; 4개) | 1.12× (1.04–1.21; 4개) | pytorch | 1.08× (1.07–1.09; 2개) | 1.08× (1.07–1.09; 2개) |
| SWA Atom DiT | 8.26× (5.52–12.17; 6개) | 3.69× (1.82–8.09; 6개) | pytorch | 10.88× (7.60–15.59; 2개) | 10.84× (7.51–15.66; 2개) |
| TriMul 단방향 | 1.09× (0.83–1.60; 24개) | 2.73× (0.80–7.13; 30개) | cuequivariance | 1.37× (0.53–2.13; 30개) | 1.40× (0.53–2.15; 30개) |
| TriMul 양방향 | 1.16× (0.93–1.52; 12개) | 2.85× (0.66–7.00; 30개) | cuequivariance | 1.19× (0.38–1.74; 30개) | 1.20× (0.38–1.76; 30개) |
| TriangleAttention | 1.11× (0.95–1.56; 12개) | 4.02× (1.96–6.48; 30개) | cuequivariance | 1.72× (0.90–2.27; 30개) | 1.87× (1.31–2.28; 30개) |
| AttentionPairBias | 1.03× (0.64–1.27; 30개) | 1.84× (1.08–2.59; 30개) | cuequivariance | 0.73× (0.43–0.93; 30개) | 1.64× (1.10–2.14; 30개) |
| OPM | 2.23× (1.67–2.95; 18개) | 1.24× (1.13–1.44; 18개) | pytorch | 1.14× (0.94–1.22; 6개) | 1.20× (1.17–1.24; 6개) |
| PWA | 1.36× (1.22–1.50; 18개) | 1.89× (1.59–2.13; 18개) | pytorch | 1.40× (1.26–1.46; 6개) | 1.51× (1.47–1.57; 6개) |
| Pair Transition | 1.79× (1.20–3.36; 6개) | 2.12× (1.23–5.07; 10개) | pytorch | 1.26× (0.95–2.43; 10개) | 1.33× (0.95–2.46; 10개) |

## Token DiT

| 변형 | 모드 | graph | L 또는 N | A | ours ms | baseline | baseline ms | baseline / ours | PyTorch compiled ms |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| main | inference | manual | 128 | 5 | 0.2888 | anthropic | 0.1925 † | 0.67× | 0.2089 |
| main | inference | manual | 256 | 5 | 0.4137 | anthropic | 0.3328 † | 0.80× | 0.3686 |
| main | inference | manual | 384 | 5 | 0.5059 | anthropic | 0.4424 † | 0.87× | 0.5294 |
| main | inference | manual | 512 | 5 | 0.6595 | anthropic | 0.5796 † | 0.88× | 0.8274 |
| main | inference | manual | 640 | 5 | 0.7997 | anthropic | 0.7496 † | 0.94× | 1.1100 |
| main | inference | manual | 768 | 5 | 0.9841 | anthropic | 0.9667 † | 0.98× | 1.4653 |
| main | training | disabled | 384 | 48 | 7.4967 | pytorch | 10.8278 | 1.44× | — |
| main | training | disabled | 768 | 48 | 16.6666 | pytorch | 30.4886 | 1.83× | — |
| main | training | manual | 384 | 48 | 7.4793 | pytorch | 10.6865 | 1.43× | — |
| main | training | manual | 768 | 48 | 16.5289 | pytorch | 29.7503 | 1.80× | — |
| pad200 | inference | manual | 200 | 5 | 0.2775 | anthropic | 0.2806 † | 1.01× | 0.2775 |
| pad450 | inference | manual | 450 | 5 | 0.4424 | anthropic | 0.5540 † | 1.25× | 0.6820 |
| pad700 | inference | manual | 700 | 5 | 0.6769 | anthropic | 0.9472 † | 1.40× | 1.3281 |

## Bias-only Token DiT

| 변형 | 모드 | graph | L 또는 N | A | ours ms | baseline | baseline ms | baseline / ours | PyTorch compiled ms |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| main | inference | manual | 128 | 5 | 0.2488 | anthropic | 0.2120 | 0.85× | 0.1731 |
| main | inference | manual | 256 | 5 | 0.3410 | anthropic | 0.3512 | 1.03× | 0.2836 |
| main | inference | manual | 384 | 5 | 0.4342 | anthropic | 0.5007 | 1.15× | 0.3707 |
| main | inference | manual | 512 | 5 | 0.5949 | anthropic | 0.7107 | 1.19× | 0.5652 |
| main | inference | manual | 640 | 5 | 0.7137 | anthropic | 0.9871 | 1.38× | 0.6892 |
| main | inference | manual | 768 | 5 | 0.8448 | anthropic | 1.3906 | 1.65× | 0.8407 |
| main | training | disabled | 384 | 48 | 5.3709 | pytorch | 6.0406 | 1.12× | — |
| main | training | disabled | 768 | 48 | 11.2015 | pytorch | 12.4006 | 1.11× | — |
| main | training | manual | 384 | 48 | 5.5081 | pytorch | 5.9965 | 1.09× | — |
| main | training | manual | 768 | 48 | 11.4432 | pytorch | 12.4447 | 1.09× | — |

## Dense Atom DiT

| 변형 | 모드 | graph | L 또는 N | A | ours ms | baseline | baseline ms | baseline / ours | PyTorch compiled ms |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| a1n1024_2048 | inference | manual | 1024 | 1 | 0.1464 | anthropic | 0.3389 | 2.31× | 0.2959 |
| a1n1024_2048 | inference | manual | 2048 | 1 | 0.2652 | anthropic | 1.1192 | 4.22× | 0.8796 |
| a1n4096 | inference | manual | 4096 | 1 | 0.7485 | anthropic | 3.9327 | 5.25× | 3.1089 |
| a48n4096 | training | disabled | 4096 | 48 | 25.9891 | pytorch | 113.0045 | 4.35× | — |
| a48n4096 | training | manual | 4096 | 48 | 26.2533 | pytorch | 114.0521 | 4.34× | — |
| a48n8192 | training | disabled | 8192 | 48 | 96.9994 | pytorch | OOM | — | — |
| a48n8192 | training | manual | 8192 | 48 | 96.7209 | pytorch | OOM | — | — |
| a5n1024_2048 | inference | manual | 1024 | 5 | 0.1976 | anthropic | 0.4332 | 2.19× | 0.5079 |
| a5n1024_2048 | inference | manual | 2048 | 5 | 0.3840 | anthropic | 1.3804 | 3.59× | 1.7316 |
| a5n4096 | inference | manual | 4096 | 5 | 1.1653 | anthropic | 4.8609 | 4.17× | 5.9668 |

## AF3 local 32×128 Atom DiT

| 변형 | 모드 | graph | L 또는 N | A | ours ms | baseline | baseline ms | baseline / ours | PyTorch compiled ms |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| a1n4096 | inference | manual | 4096 | 1 | 0.2560 | anthropic | 0.3164 | 1.24× | 0.2652 |
| a48n4096 | inference | manual | 4096 | 48 | 4.2875 | anthropic | 4.8655 | 1.13× | 5.1948 |
| a48n4096 | training | disabled | 4096 | 48 | 16.8356 | pytorch | 18.3286 | 1.09× | — |
| a48n4096 | training | manual | 4096 | 48 | 16.6820 | pytorch | 18.1791 | 1.09× | — |
| a48n8192 | training | disabled | 8192 | 48 | 33.1008 | pytorch | 35.4908 | 1.07× | — |
| a48n8192 | training | manual | 8192 | 48 | 32.8100 | pytorch | 35.2087 | 1.07× | — |
| a5n2048 | inference | manual | 2048 | 5 | 0.3338 | anthropic | 0.3942 | 1.18× | 0.3779 |
| a5n4096 | inference | manual | 4096 | 5 | 0.6093 | anthropic | 0.6533 | 1.07× | 0.6799 |

## SWA Atom DiT

| 변형 | 모드 | graph | L 또는 N | A | ours ms | baseline | baseline ms | baseline / ours | PyTorch compiled ms |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| a1n1024_2048 | inference | manual | 1024 | 1 | 0.0727 | anthropic | 0.4014 † | 5.52× | 0.1321 |
| a1n1024_2048 | inference | manual | 2048 | 1 | 0.0829 | anthropic | 0.5202 † | 6.27× | 0.2499 |
| a1n4096 | inference | manual | 4096 | 1 | 0.0973 | anthropic | 0.7598 † | 7.81× | 0.5325 |
| a48n4096 | training | disabled | 4096 | 48 | 7.0052 | pytorch | 53.2214 | 7.60× | — |
| a48n4096 | training | manual | 4096 | 48 | 7.0830 | pytorch | 53.1978 | 7.51× | — |
| a48n8192 | training | disabled | 8192 | 48 | 13.0130 | pytorch | 202.9240 | 15.59× | — |
| a48n8192 | training | manual | 8192 | 48 | 13.0673 | pytorch | 204.6290 | 15.66× | — |
| a5n1024_2048 | inference | manual | 1024 | 5 | 0.1004 | anthropic | 0.8663 † | 8.63× | 0.2191 |
| a5n1024_2048 | inference | manual | 2048 | 5 | 0.1352 | anthropic | 1.5124 † | 11.19× | 0.6400 |
| a5n4096 | inference | manual | 4096 | 5 | 0.2355 | anthropic | 2.8662 † | 12.17× | 1.9057 |

## TriMul 단방향

| 변형 | 모드 | graph | L 또는 N | A | ours ms | baseline | baseline ms | baseline / ours | PyTorch compiled ms |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| d128 | inference | manual | 128 | — | 0.0655 | anthropic | 0.0748 | 1.14× | 0.1710 |
| d128 | inference | manual | 256 | — | 0.1731 | anthropic | 0.2017 | 1.17× | 0.6011 |
| d128 | inference | manual | 384 | — | 0.3492 | anthropic | 0.4065 | 1.16× | 1.3722 |
| d128 | inference | manual | 512 | — | 0.6369 | anthropic | 0.7066 | 1.11× | 3.3316 |
| d128 | inference | manual | 640 | — | 1.0701 | anthropic | 1.1756 | 1.10× | 7.5438 |
| d128 | inference | manual | 768 | — | 1.5964 | anthropic | 1.7275 | 1.08× | 11.3828 |
| d128 | training | disabled | 128 | — | 1.6251 | cuequivariance | 2.4238 | 1.49× | — |
| d128 | training | disabled | 256 | — | 1.7019 | cuequivariance | 2.5334 | 1.49× | — |
| d128 | training | disabled | 384 | — | 1.7720 | cuequivariance | 2.6296 | 1.48× | — |
| d128 | training | disabled | 512 | — | 2.6675 | cuequivariance | 4.5445 | 1.70× | — |
| d128 | training | disabled | 640 | — | 4.3448 | cuequivariance | 7.2059 | 1.66× | — |
| d128 | training | disabled | 768 | — | 6.4102 | cuequivariance | 10.4182 | 1.63× | — |
| d128 | training | manual | 128 | — | 0.2918 | cuequivariance | 0.4188 | 1.44× | — |
| d128 | training | manual | 256 | — | 0.7444 | cuequivariance | 1.1919 | 1.60× | — |
| d128 | training | manual | 384 | — | 1.4909 | cuequivariance | 2.5948 | 1.74× | — |
| d128 | training | manual | 512 | — | 2.6911 | cuequivariance | 4.5476 | 1.69× | — |
| d128 | training | manual | 640 | — | 4.3940 | cuequivariance | 7.2550 | 1.65× | — |
| d128 | training | manual | 768 | — | 6.5065 | cuequivariance | 10.5380 | 1.62× | — |
| d256 | inference | manual | 128 | — | 0.1444 | anthropic | 0.1761 | 1.22× | 0.3215 |
| d256 | inference | manual | 256 | — | 0.5161 | anthropic | 0.5335 | 1.03× | 1.1950 |
| d256 | inference | manual | 384 | — | 1.1192 | anthropic | 1.1715 | 1.05× | 2.7638 |
| d256 | inference | manual | 512 | — | 2.0306 | anthropic | 2.0787 | 1.02× | 7.0011 |
| d256 | inference | manual | 640 | — | 3.3004 | anthropic | 3.3546 | 1.02× | 14.9243 |
| d256 | inference | manual | 768 | — | 4.8845 | anthropic | 4.9336 | 1.01× | 22.8111 |
| d256 | training | disabled | 128 | — | 1.3921 | cuequivariance | 1.1264 | 0.81× | — |
| d256 | training | disabled | 256 | — | 2.0598 | cuequivariance | 2.9440 | 1.43× | — |
| d256 | training | disabled | 384 | — | 4.4774 | cuequivariance | 6.3959 | 1.43× | — |
| d256 | training | disabled | 512 | — | 7.8848 | cuequivariance | 12.5932 | 1.60× | — |
| d256 | training | disabled | 640 | — | 12.6172 | cuequivariance | 22.9652 | 1.82× | — |
| d256 | training | disabled | 768 | — | 18.5508 | cuequivariance | 33.9871 | 1.83× | — |
| d256 | training | manual | 128 | — | 0.6287 | cuequivariance | 0.8806 | 1.40× | — |
| d256 | training | manual | 256 | — | 2.0367 | cuequivariance | 2.8887 | 1.42× | — |
| d256 | training | manual | 384 | — | 4.4380 | cuequivariance | 6.2444 | 1.41× | — |
| d256 | training | manual | 512 | — | 7.8541 | cuequivariance | 12.5747 | 1.60× | — |
| d256 | training | manual | 640 | — | 12.7263 | cuequivariance | 22.7118 | 1.78× | — |
| d256 | training | manual | 768 | — | 18.7832 | cuequivariance | 34.0306 | 1.81× | — |
| d384 | inference | manual | 128 | — | 0.2386 | anthropic | 0.3809 | 1.60× | 0.4700 |
| d384 | inference | manual | 256 | — | 0.9165 | anthropic | 1.2564 | 1.37× | 1.9896 |
| d384 | inference | manual | 384 | — | 2.0285 | anthropic | 2.7392 | 1.35× | 5.1128 |
| d384 | inference | manual | 512 | — | 3.7094 | anthropic | 4.8742 | 1.31× | 11.6884 |
| d384 | inference | manual | 640 | — | 6.0559 | anthropic | 7.7640 | 1.28× | 26.7581 |
| d384 | inference | manual | 768 | — | 8.8996 | anthropic | 11.3521 | 1.28× | 40.5883 |
| d384 | training | disabled | 128 | — | 1.4582 | cuequivariance | 2.2446 | 1.54× | — |
| d384 | training | disabled | 256 | — | 3.6915 | cuequivariance | 7.6723 | 2.08× | — |
| d384 | training | disabled | 384 | — | 7.9872 | cuequivariance | 17.0486 | 2.13× | — |
| d384 | training | disabled | 512 | — | 14.4394 | cuequivariance | 30.4486 | 2.11× | — |
| d384 | training | disabled | 640 | — | 23.0390 | cuequivariance | 48.0425 | 2.09× | — |
| d384 | training | disabled | 768 | — | 34.1560 | cuequivariance | 67.8533 | 1.99× | — |
| d384 | training | manual | 128 | — | 1.0445 | cuequivariance | 1.5206 | 1.46× | — |
| d384 | training | manual | 256 | — | 3.6526 | cuequivariance | 7.6078 | 2.08× | — |
| d384 | training | manual | 384 | — | 7.8966 | cuequivariance | 16.9779 | 2.15× | — |
| d384 | training | manual | 512 | — | 14.3048 | cuequivariance | 30.3176 | 2.12× | — |
| d384 | training | manual | 640 | — | 22.7108 | cuequivariance | 47.5535 | 2.09× | — |
| d384 | training | manual | 768 | — | 33.4193 | cuequivariance | 67.0054 | 2.00× | — |
| d512 | inference | manual | 128 | — | 0.8520 | anthropic | 미지원 | — | 0.6799 |
| d512 | inference | manual | 256 | — | 4.0233 | anthropic | 미지원 | — | 3.3823 |
| d512 | inference | manual | 384 | — | 8.8914 | anthropic | 미지원 | — | 7.4885 |
| d512 | inference | manual | 512 | — | 19.9414 | anthropic | 미지원 | — | 17.6681 |
| d512 | inference | manual | 640 | — | 37.2239 | anthropic | 미지원 | — | 33.3594 |
| d512 | inference | manual | 768 | — | 55.6554 | anthropic | 미지원 | — | 50.1801 |
| d512 | training | disabled | 128 | — | 4.3612 | cuequivariance | 2.9578 | 0.68× | — |
| d512 | training | disabled | 256 | — | 12.6177 | cuequivariance | 12.7263 | 1.01× | — |
| d512 | training | disabled | 384 | — | 29.8424 | cuequivariance | 28.4488 | 0.95× | — |
| d512 | training | disabled | 512 | — | 73.1279 | cuequivariance | 50.5651 | 0.69× | — |
| d512 | training | disabled | 640 | — | 140.7549 | cuequivariance | 79.3969 | 0.56× | — |
| d512 | training | disabled | 768 | — | 214.9018 | cuequivariance | 114.6675 | 0.53× | — |
| d512 | training | manual | 128 | — | 2.7924 | cuequivariance | 2.4934 | 0.89× | — |
| d512 | training | manual | 256 | — | 12.5737 | cuequivariance | 12.6966 | 1.01× | — |
| d512 | training | manual | 384 | — | 29.8670 | cuequivariance | 28.4518 | 0.95× | — |
| d512 | training | manual | 512 | — | 73.6184 | cuequivariance | 50.6010 | 0.69× | — |
| d512 | training | manual | 640 | — | 140.5573 | cuequivariance | 79.3303 | 0.56× | — |
| d512 | training | manual | 768 | — | 214.9755 | cuequivariance | 115.0024 | 0.53× | — |
| d64 | inference | manual | 128 | — | 0.0410 | anthropic | 0.0379 | 0.93× | 0.1044 |
| d64 | inference | manual | 256 | — | 0.1004 | anthropic | 0.0829 | 0.83× | 0.3031 |
| d64 | inference | manual | 384 | — | 0.1997 | anthropic | 0.1761 | 0.88× | 0.6707 |
| d64 | inference | manual | 512 | — | 0.3446 | anthropic | 0.2949 | 0.86× | 1.5821 |
| d64 | inference | manual | 640 | — | 0.5704 | anthropic | 0.4854 | 0.85× | 3.5681 |
| d64 | inference | manual | 768 | — | 0.8489 | anthropic | 0.7260 | 0.86× | 5.4999 |
| d64 | training | disabled | 128 | — | 1.4305 | cuequivariance | 2.4115 | 1.69× | — |
| d64 | training | disabled | 256 | — | 1.5780 | cuequivariance | 2.4433 | 1.55× | — |
| d64 | training | disabled | 384 | — | 1.5949 | cuequivariance | 2.3352 | 1.46× | — |
| d64 | training | disabled | 512 | — | 1.6072 | cuequivariance | 2.4975 | 1.55× | — |
| d64 | training | disabled | 640 | — | 2.4986 | cuequivariance | 3.6480 | 1.46× | — |
| d64 | training | disabled | 768 | — | 3.6096 | cuequivariance | 5.3489 | 1.48× | — |
| d64 | training | manual | 128 | — | 0.1812 | cuequivariance | 0.2427 | 1.34× | — |
| d64 | training | manual | 256 | — | 0.4444 | cuequivariance | 0.6267 | 1.41× | — |
| d64 | training | manual | 384 | — | 0.9257 | cuequivariance | 1.3527 | 1.46× | — |
| d64 | training | manual | 512 | — | 1.5887 | cuequivariance | 2.3142 | 1.46× | — |
| d64 | training | manual | 640 | — | 2.4791 | cuequivariance | 3.6311 | 1.46× | — |
| d64 | training | manual | 768 | — | 3.6270 | cuequivariance | 5.2122 | 1.44× | — |

## TriMul 양방향

| 변형 | 모드 | graph | L 또는 N | A | ours ms | baseline | baseline ms | baseline / ours | PyTorch compiled ms |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| d128 | inference | manual | 128 | — | 0.0952 | anthropic | 0.1382 | 1.45× | 0.2918 |
| d128 | inference | manual | 256 | — | 0.2734 | anthropic | 0.4147 | 1.52× | 1.0885 |
| d128 | inference | manual | 384 | — | 0.6164 | anthropic | 0.8504 | 1.38× | 2.5754 |
| d128 | inference | manual | 512 | — | 1.1464 | anthropic | 1.5642 | 1.36× | 6.5219 |
| d128 | inference | manual | 640 | — | 1.9098 | anthropic | 2.5231 | 1.32× | 13.3007 |
| d128 | inference | manual | 768 | — | 2.9133 | anthropic | 3.7929 | 1.30× | 20.3873 |
| d128 | training | disabled | 128 | — | 1.6538 | cuequivariance | 1.9548 | 1.18× | — |
| d128 | training | disabled | 256 | — | 1.5544 | cuequivariance | 2.2323 | 1.44× | — |
| d128 | training | disabled | 384 | — | 2.5728 | cuequivariance | 4.4780 | 1.74× | — |
| d128 | training | disabled | 512 | — | 4.6607 | cuequivariance | 7.8428 | 1.68× | — |
| d128 | training | disabled | 640 | — | 7.6467 | cuequivariance | 12.5420 | 1.64× | — |
| d128 | training | disabled | 768 | — | 11.3111 | cuequivariance | 18.4125 | 1.63× | — |
| d128 | training | manual | 128 | — | 0.4183 | cuequivariance | 0.5837 | 1.40× | — |
| d128 | training | manual | 256 | — | 1.1971 | cuequivariance | 2.0449 | 1.71× | — |
| d128 | training | manual | 384 | — | 2.5569 | cuequivariance | 4.4984 | 1.76× | — |
| d128 | training | manual | 512 | — | 4.5916 | cuequivariance | 7.8889 | 1.72× | — |
| d128 | training | manual | 640 | — | 7.5264 | cuequivariance | 12.6607 | 1.68× | — |
| d128 | training | manual | 768 | — | 11.2374 | cuequivariance | 18.4873 | 1.65× | — |
| d256 | inference | manual | 128 | — | 0.2376 | anthropic | 미지원 | — | 0.5448 |
| d256 | inference | manual | 256 | — | 0.8991 | anthropic | 미지원 | — | 2.8539 |
| d256 | inference | manual | 384 | — | 1.9784 | anthropic | 미지원 | — | 6.1614 |
| d256 | inference | manual | 512 | — | 3.6147 | anthropic | 미지원 | — | 15.3861 |
| d256 | inference | manual | 640 | — | 5.8614 | anthropic | 미지원 | — | 29.5291 |
| d256 | inference | manual | 768 | — | 8.7665 | anthropic | 미지원 | — | 45.2198 |
| d256 | training | disabled | 128 | — | 1.6502 | cuequivariance | 2.1381 | 1.30× | — |
| d256 | training | disabled | 256 | — | 3.5374 | cuequivariance | 5.0985 | 1.44× | — |
| d256 | training | disabled | 384 | — | 7.6882 | cuequivariance | 11.2292 | 1.46× | — |
| d256 | training | disabled | 512 | — | 13.7431 | cuequivariance | 20.0325 | 1.46× | — |
| d256 | training | disabled | 640 | — | 22.1251 | cuequivariance | 31.8382 | 1.44× | — |
| d256 | training | disabled | 768 | — | 32.1720 | cuequivariance | 46.2290 | 1.44× | — |
| d256 | training | manual | 128 | — | 1.0363 | cuequivariance | 1.2626 | 1.22× | — |
| d256 | training | manual | 256 | — | 3.4437 | cuequivariance | 5.0504 | 1.47× | — |
| d256 | training | manual | 384 | — | 7.4793 | cuequivariance | 11.1427 | 1.49× | — |
| d256 | training | manual | 512 | — | 13.5188 | cuequivariance | 19.6946 | 1.46× | — |
| d256 | training | manual | 640 | — | 21.8429 | cuequivariance | 31.2054 | 1.43× | — |
| d256 | training | manual | 768 | — | 31.8966 | cuequivariance | 46.1947 | 1.45× | — |
| d384 | inference | manual | 128 | — | 0.4096 | anthropic | 미지원 | — | 0.9011 |
| d384 | inference | manual | 256 | — | 1.5698 | anthropic | 미지원 | — | 4.1349 |
| d384 | inference | manual | 384 | — | 3.5103 | anthropic | 미지원 | — | 10.5452 |
| d384 | inference | manual | 512 | — | 6.3898 | anthropic | 미지원 | — | 23.7266 |
| d384 | inference | manual | 640 | — | 10.3506 | anthropic | 미지원 | — | 51.9506 |
| d384 | inference | manual | 768 | — | 15.1060 | anthropic | 미지원 | — | 79.1142 |
| d384 | training | disabled | 128 | — | 1.7869 | cuequivariance | 2.2282 | 1.25× | — |
| d384 | training | disabled | 256 | — | 6.3432 | cuequivariance | 9.9523 | 1.57× | — |
| d384 | training | disabled | 384 | — | 13.6428 | cuequivariance | 22.2597 | 1.63× | — |
| d384 | training | disabled | 512 | — | 24.8428 | cuequivariance | 40.0123 | 1.61× | — |
| d384 | training | disabled | 640 | — | 39.8551 | cuequivariance | 63.7583 | 1.60× | — |
| d384 | training | disabled | 768 | — | 59.6613 | cuequivariance | 92.2583 | 1.55× | — |
| d384 | training | manual | 128 | — | 1.8068 | cuequivariance | 2.2426 | 1.24× | — |
| d384 | training | manual | 256 | — | 6.4952 | cuequivariance | 9.8028 | 1.51× | — |
| d384 | training | manual | 384 | — | 14.1599 | cuequivariance | 22.0088 | 1.55× | — |
| d384 | training | manual | 512 | — | 25.4894 | cuequivariance | 40.0579 | 1.57× | — |
| d384 | training | manual | 640 | — | 40.4091 | cuequivariance | 62.8091 | 1.55× | — |
| d384 | training | manual | 768 | — | 59.7402 | cuequivariance | 91.0029 | 1.52× | — |
| d512 | inference | manual | 128 | — | 2.0664 | anthropic | 미지원 | — | 1.6118 |
| d512 | inference | manual | 256 | — | 9.3594 | anthropic | 미지원 | — | 7.0431 |
| d512 | inference | manual | 384 | — | 22.1425 | anthropic | 미지원 | — | 15.4424 |
| d512 | inference | manual | 512 | — | 53.2193 | anthropic | 미지원 | — | 35.3029 |
| d512 | inference | manual | 640 | — | 95.1808 | anthropic | 미지원 | — | 71.8193 |
| d512 | inference | manual | 768 | — | 146.8457 | anthropic | 미지원 | — | 108.2849 |
| d512 | training | disabled | 128 | — | 5.6525 | cuequivariance | 3.5067 | 0.62× | — |
| d512 | training | disabled | 256 | — | 23.6206 | cuequivariance | 16.2555 | 0.69× | — |
| d512 | training | disabled | 384 | — | 56.0220 | cuequivariance | 36.0980 | 0.64× | — |
| d512 | training | disabled | 512 | — | 131.4744 | cuequivariance | 65.8586 | 0.50× | — |
| d512 | training | disabled | 640 | — | 258.5457 | cuequivariance | 103.4691 | 0.40× | — |
| d512 | training | disabled | 768 | — | 393.7659 | cuequivariance | 150.0600 | 0.38× | — |
| d512 | training | manual | 128 | — | 5.5112 | cuequivariance | 3.3597 | 0.61× | — |
| d512 | training | manual | 256 | — | 23.6928 | cuequivariance | 16.1029 | 0.68× | — |
| d512 | training | manual | 384 | — | 55.3011 | cuequivariance | 36.2409 | 0.66× | — |
| d512 | training | manual | 512 | — | 132.0509 | cuequivariance | 65.6794 | 0.50× | — |
| d512 | training | manual | 640 | — | 260.0868 | cuequivariance | 103.8643 | 0.40× | — |
| d512 | training | manual | 768 | — | 395.0920 | cuequivariance | 151.2407 | 0.38× | — |
| d64 | inference | manual | 128 | — | 0.0594 | anthropic | 0.0635 | 1.07× | 0.1608 |
| d64 | inference | manual | 256 | — | 0.1690 | anthropic | 0.1587 | 0.94× | 0.5110 |
| d64 | inference | manual | 384 | — | 0.3533 | anthropic | 0.3297 | 0.93× | 1.1971 |
| d64 | inference | manual | 512 | — | 0.6328 | anthropic | 0.5908 | 0.93× | 2.8948 |
| d64 | inference | manual | 640 | — | 1.0476 | anthropic | 0.9892 | 0.94× | 6.6012 |
| d64 | inference | manual | 768 | — | 1.5677 | anthropic | 1.5212 | 0.97× | 9.9963 |
| d64 | training | disabled | 128 | — | 1.4454 | cuequivariance | 2.1248 | 1.47× | — |
| d64 | training | disabled | 256 | — | 1.6051 | cuequivariance | 2.1391 | 1.33× | — |
| d64 | training | disabled | 384 | — | 1.6394 | cuequivariance | 2.1094 | 1.29× | — |
| d64 | training | disabled | 512 | — | 2.6040 | cuequivariance | 3.6091 | 1.39× | — |
| d64 | training | disabled | 640 | — | 4.1554 | cuequivariance | 5.6965 | 1.37× | — |
| d64 | training | disabled | 768 | — | 6.1030 | cuequivariance | 8.2714 | 1.36× | — |
| d64 | training | manual | 128 | — | 0.2478 | cuequivariance | 0.3328 | 1.34× | — |
| d64 | training | manual | 256 | — | 0.7168 | cuequivariance | 0.9738 | 1.36× | — |
| d64 | training | manual | 384 | — | 1.5012 | cuequivariance | 2.0849 | 1.39× | — |
| d64 | training | manual | 512 | — | 2.5713 | cuequivariance | 3.5942 | 1.40× | — |
| d64 | training | manual | 640 | — | 4.1155 | cuequivariance | 5.6637 | 1.38× | — |
| d64 | training | manual | 768 | — | 6.0774 | cuequivariance | 8.2504 | 1.36× | — |

## TriangleAttention

| 변형 | 모드 | graph | L 또는 N | A | ours ms | baseline | baseline ms | baseline / ours | PyTorch compiled ms |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| d128 | inference | manual | 128 | — | 0.0768 | anthropic | 0.1198 † | 1.56× | 0.1956 |
| d128 | inference | manual | 256 | — | 0.2611 | anthropic | 0.3338 † | 1.28× | 0.8965 |
| d128 | inference | manual | 384 | — | 0.6380 | anthropic | 0.7322 † | 1.15× | 3.3126 |
| d128 | inference | manual | 512 | — | 1.2892 | anthropic | 1.3793 † | 1.07× | 5.8511 |
| d128 | inference | manual | 640 | — | 2.3122 | anthropic | 2.3153 † | 1.00× | 14.6898 |
| d128 | inference | manual | 768 | — | 3.6864 | anthropic | 3.5635 † | 0.97× | 22.4932 |
| d128 | training | disabled | 128 | — | 1.4346 | cuequivariance | 1.4797 | 1.03× | — |
| d128 | training | disabled | 256 | — | 1.4572 | cuequivariance | 2.1647 | 1.49× | — |
| d128 | training | disabled | 384 | — | 2.6522 | cuequivariance | 5.6197 | 2.12× | — |
| d128 | training | disabled | 512 | — | 5.2337 | cuequivariance | 11.3679 | 2.17× | — |
| d128 | training | disabled | 640 | — | 9.5160 | cuequivariance | 20.3412 | 2.14× | — |
| d128 | training | disabled | 768 | — | 15.2484 | cuequivariance | 32.9779 | 2.16× | — |
| d128 | training | manual | 128 | — | 0.3195 | cuequivariance | 0.5048 | 1.58× | — |
| d128 | training | manual | 256 | — | 1.0353 | cuequivariance | 2.1033 | 2.03× | — |
| d128 | training | manual | 384 | — | 2.6040 | cuequivariance | 5.5818 | 2.14× | — |
| d128 | training | manual | 512 | — | 5.1732 | cuequivariance | 11.2691 | 2.18× | — |
| d128 | training | manual | 640 | — | 9.2355 | cuequivariance | 20.3003 | 2.20× | — |
| d128 | training | manual | 768 | — | 14.8813 | cuequivariance | 32.7895 | 2.20× | — |
| d256 | inference | manual | 128 | — | 0.1802 | anthropic | 0.2662 † | 1.48× | 0.3594 |
| d256 | inference | manual | 256 | — | 0.7444 | anthropic | 0.8049 † | 1.08× | 2.2508 |
| d256 | inference | manual | 384 | — | 1.8652 | anthropic | 1.9118 † | 1.02× | 7.3800 |
| d256 | inference | manual | 512 | — | 3.7315 | anthropic | 3.6209 † | 0.97× | 15.2177 |
| d256 | inference | manual | 640 | — | 6.2915 | anthropic | 6.0882 † | 0.97× | 34.0547 |
| d256 | inference | manual | 768 | — | 9.9149 | anthropic | 9.3829 † | 0.95× | 53.5091 |
| d256 | training | disabled | 128 | — | 1.2191 | cuequivariance | 1.4182 | 1.16× | — |
| d256 | training | disabled | 256 | — | 2.4812 | cuequivariance | 4.1318 | 1.67× | — |
| d256 | training | disabled | 384 | — | 6.1706 | cuequivariance | 10.8800 | 1.76× | — |
| d256 | training | disabled | 512 | — | 12.3540 | cuequivariance | 22.3667 | 1.81× | — |
| d256 | training | disabled | 640 | — | 22.1665 | cuequivariance | 41.4956 | 1.87× | — |
| d256 | training | disabled | 768 | — | 35.0249 | cuequivariance | 72.0865 | 2.06× | — |
| d256 | training | manual | 128 | — | 0.6308 | cuequivariance | 0.9503 | 1.51× | — |
| d256 | training | manual | 256 | — | 2.5226 | cuequivariance | 4.0714 | 1.61× | — |
| d256 | training | manual | 384 | — | 6.3468 | cuequivariance | 10.9732 | 1.73× | — |
| d256 | training | manual | 512 | — | 12.7355 | cuequivariance | 22.1358 | 1.74× | — |
| d256 | training | manual | 640 | — | 22.5495 | cuequivariance | 42.1268 | 1.87× | — |
| d256 | training | manual | 768 | — | 35.5799 | cuequivariance | 73.6983 | 2.07× | — |
| d384 | inference | manual | 128 | — | 0.2836 | anthropic | 미지원 | — | 0.5560 |
| d384 | inference | manual | 256 | — | 1.2564 | anthropic | 미지원 | — | 3.5656 |
| d384 | inference | manual | 384 | — | 3.1299 | anthropic | 미지원 | — | 11.6434 |
| d384 | inference | manual | 512 | — | 6.1297 | anthropic | 미지원 | — | 23.8976 |
| d384 | inference | manual | 640 | — | 10.4417 | anthropic | 미지원 | — | 53.4047 |
| d384 | inference | manual | 768 | — | 16.2412 | anthropic | 미지원 | — | 83.1990 |
| d384 | training | disabled | 128 | — | 2.1955 | cuequivariance | 1.9753 | 0.90× | — |
| d384 | training | disabled | 256 | — | 4.1073 | cuequivariance | 6.4246 | 1.56× | — |
| d384 | training | disabled | 384 | — | 10.0434 | cuequivariance | 16.9974 | 1.69× | — |
| d384 | training | disabled | 512 | — | 19.7325 | cuequivariance | 35.7832 | 1.81× | — |
| d384 | training | disabled | 640 | — | 34.4986 | cuequivariance | 69.8624 | 2.03× | — |
| d384 | training | disabled | 768 | — | 54.4174 | cuequivariance | 114.9573 | 2.11× | — |
| d384 | training | manual | 128 | — | 0.9938 | cuequivariance | 1.3957 | 1.40× | — |
| d384 | training | manual | 256 | — | 4.0038 | cuequivariance | 6.4369 | 1.61× | — |
| d384 | training | manual | 384 | — | 10.2769 | cuequivariance | 17.2493 | 1.68× | — |
| d384 | training | manual | 512 | — | 19.9997 | cuequivariance | 35.3270 | 1.77× | — |
| d384 | training | manual | 640 | — | 34.8124 | cuequivariance | 70.9140 | 2.04× | — |
| d384 | training | manual | 768 | — | 55.6739 | cuequivariance | 116.3080 | 2.09× | — |
| d512 | inference | manual | 128 | — | 0.4260 | anthropic | 미지원 | — | 0.9206 |
| d512 | inference | manual | 256 | — | 1.8883 | anthropic | 미지원 | — | 5.4400 |
| d512 | inference | manual | 384 | — | 4.6520 | anthropic | 미지원 | — | 17.1878 |
| d512 | inference | manual | 512 | — | 9.0163 | anthropic | 미지원 | — | 36.5916 |
| d512 | inference | manual | 640 | — | 15.2750 | anthropic | 미지원 | — | 75.5681 |
| d512 | inference | manual | 768 | — | 23.5392 | anthropic | 미지원 | — | 127.5914 |
| d512 | training | disabled | 128 | — | 1.9021 | cuequivariance | 1.9794 | 1.04× | — |
| d512 | training | disabled | 256 | — | 5.9049 | cuequivariance | 8.8740 | 1.50× | — |
| d512 | training | disabled | 384 | — | 14.6371 | cuequivariance | 24.0410 | 1.64× | — |
| d512 | training | disabled | 512 | — | 28.2614 | cuequivariance | 50.7781 | 1.80× | — |
| d512 | training | disabled | 640 | — | 49.3332 | cuequivariance | 97.6845 | 1.98× | — |
| d512 | training | disabled | 768 | — | 77.2577 | cuequivariance | 161.2012 | 2.09× | — |
| d512 | training | manual | 128 | — | 1.4694 | cuequivariance | 1.9256 | 1.31× | — |
| d512 | training | manual | 256 | — | 5.8388 | cuequivariance | 8.7665 | 1.50× | — |
| d512 | training | manual | 384 | — | 14.6657 | cuequivariance | 23.9160 | 1.63× | — |
| d512 | training | manual | 512 | — | 28.7570 | cuequivariance | 51.6065 | 1.79× | — |
| d512 | training | manual | 640 | — | 50.0531 | cuequivariance | 99.3126 | 1.98× | — |
| d512 | training | manual | 768 | — | 78.3893 | cuequivariance | 164.1964 | 2.09× | — |
| d64 | inference | manual | 128 | — | 0.0430 | anthropic | 미지원 | — | 0.1126 |
| d64 | inference | manual | 256 | — | 0.1249 | anthropic | 미지원 | — | 0.5069 |
| d64 | inference | manual | 384 | — | 0.3113 | anthropic | 미지원 | — | 1.7490 |
| d64 | inference | manual | 512 | — | 0.6113 | anthropic | 미지원 | — | 2.8549 |
| d64 | inference | manual | 640 | — | 1.1110 | anthropic | 미지원 | — | 7.2038 |
| d64 | inference | manual | 768 | — | 1.7961 | anthropic | 미지원 | — | 10.8964 |
| d64 | training | disabled | 128 | — | 1.2780 | cuequivariance | 1.6010 | 1.25× | — |
| d64 | training | disabled | 256 | — | 1.1479 | cuequivariance | 1.4100 | 1.23× | — |
| d64 | training | disabled | 384 | — | 1.3158 | cuequivariance | 2.9087 | 2.21× | — |
| d64 | training | disabled | 512 | — | 2.5764 | cuequivariance | 5.8010 | 2.25× | — |
| d64 | training | disabled | 640 | — | 4.5588 | cuequivariance | 10.3342 | 2.27× | — |
| d64 | training | disabled | 768 | — | 7.3370 | cuequivariance | 16.5652 | 2.26× | — |
| d64 | training | manual | 128 | — | 0.1792 | cuequivariance | 0.3246 | 1.81× | — |
| d64 | training | manual | 256 | — | 0.5396 | cuequivariance | 1.1284 | 2.09× | — |
| d64 | training | manual | 384 | — | 1.2820 | cuequivariance | 2.8426 | 2.22× | — |
| d64 | training | manual | 512 | — | 2.5185 | cuequivariance | 5.7308 | 2.28× | — |
| d64 | training | manual | 640 | — | 4.5220 | cuequivariance | 10.2676 | 2.27× | — |
| d64 | training | manual | 768 | — | 7.2458 | cuequivariance | 16.5161 | 2.28× | — |

## AttentionPairBias

| 변형 | 모드 | graph | L 또는 N | A | ours ms | baseline | baseline ms | baseline / ours | PyTorch compiled ms |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| h12d384 | inference | manual | 128 | — | 0.0410 | anthropic | 0.0420 † | 1.03× | 0.0696 |
| h12d384 | inference | manual | 256 | — | 0.0594 | anthropic | 0.0666 † | 1.12× | 0.1085 |
| h12d384 | inference | manual | 384 | — | 0.0829 | anthropic | 0.0973 † | 1.17× | 0.1690 |
| h12d384 | inference | manual | 512 | — | 0.1096 | anthropic | 0.1362 † | 1.24× | 0.2468 |
| h12d384 | inference | manual | 640 | — | 0.1434 | anthropic | 0.1741 † | 1.21× | 0.3451 |
| h12d384 | inference | manual | 768 | — | 0.1792 | anthropic | 0.2273 † | 1.27× | 0.4649 |
| h12d384 | training | disabled | 128 | — | 1.7833 | cuequivariance | 1.2923 | 0.72× | — |
| h12d384 | training | disabled | 256 | — | 1.7408 | cuequivariance | 1.4561 | 0.84× | — |
| h12d384 | training | disabled | 384 | — | 1.7644 | cuequivariance | 1.3445 | 0.76× | — |
| h12d384 | training | disabled | 512 | — | 1.7162 | cuequivariance | 1.4060 | 0.82× | — |
| h12d384 | training | disabled | 640 | — | 1.7818 | cuequivariance | 1.4336 | 0.80× | — |
| h12d384 | training | disabled | 768 | — | 1.8058 | cuequivariance | 1.4664 | 0.81× | — |
| h12d384 | training | manual | 128 | — | 0.1372 | cuequivariance | 0.1884 | 1.37× | — |
| h12d384 | training | manual | 256 | — | 0.1843 | cuequivariance | 0.3154 | 1.71× | — |
| h12d384 | training | manual | 384 | — | 0.2601 | cuequivariance | 0.4844 | 1.86× | — |
| h12d384 | training | manual | 512 | — | 0.3441 | cuequivariance | 0.6943 | 2.02× | — |
| h12d384 | training | manual | 640 | — | 0.4613 | cuequivariance | 0.9431 | 2.04× | — |
| h12d384 | training | manual | 768 | — | 0.5868 | cuequivariance | 1.2564 | 2.14× | — |
| h16d384 | inference | manual | 128 | — | 0.0420 | anthropic | 0.0420 † | 1.00× | 0.0707 |
| h16d384 | inference | manual | 256 | — | 0.0604 | anthropic | 0.0666 † | 1.10× | 0.1075 |
| h16d384 | inference | manual | 384 | — | 0.0850 | anthropic | 0.0983 † | 1.16× | 0.1710 |
| h16d384 | inference | manual | 512 | — | 0.1147 | anthropic | 0.1403 † | 1.22× | 0.2519 |
| h16d384 | inference | manual | 640 | — | 0.1464 | anthropic | 0.1761 † | 1.20× | 0.3482 |
| h16d384 | inference | manual | 768 | — | 0.1843 | anthropic | 0.2314 † | 1.26× | 0.4649 |
| h16d384 | training | disabled | 128 | — | 1.7459 | cuequivariance | 1.1377 | 0.65× | — |
| h16d384 | training | disabled | 256 | — | 1.7500 | cuequivariance | 1.4029 | 0.80× | — |
| h16d384 | training | disabled | 384 | — | 1.7326 | cuequivariance | 1.4054 | 0.81× | — |
| h16d384 | training | disabled | 512 | — | 1.6138 | cuequivariance | 1.2646 | 0.78× | — |
| h16d384 | training | disabled | 640 | — | 1.6312 | cuequivariance | 1.4177 | 0.87× | — |
| h16d384 | training | disabled | 768 | — | 1.6323 | cuequivariance | 1.4377 | 0.88× | — |
| h16d384 | training | manual | 128 | — | 0.1444 | cuequivariance | 0.1884 | 1.30× | — |
| h16d384 | training | manual | 256 | — | 0.1925 | cuequivariance | 0.3256 | 1.69× | — |
| h16d384 | training | manual | 384 | — | 0.2744 | cuequivariance | 0.4997 | 1.82× | — |
| h16d384 | training | manual | 512 | — | 0.3717 | cuequivariance | 0.7148 | 1.92× | — |
| h16d384 | training | manual | 640 | — | 0.4966 | cuequivariance | 0.9779 | 1.97× | — |
| h16d384 | training | manual | 768 | — | 0.6380 | cuequivariance | 1.3158 | 2.06× | — |
| h16d512 | inference | manual | 128 | — | 0.0430 | anthropic | 0.0430 † | 1.00× | 0.0707 |
| h16d512 | inference | manual | 256 | — | 0.0625 | anthropic | 0.0686 † | 1.10× | 0.1096 |
| h16d512 | inference | manual | 384 | — | 0.0870 | anthropic | 0.0993 † | 1.14× | 0.1741 |
| h16d512 | inference | manual | 512 | — | 0.1167 | anthropic | 0.1454 † | 1.25× | 0.2550 |
| h16d512 | inference | manual | 640 | — | 0.1495 | anthropic | 0.1802 † | 1.21× | 0.3533 |
| h16d512 | inference | manual | 768 | — | 0.1874 | anthropic | 0.2355 † | 1.26× | 0.4690 |
| h16d512 | training | disabled | 128 | — | 1.4828 | cuequivariance | 0.9472 | 0.64× | — |
| h16d512 | training | disabled | 256 | — | 1.4776 | cuequivariance | 1.2370 | 0.84× | — |
| h16d512 | training | disabled | 384 | — | 1.5872 | cuequivariance | 1.2396 | 0.78× | — |
| h16d512 | training | disabled | 512 | — | 1.5114 | cuequivariance | 1.2175 | 0.81× | — |
| h16d512 | training | disabled | 640 | — | 1.4991 | cuequivariance | 1.1182 | 0.75× | — |
| h16d512 | training | disabled | 768 | — | 1.7188 | cuequivariance | 1.3701 | 0.80× | — |
| h16d512 | training | manual | 128 | — | 0.1526 | cuequivariance | 0.1935 | 1.27× | — |
| h16d512 | training | manual | 256 | — | 0.2028 | cuequivariance | 0.3267 | 1.61× | — |
| h16d512 | training | manual | 384 | — | 0.2816 | cuequivariance | 0.5253 | 1.87× | — |
| h16d512 | training | manual | 512 | — | 0.3789 | cuequivariance | 0.7168 | 1.89× | — |
| h16d512 | training | manual | 640 | — | 0.5028 | cuequivariance | 0.9769 | 1.94× | — |
| h16d512 | training | manual | 768 | — | 0.6359 | cuequivariance | 1.3138 | 2.07× | — |
| h24d384 | inference | manual | 128 | — | 0.0645 | anthropic | 0.0451 † | 0.70× | 0.0696 |
| h24d384 | inference | manual | 256 | — | 0.0942 | anthropic | 0.0707 † | 0.75× | 0.1075 |
| h24d384 | inference | manual | 384 | — | 0.1536 | anthropic | 0.1044 † | 0.68× | 0.1782 |
| h24d384 | inference | manual | 512 | — | 0.2243 | anthropic | 0.1495 † | 0.67× | 0.2580 |
| h24d384 | inference | manual | 640 | — | 0.3164 | anthropic | 0.2089 † | 0.66× | 0.3676 |
| h24d384 | inference | manual | 768 | — | 0.4413 | anthropic | 0.2826 † | 0.64× | 0.4997 |
| h24d384 | training | disabled | 128 | — | 2.0490 | cuequivariance | 0.9411 | 0.46× | — |
| h24d384 | training | disabled | 256 | — | 2.5646 | cuequivariance | 1.1080 | 0.43× | — |
| h24d384 | training | disabled | 384 | — | 1.9681 | cuequivariance | 1.1581 | 0.59× | — |
| h24d384 | training | disabled | 512 | — | 1.9773 | cuequivariance | 1.1674 | 0.59× | — |
| h24d384 | training | disabled | 640 | — | 2.0152 | cuequivariance | 1.2042 | 0.60× | — |
| h24d384 | training | disabled | 768 | — | 1.8780 | cuequivariance | 1.5524 | 0.83× | — |
| h24d384 | training | manual | 128 | — | 0.1720 | cuequivariance | 0.1884 | 1.10× | — |
| h24d384 | training | manual | 256 | — | 0.2632 | cuequivariance | 0.3297 | 1.25× | — |
| h24d384 | training | manual | 384 | — | 0.4332 | cuequivariance | 0.5181 | 1.20× | — |
| h24d384 | training | manual | 512 | — | 0.6410 | cuequivariance | 0.7598 | 1.19× | — |
| h24d384 | training | manual | 640 | — | 0.9175 | cuequivariance | 1.0665 | 1.16× | — |
| h24d384 | training | manual | 768 | — | 1.2595 | cuequivariance | 1.5032 | 1.19× | — |
| h8d384 | inference | manual | 128 | — | 0.0410 | anthropic | 0.0410 † | 1.00× | 0.0696 |
| h8d384 | inference | manual | 256 | — | 0.0584 | anthropic | 0.0614 † | 1.05× | 0.1055 |
| h8d384 | inference | manual | 384 | — | 0.0819 | anthropic | 0.0891 † | 1.09× | 0.1659 |
| h8d384 | inference | manual | 512 | — | 0.1075 | anthropic | 0.1260 † | 1.17× | 0.2376 |
| h8d384 | inference | manual | 640 | — | 0.1403 | anthropic | 0.1587 † | 1.13× | 0.3256 |
| h8d384 | inference | manual | 768 | — | 0.1751 | anthropic | 0.2089 † | 1.19× | 0.4311 |
| h8d384 | training | disabled | 128 | — | 2.0716 | cuequivariance | 1.0813 | 0.52× | — |
| h8d384 | training | disabled | 256 | — | 1.3891 | cuequivariance | 1.1269 | 0.81× | — |
| h8d384 | training | disabled | 384 | — | 1.3957 | cuequivariance | 1.0742 | 0.77× | — |
| h8d384 | training | disabled | 512 | — | 1.4019 | cuequivariance | 1.2145 | 0.87× | — |
| h8d384 | training | disabled | 640 | — | 1.4141 | cuequivariance | 1.3128 | 0.93× | — |
| h8d384 | training | disabled | 768 | — | 1.7352 | cuequivariance | 1.3384 | 0.77× | — |
| h8d384 | training | manual | 128 | — | 0.1382 | cuequivariance | 0.1864 | 1.35× | — |
| h8d384 | training | manual | 256 | — | 0.1843 | cuequivariance | 0.3103 | 1.68× | — |
| h8d384 | training | manual | 384 | — | 0.2632 | cuequivariance | 0.4628 | 1.76× | — |
| h8d384 | training | manual | 512 | — | 0.3492 | cuequivariance | 0.6646 | 1.90× | — |
| h8d384 | training | manual | 640 | — | 0.4628 | cuequivariance | 0.9134 | 1.97× | — |
| h8d384 | training | manual | 768 | — | 0.5939 | cuequivariance | 1.2104 | 2.04× | — |

## OPM

| 변형 | 모드 | graph | L 또는 N | A | ours ms | baseline | baseline ms | baseline / ours | PyTorch compiled ms |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| msa1024 | inference | manual | 128 | — | 0.2345 | anthropic | 0.6912 | 2.95× | 0.3133 |
| msa1024 | inference | manual | 256 | — | 0.8192 | anthropic | 2.2927 | 2.80× | 1.1018 |
| msa1024 | inference | manual | 384 | — | 1.8043 | anthropic | 4.8374 | 2.68× | 2.3081 |
| msa1024 | inference | manual | 512 | — | 3.1805 | anthropic | 8.2760 | 2.60× | 3.8922 |
| msa1024 | inference | manual | 640 | — | 4.9981 | anthropic | 12.6577 | 2.53× | 6.1865 |
| msa1024 | inference | manual | 768 | — | 7.0328 | anthropic | 17.9835 | 2.56× | 8.8873 |
| msa1024 | training | disabled | 128 | — | 1.1766 | pytorch | 1.1080 | 0.94× | — |
| msa1024 | training | disabled | 256 | — | 2.6419 | pytorch | 3.2164 | 1.22× | — |
| msa1024 | training | disabled | 384 | — | 5.6525 | pytorch | 6.6611 | 1.18× | — |
| msa1024 | training | disabled | 512 | — | 9.5457 | pytorch | 11.2538 | 1.18× | — |
| msa1024 | training | disabled | 640 | — | 14.5060 | pytorch | 17.0998 | 1.18× | — |
| msa1024 | training | disabled | 768 | — | 21.3135 | pytorch | 24.8545 | 1.17× | — |
| msa1024 | training | manual | 128 | — | 0.8607 | pytorch | 1.0680 | 1.24× | — |
| msa1024 | training | manual | 256 | — | 2.5651 | pytorch | 3.1386 | 1.22× | — |
| msa1024 | training | manual | 384 | — | 5.5398 | pytorch | 6.5649 | 1.19× | — |
| msa1024 | training | manual | 512 | — | 9.4392 | pytorch | 11.1790 | 1.18× | — |
| msa1024 | training | manual | 640 | — | 14.4261 | pytorch | 17.2114 | 1.19× | — |
| msa1024 | training | manual | 768 | — | 21.4656 | pytorch | 25.0307 | 1.17× | — |
| msa2048 | inference | manual | 128 | — | 0.3860 | anthropic | 1.0588 | 2.74× | 0.5468 |
| msa2048 | inference | manual | 256 | — | 1.4275 | anthropic | 3.2451 | 2.27× | 1.8309 |
| msa2048 | inference | manual | 384 | — | 3.1247 | anthropic | 6.7277 | 2.15× | 3.7094 |
| msa2048 | inference | manual | 512 | — | 5.4968 | anthropic | 11.3444 | 2.06× | 6.6734 |
| msa2048 | inference | manual | 640 | — | 8.6042 | anthropic | 17.2605 | 2.01× | 10.1919 |
| msa2048 | inference | manual | 768 | — | 12.3008 | anthropic | 24.3395 | 1.98× | 14.3647 |
| msa4096 | inference | manual | 128 | — | 0.6932 | anthropic | 1.7510 | 2.53× | 1.0004 |
| msa4096 | inference | manual | 256 | — | 2.6783 | anthropic | 5.2726 | 1.97× | 3.3157 |
| msa4096 | inference | manual | 384 | — | 5.9295 | anthropic | 10.7233 | 1.81× | 6.9683 |
| msa4096 | inference | manual | 512 | — | 10.3209 | anthropic | 17.9794 | 1.74× | 11.9654 |
| msa4096 | inference | manual | 640 | — | 16.1275 | anthropic | 27.3172 | 1.69× | 18.1852 |
| msa4096 | inference | manual | 768 | — | 22.7768 | anthropic | 37.9725 | 1.67× | 25.7341 |

## PWA

| 변형 | 모드 | graph | L 또는 N | A | ours ms | baseline | baseline ms | baseline / ours | PyTorch compiled ms |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| msa1024 | inference | manual | 128 | — | 0.3041 | anthropic | 0.4301 | 1.41× | 0.6124 |
| msa1024 | inference | manual | 256 | — | 0.6205 | anthropic | 0.8673 | 1.40× | 1.2544 |
| msa1024 | inference | manual | 384 | — | 1.0291 | anthropic | 1.5401 | 1.50× | 2.0091 |
| msa1024 | inference | manual | 512 | — | 1.5411 | anthropic | 2.2170 | 1.44× | 2.9276 |
| msa1024 | inference | manual | 640 | — | 2.1248 | anthropic | 3.0679 | 1.44× | 4.0463 |
| msa1024 | inference | manual | 768 | — | 2.8027 | anthropic | 4.1411 | 1.48× | 5.0852 |
| msa1024 | training | disabled | 128 | — | 1.3906 | pytorch | 1.7572 | 1.26× | — |
| msa1024 | training | disabled | 256 | — | 2.3613 | pytorch | 3.4176 | 1.45× | — |
| msa1024 | training | disabled | 384 | — | 3.8420 | pytorch | 5.5982 | 1.46× | — |
| msa1024 | training | disabled | 512 | — | 5.6064 | pytorch | 7.9155 | 1.41× | — |
| msa1024 | training | disabled | 640 | — | 7.5500 | pytorch | 10.7448 | 1.42× | — |
| msa1024 | training | disabled | 768 | — | 9.7531 | pytorch | 13.9090 | 1.43× | — |
| msa1024 | training | manual | 128 | — | 1.1254 | pytorch | 1.7715 | 1.57× | — |
| msa1024 | training | manual | 256 | — | 2.2979 | pytorch | 3.4591 | 1.51× | — |
| msa1024 | training | manual | 384 | — | 3.7304 | pytorch | 5.6781 | 1.52× | — |
| msa1024 | training | manual | 512 | — | 5.4246 | pytorch | 8.1946 | 1.51× | — |
| msa1024 | training | manual | 640 | — | 7.5254 | pytorch | 11.0679 | 1.47× | — |
| msa1024 | training | manual | 768 | — | 9.5877 | pytorch | 14.2531 | 1.49× | — |
| msa2048 | inference | manual | 128 | — | 0.5489 | anthropic | 0.7363 | 1.34× | 1.1715 |
| msa2048 | inference | manual | 256 | — | 1.2145 | anthropic | 1.6589 | 1.37× | 2.4996 |
| msa2048 | inference | manual | 384 | — | 2.0306 | anthropic | 2.8319 | 1.39× | 3.9956 |
| msa2048 | inference | manual | 512 | — | 2.9870 | anthropic | 4.2158 | 1.41× | 5.6197 |
| msa2048 | inference | manual | 640 | — | 4.1370 | anthropic | 5.8440 | 1.41× | 7.8275 |
| msa2048 | inference | manual | 768 | — | 5.5823 | anthropic | 7.6974 | 1.38× | 9.8140 |
| msa4096 | inference | manual | 128 | — | 1.0875 | anthropic | 1.4300 | 1.31× | 2.2948 |
| msa4096 | inference | manual | 256 | — | 2.4842 | anthropic | 3.2553 | 1.31× | 4.9260 |
| msa4096 | inference | manual | 384 | — | 4.2721 | anthropic | 5.5501 | 1.30× | 7.9401 |
| msa4096 | inference | manual | 512 | — | 6.6417 | anthropic | 8.2422 | 1.24× | 11.1237 |
| msa4096 | inference | manual | 640 | — | 9.2723 | anthropic | 11.4831 | 1.24× | 15.4353 |
| msa4096 | inference | manual | 768 | — | 12.2598 | anthropic | 15.0098 | 1.22× | 19.5092 |

## Pair Transition

| 변형 | 모드 | graph | L 또는 N | A | ours ms | baseline | baseline ms | baseline / ours | PyTorch compiled ms |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| d128 | inference | manual | 384 | — | 0.2703 | anthropic | 0.3789 | 1.40× | 0.9626 |
| d128 | inference | manual | 768 | — | 1.0609 | anthropic | 1.4971 | 1.41× | 3.7601 |
| d128 | training | disabled | 384 | — | 1.4367 | pytorch | 2.5078 | 1.75× | — |
| d128 | training | disabled | 768 | — | 5.5854 | pytorch | 9.5002 | 1.70× | — |
| d128 | training | manual | 384 | — | 1.4423 | pytorch | 2.4837 | 1.72× | — |
| d128 | training | manual | 768 | — | 5.5322 | pytorch | 9.4715 | 1.71× | — |
| d256 | inference | manual | 384 | — | 1.3650 | anthropic | 1.7132 | 1.26× | 2.0531 |
| d256 | inference | manual | 768 | — | 5.6340 | anthropic | 6.7328 | 1.20× | 8.1900 |
| d256 | training | disabled | 384 | — | 5.3545 | pytorch | 5.6699 | 1.06× | — |
| d256 | training | disabled | 768 | — | 20.4882 | pytorch | 21.3970 | 1.04× | — |
| d256 | training | manual | 384 | — | 5.2977 | pytorch | 5.6719 | 1.07× | — |
| d256 | training | manual | 768 | — | 21.0673 | pytorch | 22.0032 | 1.04× | — |
| d384 | inference | manual | 384 | — | 2.8692 | anthropic | 9.6404 | 3.36× | 3.7704 |
| d384 | inference | manual | 768 | — | 11.4688 | anthropic | 37.7160 | 3.29× | 15.1828 |
| d384 | training | disabled | 384 | — | 10.7059 | pytorch | 10.6260 | 0.99× | — |
| d384 | training | disabled | 768 | — | 42.6117 | pytorch | 41.9067 | 0.98× | — |
| d384 | training | manual | 384 | — | 10.6271 | pytorch | 10.5298 | 0.99× | — |
| d384 | training | manual | 768 | — | 43.6920 | pytorch | 42.3695 | 0.97× | — |
| d512 | inference | manual | 384 | — | 4.9644 | anthropic | 미지원 | — | 6.1041 |
| d512 | inference | manual | 768 | — | 19.5164 | anthropic | 미지원 | — | 24.4588 |
| d512 | training | disabled | 384 | — | 17.6292 | pytorch | 16.8627 | 0.96× | — |
| d512 | training | disabled | 768 | — | 70.2587 | pytorch | 66.5231 | 0.95× | — |
| d512 | training | manual | 384 | — | 17.5032 | pytorch | 16.6728 | 0.95× | — |
| d512 | training | manual | 768 | — | 69.0319 | pytorch | 66.3788 | 0.96× | — |
| d64 | inference | manual | 384 | — | 0.0922 | anthropic | 미지원 | — | 0.4475 |
| d64 | inference | manual | 768 | — | 0.3308 | anthropic | 미지원 | — | 1.6773 |
| d64 | training | disabled | 384 | — | 0.8571 | pytorch | 1.2595 | 1.47× | — |
| d64 | training | disabled | 768 | — | 1.8299 | pytorch | 4.4462 | 2.43× | — |
| d64 | training | manual | 384 | — | 0.4956 | pytorch | 1.2206 | 2.46× | — |
| d64 | training | manual | 768 | — | 1.8350 | pytorch | 4.4196 | 2.41× | — |

## 실패 및 해석 범위

Dense Atom DiT N8192/A48 학습의 PyTorch OOM은 backend별 독립 프로세스로 다시 확인했다. 길이와 A를 유지했으며 해당 조건의 속도비를 계산하지 않았다. A100 local Atom DiT의 ours 경로는 PyTorch local attention과 dispatched AdaLN/ConditionedTransition의 조합으로, B200 전용 local CUDA block의 성능을 뜻하지 않는다.

각 성공 행은 공통 harness의 유한값 검사를 통과했고, graph ON 행은 graph replay 검사도 통과했다. 모듈별 수치 정확도 검사 범위는 다르므로 모든 행에서 독립 reference 대비 전체 gradient 검증을 마쳤다는 뜻은 아니다. 이 측정은 성능 비교이며 CUDA-only 또는 SOL90 달성 검증은 아니다.

| 모듈 | backend | 실패 행 수 | 원인 |
|---|---|---:|---|
| triangle_attention | anthropic | 12 | InductorError: SubprocException: An exception occurred in a subprocess: |
| triangle_attention | anthropic | 6 | Unsupported: no-cell:fpf:prologue:64x2x32:8.0/3.7+off(not-measured) |
| transition | anthropic | 2 | UnsupportedBenchmark: no qualified Anthropic transition row for d=64, n=4, stream=pair (qualified: [128, 256, 384], n=4, pair) |
| triangle_multiplication_bidirectional | anthropic | 18 | UnsupportedBenchmark: miniworld_engine.integrations.anthropic_trimul.PayloadUnavailable: the anthropic TriMul payload cannot serve this call: the payload's sm_80 member h |
| triangle_attention | anthropic | 12 | Unsupported: pow2:c384h12d32 |
| triangle_multiplication | anthropic | 6 | UnsupportedBenchmark: miniworld_engine.integrations.anthropic_trimul.PayloadUnavailable: the anthropic TriMul payload cannot serve this call: the payload's sm_80 member h |
| triangle_attention | anthropic | 12 | Unsupported: no-cell:fpf:epilogue:512x16x32:8.0/3.7+no_safe(c_z_above_128_slower_than_stock_on_cc8.0) |
| transition | anthropic | 2 | UnsupportedBenchmark: no qualified Anthropic transition row for d=512, n=4, stream=pair (qualified: [128, 256, 384], n=4, pair) |
| dit_atom | pytorch | 1 | OutOfMemoryError: CUDA out of memory. Tried to allocate 24.00 GiB. GPU 0 has a total capacity of 79.25 GiB of which 800.94 MiB is free. Including non-PyTorch memory, this |
| dit_atom | pytorch | 1 | OutOfMemoryError: CUDA out of memory. Tried to allocate 24.00 GiB. GPU 0 has a total capacity of 79.25 GiB of which 490.94 MiB is free. Including non-PyTorch memory, this |
| swa_dit | anthropic | 6 | Unsupported: Attempted to call function marked as skipped |

[전체 CSV](a100-module-comparison-20261005.csv) · [실행 명령과 전체 JSON](a100-module-comparison-20261005.json)
