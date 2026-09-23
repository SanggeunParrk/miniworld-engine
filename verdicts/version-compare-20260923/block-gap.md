# MiniPairformer L384 학습: 1.520ms와 1.648ms의 차이

> **후속 확인:** 준비 비용 개선 이후 원본과 현재 구현을 같은 GPU에서 재현했다.
> 과거의 PyTorch/cuEq 교대 부하에서는 원본 1.517ms / 현재 1.522ms,
> CUDA 두 구현끼리만 교대하면 1.618ms / 1.622ms였다.
> SM 클럭 중앙값도 각각 1980 / 1830MHz로 달랐다.
> [동일 조건 재현 및 커널별 분석](../block-history-audit-20260923/README.md).
> 아래는 이 후속 검증 전의 비용 분리 기록이다.

조건: 블록 1개(양방향 TriMul + Transition), B1, L384, D128, BF16, dropout25%, forward + backward. 이 블록에는 OPM/PWA가 없으므로 MSA depth 변경은 이 시간에 영향을 주지 않는다.

## 확인된 기록

| 측정 | 시간 (ms) | 조건 |
|---|---:|---|
| 9월 22일 연구 벤치 `latest_all` | 1.520 | 고정 dropout mask, 연구용 실행 준비, 여러 backend 교대 측정 |
| 9월 23일 공개 모듈 비교 | 1.648 | 실제 dropout RNG 및 호출별 준비 포함 |
| 동일 GPU 재측정: 현재 공개 모듈 | 1.645 | 아래 두 조건과 교대 측정 |
| 동일 GPU: dropout mask 고정 | 1.646 | 나머지 모듈 동작 유지 |
| 동일 GPU: mask 고정 + TriMul 준비 결과 재사용 | 1.601 | 원인 분석용; 가중치 업데이트를 반영하지 않는 진단 조건 |

원본 과거 기록: `/home/psk6950/MiniWorld/runs/minipairformer_block_baselines_20260922/results.json` (job15592). 현재 원본은 [block-384-engine2.json](block-384-engine2.json), 재측정은 [block-gap.json](block-gap.json), 진단 구현은 [block_gap.py](block_gap.py)이다. 재측정은 job16102, 12회 × 80 CUDA graph replay의 중앙값이며 세 조건을 순환했다.

## 해석

- 1.648ms는 약 1.645ms로 재현됐다. 이전 1.520ms와는 약 128µs, 8.4% 차이다.
- dropout RNG를 제거해도 중앙값이 빨라지지 않았다. RNG만으로 차이를 설명할 수 없다.
- 호출별 가중치 packing·전치·mask 변환 등의 TriMul 준비를 제거하면 약 44.7µs가 줄었다. 진단용 준비 결과 재사용은 fixed-mask 조건과 출력·전체 gradient가 동일했지만, 가중치 업데이트를 반영하지 않으므로 production 최적화로 적용하지 않았다.
- 이전 trace는 19개, 현재는 46개 GPU 커널 호출이었다. 현재에는 추가 복사와 준비 커널이 있다. 프로파일 단일 replay의 시간 합을 반복 측정 중앙값의 정확한 비용 분해로 해석하면 안 된다.

## 클럭과 측정 부하도 다르다

| telemetry | 이전 연구 벤치 | 현재 재측정 |
|---|---:|---:|
| SM clock | 1980MHz 고정 | 1725–1935MHz, 중앙값 1860MHz |
| memory clock | 2619MHz | 2619MHz |
| 전력 중앙값 | 약 604W | 약 689W |

이전 벤치는 PyTorch/cuEq 등과 교대로 실행했고, 현재 진단은 native 경로를 지속 실행한다. 물리 GPU도 다르고 이후 gradient 수정 등 코드 변경도 있었다. 따라서 준비 비용을 제외한 나머지 차이를 특정 커널 퇴행이나 클럭 하나로 정량 귀속할 수 없다. 현재 production 시간을 1.520ms로 대체해서 보고하지 않는다.
