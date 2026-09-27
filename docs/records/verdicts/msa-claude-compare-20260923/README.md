# Claude OPM/PWA 개발 기록과 현재 비교표 대조

## 원본

- Claude 세션: `9d153bc6-f71b-4e9e-8dae-f58f7284c83a`, MiniWorld, 2026-09-23 09:25 KST 최종 표.
- 원본 transcript: `/home/psk6950/.claude/projects/-home-psk6950-MiniWorld/9d153bc6-f71b-4e9e-8dae-f58f7284c83a.jsonl`, line8583. [발췌](session-excerpts.json).
- 개발 엔진: `/home/psk6950/miniworld-engine-msa`, `research/msa-inference`, 최종 `b47de812` (학습 `7c5b6b6e`, pair3 `4ac6deb2`).
- 원본 실측 로그: `runs/msa_bench_20260921/enginfer-15823.log`, `engtime-15737.log`, `engopmtime-15640.log` (MiniWorld 기준).
- CUDA 소스 8개는 개발 worktree와 현재 비교표의 wheel 설치본이 SHA-256까지 동일하다. [해시 대조](source-audit.json). Python dispatch와 compile 경계는 변경됐다.

## 최종 커널 수치 대조

공통 shape: H100, B1, L384, MSA depth1024, MSA64/pair128/hidden32, PWA head8×32, BF16. 아래는 동일 조건 재측정 표가 아니라 서로 다른 두 벤치 기록의 대조다.

| 모듈/모드 | Claude 최종(ms) | 현재 표 Engine2(ms) | 시간 변화 |
|---|---:|---:|---:|
| PWA 추론 | 0.411 | 0.407 | -0.9% |
| PWA 학습 fwd+bwd | 1.561 | 1.586 | +1.6% |
| OPM 추론 | 0.633 | 0.703 | +11.1% |
| OPM 학습 fwd+bwd | 2.028 | 2.035 | +0.4% |

## 비교 기준과 측정 조건의 차이

| 항목 | Claude 벤치 | 현재 비교표 |
|---|---|---|
| 기존 엔진 의미 | 개발 브랜치의 새 경로를 끈 `own` | 실제 v1.0.0 태그 `850e160c` |
| 실행 방식 | eager, CUDA event로 Python 반복 구간 측정 | static fullgraph compile + CUDA graph replay |
| 반복 | warmup3, 추론20회/학습10회 평균 | 약300ms warmup, 7×50 replay 중앙값 |
| PWA 모듈 학습 dropout | 생성자 기본값 0 | 0.15, RNG 포함 |
| mask | 모두 true | 약10% false |
| 학습 API | 입력 clone + zero_grad + backward | 고정 입력 + autograd.grad, 전체 입력/파라미터 gradient |
| OPM 추론 residual | 없음 | pair residual 포함 |
| OPM 추론 경로 | Anthropic prologue + cuBLAS + 자체 epilogue, 가중치 pack 재사용, LN stats 없음 | 학습용 native forward 재사용, 호출별 pack/변환, LN stats 생성 |
| OPM 학습 저장 | save_o=1 | save_o=1 |

**클로드 모듈 표의 PWA 1.561ms는 dropout0이다.** 같은 세션이 별도로 측정한 실제 모델 스텝에는 dropout0.15가 들어갔다. 두 결과를 혼동하면 안 된다.

### 기존 엔진 열 자체가 달라진 정도

| 모듈/모드 | Claude `own`(ms) | 현재 v1.0.0 + compile/graph(ms) |
|---|---:|---:|
| PWA 추론 | 1.526 | 1.001 |
| PWA 학습 | 4.161 | 2.955 |
| OPM 추론 | 2.177 | 0.953 |
| OPM 학습 | 5.858 | 2.726 |

따라서 Claude의 2.67–3.71배와 현재 비교표의 배율은 같은 분모가 아니다. 차이 전체를 compile 하나의 효과로 정량 귀속하지도 않는다(코드 revision·dropout·mask·반복·GPU 조건이 다름).

## OPM 추론에서 실제로 바뀐 배선

현재 `OuterProductMean.forward`는 no_grad에서도 먼저 `opm_train.outer_product_mean`으로 들어간다. 추론용 `anthropic_msa.outer_product_mean`은 뒤에 있어 기본 경로에서 도달하지 않는다. 계산 핵심은 여전히 prologue → cuBLAS → 같은 CUDA epilogue지만 전처리와 LN 통계 저장 조건이 다르다.

이것은 O(302MB)를 새로 한 번 더 복사한다는 뜻은 아니다. 두 경로 모두 GEMM에서 O를 생성하며, 학습의 save_o는 해당 결과를 유지하는 정책이다. 차이를 판단하려면 residual과 경로를 분리해 측정해야 한다. [동일 GPU 비교 코드](opm_paths.py).

## 원본 스크립트 재현 (job16134, node01 H100)

과거 개발 checkout의 원본 스크립트를 그대로 다시 실행했다. 기록의 최종 수치가 거의 그대로 재현된다. 빌드 캐시는 원본 개발 캐시를 사용했으며 첫 컴파일/튜닝은 측정 밖이다.

| 모듈/모드 | 당시 own | 이번 재현 own | 당시 fused | 이번 재현 fused |
|---|---:|---:|---:|---:|
| PWA 추론 | 1.526 | 1.529 | 0.411 | 0.412 |
| PWA 학습 | 4.161 | 4.154 | 1.561 | 1.562 |
| OPM 추론 | 2.177 | 2.182 | 0.633 | 0.636 |
| OPM 학습 | 5.858 | 5.858 | 2.028 | 2.031 |

단위 ms. [추론 로그](old-infer.log), [PWA 학습 로그](old-pwa.log), [OPM 학습 로그](old-opm.log).

## 동일 GPU/입력에서 OPM 차이 분리 (job16134)

B1/S1024/L384, 약10% false mask, BF16, 9×100 CUDA graph replay를 순환 측정한 중앙값. 원본 eager 벤치와 달리 지속 replay 부하이며, 원본 수치를 대체하지 않는다. [원본 결과](opm-paths.json).

| 경로 | residual 없음 (ms) | residual 포함 (ms) | residual 증가 |
|---|---:|---:|---:|
| Claude 추론 전용 경로 | 0.672 | 0.709 | 37.0µs |
| 현재 학습 forward 재사용 경로 | 0.693 | 0.728 | 34.9µs |

- 같은 residual 조건에서 현재 경로가 약19–21µs 느리다. stats 저장·pack/변환을 포함한 경로 전체 차이이며 각각의 비용을 개별 귀속한 수치는 아니다.
- 현재 모듈을 compile까지 적용하면 residual 포함 **0.721ms**였다. 이전 표 0.703ms와는 별도 GPU/반복 부하 측정으로, 절대시간을 그대로 합쳐 분해하면 안 된다.
- 두 raw 출력과 residual 포함 출력 모두 상대 L2 오차 0이다.
- 과거의 residual 없는 추론 전용 값을 현재의 residual 포함 학습 forward 경로 값과 같은 조건으로 비교했던 것이 차이의 핵심이다. 즉 PWA 전체 CUDA 구현 누락이 아니라, baseline/측정 조건과 OPM 추론 배선 차이였다.
