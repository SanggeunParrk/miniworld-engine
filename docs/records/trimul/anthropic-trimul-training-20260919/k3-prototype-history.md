# Anthropic를 계승한 TriMul 학습 커널 — 첫 구현

2026-09-19 · node02 H100 · C128 / BF16 / B1 / dropout 25% / residual 포함

## 성능 목표 정정

목표는 같은 작업과 조건에서 Anthropic 원본보다 빠른 forward를 만드는 것이다.
기존 MiniWorld Triton 대비 가속은 주 성능 목표의 달성 근거가 아니다.
이번 결과는 학습 확장 시제품의 기능·회귀 검증으로 분류한다. Anthropic 대비 개선은 아직 입증하지 않았다.
다음은 원본 K3의 세부 warp stall/source counter 수집과 원본 대비 A/B 실험이다.
기존 NCU에는 세부 stall 원인 지표가 없으며 64% HBM 이용률만으로 추가 개선폭을 확정할 수 없다.

개발 현황의 기본 시각화는 기존 웹 현황판 `trimul.html#training-progress`에 유지한다.
`python scripts/update_anthropic_training_dashboard.py`는 실험 JSON으로 HTML을 갱신한다.
원본 분석 페이지를 다시 생성한 뒤에도 이 갱신 스크립트를 실행하고 같은 비공개 사이트에 배포한다.

## 구현 범위

Anthropic uplifting-biomolecular-modeling의 Apache-2.0 native K3를 직접 차용·확장했다.
원본 revision: `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`.
원본 vendored 파일은 변경하지 않았다. 새 CUDA 파일 헤더에 출처와 변경 내용을 명시했다.

- **새 CUDA:** 출력 LayerNorm + projection + output gate + dropout + residual + backward용 저장값.
- **계승:** register LN→WGMMA RS, persistent producer/consumer, TMA, swizzled shared staging, accumulator overlap.
- **새 설계:** 정규화 입력을 읽은 shared 버퍼에 residual을 TMA로 다시 읽는 2-phase 파이프라인. 버퍼를 추가하지 않는다.
- **학습용 저장:** normalized contraction, mean/rstd, projection, gate. 큰 저장값은 TMA store.
- **기존 경로 유지:** 입력 LN/front, cuBLAS contraction, backward 전체는 기존 Triton/cuBLAS.
- 현재 gate 입력은 기존 front가 만든 normalized input이다. Anthropic 전체 forward의 LN 재계산 정책까지 이식한 것은 아니다.

단방향 outgoing/incoming과 양방향 모듈에 명시적 실험 옵션으로 연결했다.

```python
BidirectionalTriangleMultiplication(
    128, implementation="triton",
    training_output_backend="anthropic_cuda", p_drop=0.25,
)
```

추론은 기존 경로다. 지원은 C128, H128/256, BF16 activations/weights, FP32 LN affine,
B1이다. 현재 전체 학습 연결은 L이 8의 배수인 경우를 지원한다. 낮은 수준 K3는 N65/Np72 padding도 검증했다.

## 성능

동일 프로세스에서 순서를 교대하며 CUDA Graph replay를 측정한 중앙값. 학습은 forward+backward이며 optimizer/RNG 생성은 포함하지 않는다.
양쪽에 같은 외부 dropout scale과 mask를 공급했다. 아래는 새 경로의 기본 config 결과다.

| 양방향 구간 | L | Triton | 새 CUDA 출력 경로 | 가속 |
|---|---:|---:|---:|---:|
| 출력 LN+F567 | 384 | 163.9 µs | 142.0 µs | 1.15× |
| 출력 LN+F567 | 768 | 617.1 µs | 534.3 µs | 1.15× |
| 전체 training forward | 384 | 551.4 µs | 533.1 µs | 1.03× |
| 전체 training forward | 768 | 2115.8 µs | 2024.5 µs | 1.05× |
| 전체 forward+backward | 384 | 1631.8 µs | 1619.6 µs | 1.007× |
| 전체 forward+backward | 768 | 6422.5 µs | 6328.0 µs | 1.015× |

**제한:** 일부 Triton cache miss에 24개 heuristic 후보 튜닝을 사용했다. 완전히 튜닝된 모든 Triton 설정 대비 우위를 입증한 결과는 아니다.
새 CUDA도 각 H에서 6개 초기 후보만 비교했다. 60개(H128), 24개(H256)의 유효 config 공간을 제공하지만 전체 탐색은 미완료다.
전체 학습의 0.7~1.5% 감소는 작고 반복 측정 변동도 있으므로 큰 전체 가속을 주장하지 않는다.

## NCU와 바이너리

L768/H256, 별도의 NCU replay. 첫 버전은 fragment별 global 저장과 global residual 읽기로 Triton보다 느렸다.
최종 버전은 저장과 residual 입력을 TMA로 옮겼다.

| 지표 | 첫 버전 | 최종 버전 |
|---|---:|---:|
| K3 시간 | 862.8 µs | 536.3 µs |
| HBM 처리량 | 1.61 TB/s | 2.54 TB/s |
| DRAM active cycles / peak | 47.9% | 75.8% |
| eligible warps / scheduler cycle | 0.272 | 0.409 |

실제 cubin에서 UTMALDG, UTMASTG, HGMMA를 확인했다. 두 기본 config 모두 local memory 0, LDL/STL 0, ptxas spill 0이다.
일부 실험 config는 spill이 있었으므로 모든 config가 spill-free라는 뜻은 아니다.
NCU 시간과 Graph 시간은 구분한다. 아직 roofline 도달이나 세계 최고 성능을 주장하지 않는다.

## 검증 및 남은 작업

12개 GPU 테스트 통과: PyTorch 독립 수식·모든 gradient, 기존 Triton 비교, zero mask/dropout/projection,
모듈 옵션, compile+CUDA Graph, weight/dropout 변경 후 replay, N65/Np72 경계 및 split-N.
Compute Sanitizer memcheck·synccheck는 0 errors, racecheck는 0 hazards다.
L384/L768 양방향의 입력 및 10개 파라미터 gradient도 기존 Triton과 비교했다.

다음 개발 순서:
1. Anthropic K1의 입력 LN+front 융합을 학습용 저장/재계산 정책과 함께 이식.
2. backward의 gate/projection dgrad 및 LayerNorm 연계 구간을 CUDA/TMA/WGMMA로 개발.
3. 넓은 config 탐색과 engine autotune/cache 체계 통합.
4. 완전히 튜닝된 baseline, 실제 모듈 harness와 전체 모델에서 재측정.

현재 결과는 **첫 K3 학습 확장 완료**이며 전체 CUDA TriMul 학습 완성이 아니다.
개발 GPU 2장은 반환했고 기존 MiniWorld 학습 잡은 변경하지 않았다.

## 증거

- [실험 원자료](records/anthropic-trimul-training-20260919/)
- CUDA: `src/miniworld_engine/kernels/trimul_inproj/cuda/anthropic_k3_training.cu`
- launcher/config: `src/miniworld_engine/kernels/trimul_inproj/cuda/anthropic_training.py`
- tests: `tests/numerics/test_trimul_anthropic_training_gpu.py`
- 원본: https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e
