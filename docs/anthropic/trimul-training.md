# Anthropic를 계승한 TriMul 학습 — 기존 융합·저장 정책 유지

2026-09-19 · node02 H100 · C128 / BF16 / B1 / dropout 25% / residual 포함

## 현재 구현

**이전 학습의 융합 경계와 저장 텐서를 유지하고, 입력 projection과 출력 F567의
커널 구현을 Anthropic 원본 CUDA에서 파생한 구현으로 바꿨다.**
원본 전체 forward + 느린 PyTorch 재계산 backward는 별도의 정확도 기준선으로 남긴다.
현재 경로는 그 재계산 backward를 쓰지 않는다.

Anthropic의 inference 결과가 우리의 이전 개발보다 뛰어났음을 인정하며 계승한다.
단순히 아이디어만 참고한 것이 아니라 원본 `k1_body` / `k3_body`를 직접 변형했다.
출처: [uplifting-biomolecular-modeling · f4f62fa](https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e), Apache-2.0.
Vendored 원본은 수정하지 않았고 파생 CUDA 파일에 출처와 변경 내용을 명시했다.

## 배선과 저장 — 이전과 같은 경계

| 단계 | 실행 구현 | 저장값 / 변경 |
|---|---|---|
| F1: 입력 LN | 기존 Triton | x_n, mean/rstd 유지 |
| F2: left/right projection + gate + mask | **Anthropic K1 파생 CUDA** | a/b 및 기존 interleaved preact 저장 |
| F3: outgoing/incoming contraction | 기존 cuBLAS | 최종 X buffer로 직접 출력, 동일 |
| F4: 출력 LN | 기존 Triton | normalized X, mean/rstd 유지 |
| F567: projection + output gate + dropout + residual | **Anthropic K3 파생 CUDA** | projection, sigmoid gate 저장 유지 |
| Backward 전체 | 기존 Triton/cuBLAS | 같은 저장값을 소비, GEMM 재계산 추가 없음 |

이전 K3 시제품은 F4+F567을 묶었다. **이번 구현은 F4와 F567을 다시 분리**했다.
K1 안의 입력 LN과 K3 안의 LN들은 수행하지 않고 기존 별도 LN의 결과를 읽는다.
따라서 기존 두 LN과 같은 반올림을 유지하며 입력 normalized 값을 공유한다.

원본에서 유지한 내부 구조는 TMA load, WGMMA, persistent producer/consumer,
weight ring, MMA/epilogue overlap 및 shared staging이다.
K1은 추가 preact를 gate/projection 각각의 TMA map으로 저장하되 기존 interleaved
`[g0,p0,g1,p1,...] × M` layout에 직접 쓴다. 변환용 큰 cat/copy는 없다.
F567은 이전에 개발한 dropout/residual 및 projection/gate 저장 확장을 사용한다.
이전 backward의 BF16 반올림 규약도 유지한다.

## 사용

```python
from miniworld_engine import settings
from miniworld_engine.modules.triangle_multiplication import BidirectionalTriangleMultiplication

# CUDA를 쓰므로 global Triton-only 강제 설정과는 함께 사용하지 않는다.
settings.configure(engine_backend="auto", trimul_sm90_kernels=frozenset())
model = BidirectionalTriangleMultiplication(
    128, implementation="anthropic", anthropic_row="training_saved", p_drop=0.25,
).cuda().bfloat16().train()
y = model(pair, residue_mask)
y.float().square().mean().backward()
```

단방향 `TriangleMultiplication`도 같은 row로 outgoing/incoming을 지원한다.
검증 범위: H100, B1, C128, 단방향 H128 / 양방향 H256, BF16 activation,
L이 8의 배수. L64/72/384/768을 검사했다. 지원 범위 밖은 명시적으로 거절한다.
JIT 빌드에는 nvcc 및 vendored 원본 header가 필요하다. `native_rebuilt` cubin은 이 경로에 필요하지 않다.
기존 전역 기본 dispatch는 변경하지 않았다. 실행 중이던 MiniWorld 모델 학습 잡도 변경하지 않았다.

## 같은 저장 정책에서의 성능

같은 프로세스에서 순서를 교대하며 CUDA Graph replay를 8라운드 측정한 중앙값.
단위 ms. 양방향 C128/H256, BF16, B1, 같은 mask/dropout scale, dropout 25%와 residual 포함.
Weight packing 포함, RNG 생성과 optimizer 제외. 아래 비교 기준은 **이전 Triton 학습 경로**다.
Anthropic 원본 inference를 이겼다는 의미가 아니다.

| L | 범위 | 이전 Triton | Anthropic 파생 구현 | 시간 감소 |
|---:|---|---:|---:|---:|
| 384 | forward | 0.548 | 0.508 | 7.3% |
| 384 | forward+backward | 1.633 | 1.588 | 2.8% |
| 768 | forward | 2.124 | 1.859 | 12.5% |
| 768 | forward+backward | 6.419 | 6.175 | 3.8% |

초기 커널 호출 비교에서 입력 projection(pack 포함)은 L384 253.0→202.8µs,
L768 975.7→756.0µs였다. F567은 L384 106.5→108.6µs로 약간 느렸고,
L768 401.1→380.4µs로 빨랐다. 작은 출력 커널의 회귀를 숨기지 않는다.
F567을 더 최적화할 여지가 남아 있으며 전체 학습 15% 개선을 달성한 것은 아니다.

주의: Triton cache miss에는 기존 실험과 같이 24개 heuristic 후보를 사용했다.
새 CUDA는 H256에서 front/F567 각각 6개 초기 후보를 비교했다. 전수 튜닝 결과는 아니다.
유효 공간은 front 58개, F567 H128 30개/H256 12개다. F567에는 LN이 없으므로
불필요한 LNSERIAL 축은 제거했다. 일부 후보는 spill이 있었으며 선택한 기본 설정은 spill-free다.

## 검증과 NCU

- **14개 GPU 테스트 통과**: 단방향/양방향, L64/72, 모든 gradient, zero mask/dropout/projection,
  모듈 SGD/eval, 여러 forward, fullgraph compile + CUDA Graph에서 weight/dropout 갱신.
- `saved_tensors_hooks`로 기존과 저장 목록의 shape/dtype/순서가 일치함을 검사했다.
  L384/768에서도 동일 저장 목록 및 모든 gradient를 확인했다.
- L384/768 전체 출력 및 GEMM weight gradient는 이번 입력에서 기존 Triton과 비트 일치.
  전체 gradient 상대 L2 차이 최대 1.4e-6 수준(atomic norm reduction 포함).
- Compute Sanitizer: memcheck 0 errors, synccheck 0 errors, racecheck 0 hazards.
- 기본 H128/H256 cubin에서 TMA load/store와 HGMMA 확인. local memory 및 LDL/STL 0.
- NCU L768/H256: front 754.8µs / 2.59TB/s, F567 396.6µs / 2.61TB/s.
  DRAM active 약 77.1% / 78.0%. 이 수치만으로 roofline 도달을 단정하지 않는다.

전체 MiniWorld 모델 수렴 검증, 다른 폭/배치/GPU, engine 전체 autotune/cache 통합은 남아 있다.

## 파일과 기록

- `src/miniworld_engine/kernels/trimul_inproj/cuda/anthropic_saved_front.cu`
- `src/miniworld_engine/kernels/trimul_inproj/cuda/anthropic_saved_output.cu`
- `src/miniworld_engine/kernels/trimul_inproj/cuda/anthropic_saved.py`
- `tests/numerics/test_trimul_anthropic_saved_gpu.py`
- [실측·검증·provenance 기록](../records/trimul/anthropic-saved-training-20260919/)
- [이전 원본 전체 forward + 재계산 backward 기준선](../records/trimul/anthropic-trimul-training-20260919/native-recompute-baseline-history.md)
- [이전 K3 시제품](../records/trimul/anthropic-trimul-training-20260919/k3-prototype-history.md)

개발 시각화는 기존 `trimul.html#training-progress`에 유지한다.
루트의 `scripts/update_anthropic_training_dashboard.py`로 실측 JSON에서 갱신한다.
