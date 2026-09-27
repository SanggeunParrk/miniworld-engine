# Anthropic 원본 TriMul forward의 학습 연결

2026-09-19 · node02 H100 · C128 / BF16 / B1 / dropout 25% / residual 포함

## 현재 상태와 출처

Anthropic 원본 **K1 → cuBLAS → K3 전체**를 autograd 학습 경로에 연결했다.
입력 front가 Triton인 이전 K3 파생 시제품과 구분한다. 원본 native CUDA 소스와
빌드된 K1/K3는 수정하지 않았다. 원본에는 backward가 없으므로 PyTorch 재계산과
cuBLAS contraction 미분을 추가했다. 이것은 정확도 기준선이며 빠른 backward 완성이 아니다.

우리가 개발했던 inference 커널보다 Anthropic의 결과가 뛰어났음을 인정하고 계승한다.
성능 목표는 같은 연산·dtype·mask·residual 조건에서 **Anthropic 원본 대비 개선**이다.
기존 MiniWorld Triton 대비 가속을 이 목표의 달성 근거로 삼지 않는다.
이번에는 원본 자체를 더 빠르게 바꾼 것이 아니라 학습에 실제 넣었다.

원본: [Anthropic uplifting-biomolecular-modeling](https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e), Apache-2.0.
고정 revision `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`의 native 소스를 CUDA 12.9로 재빌드한 `native_rebuilt`를 쓴다.

## 실제 배선

| 단계 | 실행 구현 | backward에 필요한 값 |
|---|---|---|
| 입력 LN + left/right projection/gate + mask | **원본 K1**, TMA/WGMMA | 실제 native a/b 저장 |
| Triangle contraction | cuBLAS, 단방향 1회/양방향 2회 | 실제 native X 저장 |
| 출력 LN + projection + 입력 LN 재계산 + output gate | **원본 K3**, TMA/WGMMA | 별도 activation은 backward에서 재계산 |
| row dropout + residual | 외부 PyTorch 연산 | PyTorch autograd |
| 출력 surround 미분 | PyTorch FP32 재계산 | 실제 native X에서 미분 |
| contraction 미분 | cuBLAS | 실제 native a/b 소비 |
| 입력 surround 미분 | PyTorch FP32 재계산 | 입력/10개 weight gradient 반환 |

양방향은 H256 K1 출력을 H128씩 나눠 outgoing/incoming을 계산하고,
하나의 X buffer에 직접 쓴 뒤 **H256 전체에 한 번의 LN_out**과 원본 K3를 적용한다.
독립적인 단방향 모듈 두 개를 더하면 다른 함수가 되므로 그렇게 연결하지 않았다.
이 양방향 조립은 engine 코드이고, upstream의 완성된 양방향 모듈이라고 주장하지 않는다.

매 forward에서 live weights를 새로 pack한다. inference weight pointer cache를 그대로 재사용하지 않는다.
저장하는 a/b와 X는 각 호출에서 별도로 할당하므로 여러 forward의 backward 저장값이 서로 덮어써지지 않는다.
기본 dispatch는 변경하지 않았고 명시적 `implementation="anthropic"` 옵션으로 선택한다.

## 사용법과 제한

```python
import torch
from miniworld_engine.modules.triangle_multiplication import BidirectionalTriangleMultiplication

torch.backends.cuda.matmul.allow_tf32 = False
model = BidirectionalTriangleMultiplication(
    128, implementation="anthropic", anthropic_row="native_rebuilt", p_drop=0.25,
).cuda().bfloat16().train()
y = model(pair, residue_mask)
y.float().square().mean().backward()
```

단방향 `TriangleMultiplication`도 같은 옵션으로 outgoing/incoming을 지원한다.
`TRIMUL_NATIVE_BUILD_DIR`는 원본 재빌드 payload의 build/를 가리켜야 하며 sibling python/이 필요하다.
이 실행 환경에서는 `runs/anthropic_adoption_20260919/env.sh`를 사용했다.

현재 검증 범위는 SM90, BF16 pair [1,N,N,C128], H128/256, BF16/FP32 parameters다.
L64·65·384·768을 검증했다. 다른 폭은 upstream cubin/table이 없으면 명시적으로 거절한다.
1차 gradient만 지원한다. Eager와 CUDA Graph는 검증했고 **torch.compile fullgraph는 아직 미지원**이다.
명시적 graph break를 둔다. TF32 비활성화는 검증용 FP32 surround 수식의 정확도 조건이다.
전체 MiniWorld 모델 학습 실행이나 모델 수렴 검증은 아직 하지 않았다.

## 기준선 실측

CUDA Graph replay 15회 중앙값, 단위 ms. 동일 고정 dropout scale/mask, live weight pack 포함.
RNG 생성 및 optimizer 비용 제외. 아래 backward는 최적화 전 **검증용 재계산**이어서 느리다.

| 경로 | L | 학습 forward | forward + 검증용 backward |
|---|---:|---:|---:|
| outgoing | 384 | 0.258 | 8.837 |
| bidirectional | 384 | 0.386 | 13.954 |
| outgoing | 768 | 0.915 | 50.803 |
| bidirectional | 768 | 1.420 | 89.458 |

참고용 원본 단방향 update-only, cached weight pack, dropout/residual 제외 시간은 L384 0.155ms,
L768 0.603ms였다. 작업량이 달라 위 학습 forward와 가속률 비교에 쓰지 않는다.
원본에는 학습 backward가 없으므로 위 전체 학습 시간을 Anthropic 학습 성능이라고 부르지 않는다.

## 검증

- GPU pytest **14개 통과**: 단방향/양방향, padding N65, dropout/residual, zero mask,
  실제 모듈 API, SGD 갱신, 두 forward를 쌓은 뒤 backward, CUDA Graph 중 weight/dropout 변경.
- 원본 단방향 update와 비트 단위 출력 일치: outgoing L64/65/384/768, incoming L64/65.
- 별도 row-major PyTorch 수식 대비 L384/768 outgoing/양방향 전체 gradient 확인.
  출력 상대 L2 최대 0.052%, 입력과 10개 parameter gradient 최대 0.290%.
  reduction/근사 sigmoid 및 BF16 gradient 반올림 때문에 gradient 비트 일치를 주장하지 않는다.
- GPU trace에서 원본 `tmn_k1_*` 및 `tmn_k3_*_u` 호출 확인. 이전 Triton front로 우회하지 않는다.

원자료는 [records](records/anthropic-trimul-training-20260919/)의 `native-training-*.json`과
`native-bench.log`, 구현은 `src/miniworld_engine/integrations/anthropic_training.py`,
검증은 `tests/numerics/test_trimul_anthropic_native_training_gpu.py`다.

## 다음 순서

1. 이 경로를 원본 forward 및 학습 정확도 기준으로 고정한다.
2. 원본 FP32 gate/GEMM 반올림 규약을 유지하며 빠른 backward를 연결·개발한다.
3. live weight packing, dropout/residual, compile/cache 통합 비용을 줄인다.
4. 원본 자체 최적화는 세부 NCU stall 분석 및 동일 작업량 A/B로 판정한다.

개발 시각화는 기존 사이트 `trimul.html#training-progress`에 유지한다.
루트 `scripts/update_anthropic_training_dashboard.py`가 실측 JSON에서 HTML을 갱신한다.

[이전 K3 파생 시제품 기록](records/anthropic-trimul-training-20260919/k3-prototype-history.md)은
과거의 별도 실험이며, 원본 전체 forward를 연결한 현재 경로와 섞어 해석하지 않는다.
