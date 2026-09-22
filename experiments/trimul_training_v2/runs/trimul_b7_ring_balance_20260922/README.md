# B7–B12 단일 CUDA 커널: 16 KiB ring pipeline

L384 / C128 / projection width256 / H100 / 단일 cooperative launch.
동일 프로세스에서 입력을 공유하고 순서를 교대로 바꾸며 5×160회 측정.
각 커널의 초기화·최종 reduction을 포함한 B7–B12 범위다.

## 결과

| 구현 | µs |
| --- | ---: |
| 현재 분리형 | 394.592 |
| 재개 출발점: 약 400 µs 통합형 | 414.848 |
| 직전 개선 통합형: 약 388 µs | 395.328 |
| 이번 선택: 16 KiB × 4, producer 64 regs | 373.408 |
| 비교 후보: producer 48 regs | 376.672 |

직전 통합형보다 5.54%, 현재 분리형보다 5.37% 시간 단축.
252 µs 및 SoL90 미달/미입증. 전체 학습이나 L768 결과로 외삽하지 않는다.
최종 반복 측정에서 producer 64 regs가 48 regs보다 소폭 빨랐다. 선택값은 64 regs다.

## 구현

- source CTA가 projection/gate를 재계산하고 dp/dg를 한 번 생성한다.
- 같은 shared dp/dg를 dW WGMMA가 소비하는 동안 global ring으로 bulk store한다.
- ring을 left dg / left dp / right dg / right dp 네 개 32 KiB plane으로 배치했다.
- dX CTA는 16 KiB씩 네 shared slot으로 읽으며 다음 전송과 현재 WGMMA를 겹친다.
- ring-load warp와 weight-load warp를 분리한다. 가중치는 16 KiB 두 버퍼다.
- 기존 GEMM 합산 및 BF16 반올림 순서를 유지한다. LN 미분과 residual도 같은 커널이다.
- 끝에서 cooperative grid sync 후 dW와 LN 파라미터 partial을 합산한다.
- global ring 크기와 논리적 전송량은 그대로다. global memory ring은 L2 재사용을 유도하며 shared-only 전송이 아니다.
- 8 KiB는 전송/동기화 증가를 상쇄하지 못했다. 3D TMA store·early gate·CTA/레지스터 배분도 비교했다.

## 검증과 한계

5개 입력/가중치/마스크 변형에서 7개 출력의 기존 허용오차 통과.
scratch poison, eager/graph 일치, counter/flag 재사용 검사 통과. memcheck·racecheck 오류 0건.
원본: [confirmation.json](confirmation.json), [selected.json](selected.json).
컴파일러 로그 및 cubin은 build/에 있다. 현재 후보는 spill 0 bytes다.
NCU Tensor activity는 SoL 수치가 아니며 단독 프로파일 latency를 graph latency와 섞지 않는다.
Anthropic 유래 Apache-2.0 TMA/WGMMA primitives를 바탕으로 한 학습 확장이다.
실험 후보이며 production dispatch는 변경하지 않았다.

## 선택 후보 NCU

별도 프로파일 367.104 us, HBM read 303.756 MB / write 56.802 MB.
Tensor active 34.709%, local spill traffic 0. Tensor active는 SoL이 아니다.
[원본 지표](profile-summary-p64.json), [NCU report](ncu-L384-mode4-p64.ncu-rep).
