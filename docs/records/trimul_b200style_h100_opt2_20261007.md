# TriMul B200-style H100 추가 융합 탐색 — 2026-10-07

이 탐색에서는 앞선 최적화 경로보다 빠른 후보를 얻지 못했다. 새 후보는 모두 실험용이며, 기본 dispatch와 기존 opt-in 경로를 유지했다. 기존 경로의 8개 핵심 소스는 이전 snapshot-final과 바이트 단위로 같음을 확인했다.

## 비교 조건

- GPU: Slurm normal_h100의 NVIDIA H100 80GB HBM3, 한 작업당 GPU 1개.
- B=1, D=512, L=768, bidirectional, BF16 mixed, mask_prob=0.1, 학습 dropout=0.25 / 추론 dropout=0.
- 기준은 **이전 최적화된 B200-style opt-in 경로**: K1 SS `(128,4,128,0,1)`, K3 `(256,3)`. 프로젝트 기본 커널 대비 수치가 아니다.
- 변경하지 않은 공식 benchmarks/runners/bench.py의 torch.compile + manual CUDA Graph 전체 모듈 경로를 캡처하여 90번 AB/BA 교차 측정했다. 학습은 전체 F+B이다.
- 각 측정 전에 D512/L128 독립 FP32 reference 출력·모든 gradient pilot을 통과했다. 각 baseline/candidate의 입력과 파라미터가 같음을 검사했다.
- 아래 각 행은 같은 작업 안의 paired median이다. 작업 간 절대 시간은 직접 비교하지 않는다. eager profile은 실제 dispatch 확인용이며 최종 timing에 사용하지 않았다.

## 전체 학습 F+B

배속 = 기준 / 후보. 1보다 작으면 후보가 느리다.

| 후보 | Job / cell | 기준 ms | 후보 ms | 배속 |
|---|---|---:|---:|---:|
| 계산/epilogue 별도 warpgroup | sweep-overlap-27511/c0 | 32.47683 | 40.43350 | 0.80322× |
| WGMMA commit grouping | sweep-group-27517/c0 | 32.92598 | 36.22755 | 0.90887× |
| WGMMA commit grouping | sweep-group-27517/c1 | 33.50650 | 33.95728 | 0.98673× |
| WGMMA commit grouping | sweep-group-27517/c2 | 33.10810 | 34.41059 | 0.96215× |
| 2-CTA weight multicast | sweep-multicast-27521/c0 | 32.52678 | 39.18931 | 0.82999× |
| SM당 2 CTA + 8KB stage | sweep-dual-27525/c0 | 32.14071 | 38.21603 | 0.84103× |
| 64-token 깊은 입력 pipeline | sweep-deep-27526/c0 | 32.60269 | 35.64707 | 0.91460× |
| 64-token 깊은 입력 pipeline | sweep-deep-27526/c1 | 32.98310 | 34.21123 | 0.96410× |
| 64-token 깊은 입력 pipeline | sweep-deep-27526/c2 | 33.39014 | 33.88589 | 0.98537× |
| 64-token 깊은 입력 pipeline | sweep-deep-27526/c3 | 33.15027 | 34.41114 | 0.96336× |
| 깊은 출력 WGMMA pipeline | sweep-output-27528/c0 | 33.12349 | 33.81811 | 0.97946× |
| 깊은 출력 WGMMA pipeline | sweep-output-27528/c1 | 33.38992 | 34.17754 | 0.97696× |
| 깊은 출력 WGMMA pipeline | sweep-output-27528/c2 | 33.49786 | 34.19318 | 0.97966× |
| 깊은 출력 WGMMA pipeline | sweep-output-27528/c3 | 33.26077 | 34.70346 | 0.95843× |

## 추론

| 후보 | Job / cell | 기준 ms | 후보 ms | 배속 |
|---|---|---:|---:|---:|---:|
| group | 27533/c0 | 8.69699 | 8.96646 | 0.96995× |
| dual | 27533/c1 | 8.42618 | 11.70016 | 0.72018× |
| multicast | 27533/c2 | 8.35523 | 12.12333 | 0.68919× |
| overlap | 27533/c3 | 8.34787 | 12.96954 | 0.64365× |

## 검증과 중단된 후보

- separate-epilogue BN128: 49/49 통과(job 27508). 입력 BN64 변형은 strict pilot이 종료되지 않아 job 27511을 취소했다. 원인은 미확정이며 spill로 단정하지 않는다. BN64 컴파일 검사에서 spill은 관찰되지 않았다.
- grouped default GROUP=2/WAIT=1: 49/49 통과(job 27512). 다른 grouping 후보는 해당 D512 pilot 범위만 검증했다.
- multicast BN128: 49/49 통과(job 27520). memcheck 25 cases, racecheck 5 scenarios, synccheck 25 cases가 모두 errors/hazards/warnings 0(job 27522). BN256 변형은 unspecified launch failure로 실패했으며, 유효한 timing도 없다.
- dual: 49/49 통과(job 27523). 타이밍 job 27525에서 cuOccupancyMaxActiveBlocksPerMultiprocessor로 SM당 2 CTA가 가능함을 확인하는 runtime guard를 통과했다.
- deep input/output pipeline은 개별 strict pilot과 공식 runner 검증만 통과했다. full49 또는 sanitizer 전체를 통과했다고 주장하지 않는다.
- 어느 새 후보도 학습 배속을 얻지 못해 추가 promotion gate를 실행하거나 dispatch에 반영하지 않았다. overlap/dual/group/deep 후보에 대한 전체 sanitizer 통과 주장은 하지 않는다.
- 출력 후보 준비 중 source 보정 후 job 27527을 취소하고 고정 snapshot으로 27528을 새로 실행했다. 27527의 결과는 채택하지 않았다.
- 초기 추론 job 27530에서 overlap에 잘못된 길이의 config를 넘겨 Python unpack 오류가 발생했다. scratch driver를 고쳐 job 27533에서 네 후보 모두 다시 측정했다. 표에는 27533 결과만 사용했다.

## 구현·증거

- 실험용 구현: h100_b200style_{overlap,group,multicast,dual,output}.py 및 h100_sources/b200style/{k1overlap,k1group,k1multicast,k1dual,k3deep}.cu. 기존 selector에서는 호출하지 않는다.
- 원본 90-pair samples, 실제 커널 이름·dispatch profile, strict test/sanitizer logs, 재현 scratch driver, frozen-source hash와 공식 runner artifacts: [b200style_opt2_20261007](../../runs/benchmarks-h100/benchmarks/modules/triangle_multiplication/results/h100/b200style_opt2_20261007/).
- 재현 snapshot과 일회성 실행 도구: /home/psk6950/miniworld-engine-scratch/trimul-b200style-opt2-20261007/.
- 이전 원본 B200 port 대비 개선 및 프로젝트 기본 커널과의 차이는 [이전 기록](trimul_b200style_h100_opt_20261007.md)에 남아 있다. 이번의 실패 결과로 그 비교 수치를 갱신하지 않는다.
- 낮아진 명목 weight traffic이나 높아진 CTA occupancy만으로 실제 병목 개선을 입증하지 못했다. 이번에는 hardware counter 또는 SOL90을 측정하지 않았다.
