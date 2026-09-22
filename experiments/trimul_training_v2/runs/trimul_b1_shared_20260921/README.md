# B1–B4: shared tile recomputation, saved input x_n

2026-09-21 · **개발 경로에 연결 완료. Production 승격 아님.**
Anthropic v5 TMA/WGMMA/LN primitives와 reference 연산 순서를 계승하고, Miniworld 학습 미분·공유 스케줄을 추가했다.
비교 대상은 **직전 Anthropic 파생 saved-x_n 학습 경로 `split_xn_pc1`**이다. 원본 Anthropic 추론이나 cuEquivariance와의 새 비교가 아니다.

## 같은 실행에서 직접 측정

| L | 범위 | 직전 saved-x_n 기준 | 새 공유 계산 커널 | 속도 배율 | 지연 감소 |
|---|---|---|---|---|---|
| 384 | B1–B4 | 316.384 μs | 262.336 μs | 1.206× | −17.08% |
| 384 | 전체 BWD | 1021.584 μs | 966.240 μs | 1.057× | −5.42% |
| 384 | 전체 FWD+BWD | 1311.136 μs | 1259.424 μs | 1.041× | −3.94% |
| 768 | B1–B4 | 1227.776 μs | 904.112 μs | 1.358× | −26.36% |
| 768 | 전체 BWD | 4143.952 μs | 3791.232 μs | 1.093× | −8.51% |
| 768 | 전체 FWD+BWD | 5387.984 μs | 5017.744 μs | 1.074× | −6.87% |

node02 H100, BF16 C128 / 양방향 H256, L384/768, mask/dropout25%/residual.
경로별 CUDA graph 600회(200 ×3 블록) 교대 측정 중앙값. Live packing·모든11개 gradient 포함; RNG·optimizer·CPU dispatch·compile 제외.
전체 FWD+BWD는 직접 측정했다. 독립 구간 중앙값을 더한 결과가 아니다. GPU clocks를 고정하지 않았고 다른 GPU의 기존 학습 작업은 계속 동작했다.

## 선택한 구조: 같은 커널 안의 두 순차 단계

- Forward K3가 입력 affine BF16 `x_n`을 저장. Forward/B5–B6/B7은 기존과 동일.
- **Phase A, 132 CTA ×256 threads:** 행64개 타일의 출력 LN·gate·projection을 각각 한 번 계산. dGate와 dProj 생성. 같은 CTA에서 dWproj 부분합, dTri 및 출력 LN 미분 계산.
- dNorm BF16 [64,256]은 shared memory에만 잠시 둔다. LN은 동일한 balanced-tree 수식으로 계산하되 shared raw tile을 다시 읽어 레지스터 수명을 줄였다. Affine loop unroll4.
- **Grid barrier → Phase B:** 이미 B7을 위해 출력해야 하는 dGate와 저장된 x_n을 HBM/L2에서 다시 읽고 dWgate를 계산한다. 추가 dGate 버퍼 복사나 새 activation 버퍼는 없다.
- Phase B의 TMA 로드는 두 버퍼로 다음 타일을 미리 읽고 WGMMA와 겹친다. 별도 로드 전용 warpgroup은 선택하지 않았다.
- 최종 partial reduction까지 CUDA 호출은 1회. 이전의 세 역할 CTA 그룹과 다르며, 동일한132 CTA가 두 단계를 순서대로 수행한다.
- 선택 설정은 `selected-L*.json`, 호출 가능한 전체 개발 경로는 `training.py:Training`.

## 선택의 대가와 제외한 후보

- 완전한 one-pass에서 dWgate·dWproj 누적값을 모두 유지한 첫 버전: static spill stores/loads1472/1488B. L384386.7 μs, L7681431.0 μs로 느려 제외.
- shared dNorm / streaming LN을 추가한 one-pass도 약279/984 μs로 선택한 two-phase보다 느렸다.
- 로드 전용 producer128 threads + consumer256 threads도 구현·검사했다. 안전한32/232 레지스터 분배로 정상 동작하지만 추가 레지스터 제약·spill 때문에 선택하지 않았다.
- 32/240 분배는 SM 전체 용량 이내여도 초기 CTA pool168×384를 초과해 진행하지 않았다. 실험 잡13398만 중단했고 pool static_assert 및 ptxas C7507 거부를 추가했다.
- CTA 수66/96/132, on-chip dNorm 여부, streaming LN, producer 자원 분배, WGMMA pair issue, affine unroll1/2/4/8을 비교했다. 모든 Hopper shape/config의 전역 최적을 증명한 것은 아니다.
- dW partial이 [132,49152] FP3225.952256MB, LN partial [132,512]0.270336MB로 증가. **합계26.222592MB씩 WRITE/READ**한다. 이전5.60MB보다 크다.
- 선택한 cubin에도 정적 spill16B stores/16B loads가 남는다. Spill-free라고 주장하지 않는다.

## NCU 실측

| L | 경로 | DRAM 읽기 | DRAM 쓰기 | DRAM 처리율 / peak | Tensor active / peak | Occupancy |
|---|---|---|---|---|---|---|
| 384 | baseline | 256.3 MB | 119.3 MB | 35.6% | 15.7% | 12.5% |
| 384 | candidate | 244.4 MB | 142.0 MB | 43.6% | 15.4% | 12.5% |
| 768 | baseline | 1577.0 MB | 459.5 MB | 49.7% | 16.1% | 12.5% |
| 768 | candidate | 924.0 MB | 481.7 MB | 46.3% | 17.8% | 12.5% |

각 경로 `--set full`, 38 replay passes, caches/clocks uncontrolled; CUDA profiler range 안의 B1만 측정.
SASS에서 명시적 TMA(`UTMALDG/UTMASTG`)와 WGMMA(`HGMMA`)를 확인했다.
NCU traffic은 위 버퍼 payload와 구분한다. L384는 전체 DRAM read+write가 오히려 늘어도 중복 연산 감소로 빨라졌다. L768은 반복 읽기 감소 효과가 크다.
**1.7× 및 SoL90 목표는 미달.** Tensor active나 DRAM throughput은 roofline 달성률 자체가 아니며, 이 수치만으로 이론적 상한을 주장하지 않는다. 낮은 occupancy·barrier/long-scoreboard 대기·부분합 트래픽이 남아 있다.

## 정확도 / 검증

- B1 dGate와 dTri는 이전 커널과 bit-exact. 나머지4개 출력도 기존 한도 내. 독립 B1 기준과도 검증했다.
- 전체 모듈 일반 입력:11개 gradient 독립 기준 한도0.05% 통과.
- x·입력/출력 가중치·dy·dropout mask·pair mask 변경: 기존 경로 대비 한도 통과, CUDA graph와 eager는 모든 출력 bit-exact.
- 새 B1 변경으로 B7 입력 dGate/dTri가 달라지지 않아 입력 가중치 gradient와 dx는 기존과 bit-exact.
- **기존 L768 B7 문제는 그대로:** 변경 입력 dWL 독립 기준 상대L2 0.055569% >0.05%. 기존 경로도 정확히 같은 값으로 실패. 허용치를 완화하거나 production에 승격하지 않았다.
- 최종 선택 커널 memcheck L384/L768 오류0, racecheck L384 hazard0. 전체 모델 학습 재시작·optimizer step 시험은 하지 않았다.

## 재현

```bash
sbatch runs/trimul_b1_shared_20260921/final.sbatch
```

실험은 node02 GPU 두 장만 사용했다. 기존 node02 학습·Claude 잡과 node01을 건드리지 않았다.
`module-L*.json`에 paired samples·검증, `ncu-*.ncu-rep`/CSV에 프로파일, `compiler-summary.json`에 cubin/source hash와 컴파일 자원, `selection` 파일에 최종 설정을 남겼다.
