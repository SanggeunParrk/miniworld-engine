from pathlib import Path
import json,statistics,collections
R=Path(__file__).resolve().parent
d=json.loads((R/'results.json').read_text());assert d.get('complete') and d['job']=='16209'
groups=collections.defaultdict(list)
for row in d['telemetry']:
 try:groups[row['phase'].split('/')[0]].append([float(x) for x in row['csv'].split(',')])
 except ValueError:pass
tele={k:dict(n=len(v),sm_mhz=statistics.median(a[0] for a in v),power_w=statistics.median(a[2] for a in v),power_limit_w=statistics.median(a[3] for a in v)) for k,v in groups.items()}
profiles={}
for name in ('historical','corrected_research','current_fixed'):
 events=[e for e in json.loads((R/(name+'-trace.json')).read_text())['traceEvents'] if e.get('cat')=='kernel' and 'spin_kernel' not in e['name']]
 sums=collections.defaultdict(float);counts=collections.Counter()
 for e in events:
  n=e['name'];cat='math'
  if 'FillFunctor' in n:cat='scratch_zero'
  elif 'copy' in n.lower() or 'clone' in n or 'cat_stack' in n:cat='packing_copy'
  sums[cat]+=e['dur']/5;counts[n]+=1
 assert all(v%5==0 for v in counts.values()),counts
 profiles[name]=dict(launches_per_replay=len(events)//5,gpu_us=dict(sums),kernels={n:dict(count_per_replay=v//5,mean_us_per_replay=sum(e['dur'] for e in events if e['name']==n)/5) for n,v in counts.items()})
summary=dict(job=d['job'],gpu_uuid=d['gpu_uuid'],timings_us=d['times'],telemetry=tele,profiles=profiles)
(R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# 1.520ms와 1.607ms: 같은 GPU에서 원본 재현', '', '**대부분의 차이는 측정 부하와 GPU 클럭 차이였다. 같은 조건에서 원본과 현재 구현의 차이는 약 5µs(0.34%)였다. Transition은 퇴행하지 않았으며 현재 구현이 더 빨랐다.**', '', '## 전체 블록: 측정 방식만 바꾼 비교', '', 'job16209, node01 H100, B1/L384/D128, 양방향 TriMul + Transition 1블록. 원본 연구 fixture의 입력·가중치·upstream gradient를 모든 구현에 동일하게 사용. dropout25% 고정 마스크·pair mask·두 residual·모든 gradient 포함. 아래 두 행 모두 CUDA graph 한 replay마다 event를 기록하는 동일한 타이머다. 서로 다른 구성은 함께 실행하는 비교 대상뿐이다.', '', '| 실행 부하 | 과거 구현 ms | 현재 구현 ms | SM 중앙값 | 전력 중앙값 |', '|---|---:|---:|---:|---:|']
for k,label in [('native_per_replay','CUDA 두 구현끼리 교대'),('original_mix','과거 PyTorch/cuEq 4개 비교 대상 + CUDA 두 구현 교대')]:
 t=d['times'][k];v=tele[k];lines.append(f"| {label} | {t['historical']/1000:.6f} | {t['current_fixed']/1000:.6f} | {v['sm_mhz']:.0f}MHz | {v['power_w']:.1f}W |")
lines += ['', '각 방식 5라운드 × 250회/구현. 과거 비교 대상은 PyTorch eager/compile, cuEq+PyTorch Transition, cuEq+원본 CUDA Transition이다. CUDA 두 구현끼리 반복하면 전력 사용량이 700W 제한에 가까워지고 클럭이 낮아졌다. 비교 대상 구성이 바뀌면서 두 코드 모두 약 101µs 빨라졌다. 전력 제한에 의한 throttle 플래그 자체를 측정한 것은 아니므로, 모든 지연을 특정 전력 관리 원인으로 단정하지 않는다.', '', '과거 job15592의 1.519600ms를 이번 원본 1.516800ms로 약 0.18% 이내 재현했다. 원본 B1은 당시 선택 파일과 같은 cubin 경로, B7은 선택 manifest의 SHA-256과 일치함을 확인했다. Transition은 당시 보존한 소스 스냅샷을 독립된 확장 이름으로 빌드했다.', '', '## 최근 표와 같은 연속 batch 방식', '', '12라운드 × 120 replay/구현, 순서 순환·반전. 전후 60 replay warmup. 이 값과 위의 per-replay 중앙값을 혼합해 차이를 분해하지 않는다.', '', '| 조합 | µs |', '|---|---:|']
for k,v in d['times']['whole_block'].items():lines.append(f'| {k} | {v:.3f} |')
lines += ['', '- historical: 1.520ms를 냈던 원본 TriMul + 원본 Transition.', '- historical_new_transition: 원본 TriMul + 현재 Transition.', '- corrected_research: LN gradient 정확도를 수정한 연구 TriMul + 현재 Transition.', '- current_fixed: 준비 비용을 개선한 현재 모듈 경로 + 현재 Transition, 같은 dropout mask.', '- current_rng: 같은 현재 모듈 경로이며 dropout mask RNG 생성까지 포함.', '', '현재 fixed/RNG 차이 약 -2.7µs는 이 반복의 변동 수준이며, RNG가 속도를 개선했다는 뜻이 아니다. 과거 구현도 이 부하에서 1.607ms이므로, 과거의 1.520ms와 현재의 1.607ms를 그대로 비교해 87µs의 코드 퇴행이라고 해석할 수 없다.', '', '## TriMul 및 Transition 분리 측정', '', '각 행 안의 과거/현재는 같은 입력·upstream·GPU에서 교대 측정했다. 행마다 클럭/부하가 달라지므로 단독 시간을 합쳐 전체 블록 시간을 만들지 않는다.', '', '| 범위 | 과거 µs | 현재 µs | 현재-과거 µs |', '|---|---:|---:|---:|']
for k in ('trimul','transition'):
 t=d['times'][k];b=t['historical'];a=t['current'];lines.append(f'| {k} | {b:.3f} | {a:.3f} | {a-b:+.3f} |')
lines += ['', 'Transition forward 소스 SHA-256은 동일하다. backward 본체도 동일하고, 현재는 부분합 reduction을 병렬화한 버전이다. trace에서 reduction은 약 19.5→4.5µs였다. 따라서 이전보다 60µs 느려졌다는 추정은 동일 조건 비교로 지지되지 않는다.', '', '## 남은 실제 모듈 비용', '', 'corrected_research와 current_fixed는 출력 및 모든 gradient가 bit-exact였다. 같은 수식을 계산하지만, 연구용 경로는 계획/작업 버퍼를 미리 할당하여 반복 사용한다. 현재 모듈은 호출별 소유권을 가진 작업 버퍼를 초기화하고 mask 및 남은 출력 가중치 배치를 변환한다.', '', '| 동일 입력 trace · 5 replay 평균 | 정확도 수정 연구 경로 | 현재 모듈 | 증가 |', '|---|---:|---:|---:|']
for cat in ('packing_copy','scratch_zero'):
 b=profiles['corrected_research']['gpu_us'][cat];a=profiles['current_fixed']['gpu_us'][cat];lines.append(f'| {cat} µs | {b:.3f} | {a:.3f} | {a-b:+.3f} |')
lines += ['', '합계 약 13.1µs의 추가 복사/packing/zero GPU 작업이 관찰됐다. 별도 연속 batch 측정의 corrected_research→current_fixed 차이는 약14.5µs다. trace의 GPU 실행 시간 합과 전체 반복 중앙값은 서로 다른 측정이므로 정확한 가산 분해로 취급하지 않는다. 단일 B1/B7 body 시간은 [원본 결과](results.json)의 kernel_bodies 항목에 있다. 과거 B7은 LN gradient의 엄격한 한도를 넘던 정확도 수정 전 버전이다.', '', '## 검증 및 재현 범위', '', '- current_fixed와 corrected_research: 출력 및 16개 gradient 모두 bit-exact. 과거 gradient와의 작은 차이는 기존 B7 정확도 수정과 Transition LN reduction 순서 변경을 포함한다.', '- historical snapshot의 Transition 소스는 원본 manifest SHA와 대조. forward SHA 동일, backward diff는 reduction 변경.', '- Native Transition fwd/bwd와 B1/B7이 실제로 실행된 것을 5 replay profiler trace로 확인. CUPTI 준비용 spin kernel은 집계에서 제외.', '- 모델 소스나 실행 중 학습을 변경하지 않은 진단 작업이다.', '- 초기 16205/16206은 원본 연구 의존성 import 복원 단계에서 중단됐다. 16207은 원본 fixture가 전역 backend를 Triton으로 바꾼 영향으로 현재 Transition이 Triton에 들어가 제외했다. `rejected-triton-transition.json`에 보존했으며 최종 16209는 fixture 구성 후 production 설정 복원 및 native available 검사를 통과했다.', '- [실행 스크립트](audit.py), [Slurm](audit.sbatch), [원본 결과·클럭·반복 표본](results.json), [요약·프로파일 분류](summary.json).', '']
(R/'README.md').write_text('\n'.join(lines))
