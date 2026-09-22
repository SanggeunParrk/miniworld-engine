import json,csv,shutil,collections
from pathlib import Path
R=Path(__file__).resolve().parent;E=R.parent/'trimul_sm90_parity_20260917/engine'
D=E/'docs/benchmarks/anthropic-h100-20260919';D.mkdir(parents=True,exist_ok=True)
rs=json.loads((R/'all-results.json').read_text());ps=json.loads((R/'ncu-summary.json').read_text())
for name in ('all-results.json','ncu-summary.json','boundary-checks.json','native-rebuild-provenance.json','flash_check.json','cpu-tests.log','upstream-gpu-tests.xml','fresh-probe-test.log','source-audit.log'):
 shutil.copy2(R/name,D/name)
with (D/'timings.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['family','L','C','direction','module','row','status','ms','output_rel_rms','update_rel_rms','reason'])
 for r in rs:w.writerow([r['family'],r['length'],r['width'],r['direction'],r['module'],r['row'],r.get('status'),r.get('ms'),r.get('error',{}).get('rel_rms'),r.get('update_error',{}).get('rel_rms'),r.get('reason')])
with (D/'ncu-kernels.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['profile','kernel','us','time_share','HBM_pct','L2_pct','BF16_tensor_pct','issue_pct','eligible_warps','registers'])
 for p in ps:
  for k in p['kernels']:w.writerow([p['profile'],k['name'],k['us'],k['time_share'],k['hbm_pct'],k['l2_pct'],k['tensor_pct'],k['issue_pct'],k['eligible_warps'],k['regs']])
def find(f,L,C,row,module=False,direction='outgoing'):
 return next((r for r in rs if (r['family'],r['length'],r['width'],r['row'],r['module'],r['direction'])==(f,L,C,row,module,direction) and r.get('status')=='passed'),None)
lines=['''# Anthropic inference adoption — H100 audit, 2026-09-19

## 결론

Anthropic의 inference 개발을 계승한다. 자체 engine 구조는 유지하며, 원본
알고리즘과 코드를 출처·라이선스와 함께 가져오고 명시적 inference backend로 연결했다.
학습용 forward/backward 개발은 이번 작업에서 진행하지 않았다.

원본의 모든 optimization kit와 shared runtime을 가져온 것과, 모든 모델을
실행한 것은 다르다. 이번 GPU 검증은 MiniWorld에 대응하는 **16개 연산/부분연산
분류**, L=384/768 중심이다. Evo2/Enformer 등 모델별 전용 경로와 JAX/Pallas는
소스 보존·inventory까지이며 실행·NCU 분석을 완료했다고 주장하지 않는다.
MiniWorld 전체 모델 실행/학습 설정의 일괄 교체도 이번 측정에 포함되지 않는다.

- 원본: [Anthropic release](https://www.anthropic.com/research/claude-uplifts-biomolecular-modeling),
  [pinned source](https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e).
- 원본 파일 5,199개를 Git blob 및 SHA-256로 확인. GPU 문법 탐지 inventory는
  182개 파일/176개 서로 다른 내용이며 **커널 개수가 아니다**. 테스트/변형/문서 오탐도 포함한다.
- 원본 kernel source 수정 없음. 재빌드한 cubin 1개와 매니페스트/체크섬 4개는 원본과
  다르며 [LOCAL_BUILD.json](../third_party/anthropic/LOCAL_BUILD.json)에 별도로 기록했다.
- [연결 API 및 재현 방법](anthropic-integration.md), [출처/개발 방향](project-direction.md).

## 검증 및 측정 조건

node02, H100 80GB HBM3, PyTorch 2.10.0+cu128 / Python 3.10. CUDA 12.9 nvcc,
T16 NVRTC 12.8 재빌드. 기존 학습 잡 13228과 별도의 GPU 2개(할당 13308)를 사용했다.

BF16 projection/input 및 FP32 LayerNorm affine, nonzero weights, TF32 off가 기본이다.
Atom-window는 FP32 API의 IEEE 모드와 원본 기본 `tf32rn` 모드를 **별도로** 측정했다.
DiT modulation/gate는 주기적 conditioning/mask와 FP32 residual도 검사했다.
컴파일·일회성 weight packing을 제외한 1회 호출 CUDA Graph, 50 replay × 5라운드의
중앙값이다. 후보별 별도 프로세스이며 interleaved A/B나 통계적 성능 보증은 아니다.
기존 Triton은 `engine_backend=triton`으로 강제했다. 기존 cache miss 일부에는
heuristic config 선택 로그가 있으므로 전체 후보 공간을 재튜닝한 최선값 비교는 아니다.

**218개 후보/shape 시도: 178개 수치·Graph 검사 통과, 40개 명시적 지원범위 거절.**
미완료/원인 불명 실패 항목은 없다. 거절은 속도가 느리다는 뜻이 아니다.
독립 FP32 식과 비교했으며 BF16 중간 반올림을 명시한 연산은 해당 순서를 반영했다.
전체 출력 상대 RMS 최대 0.538%, residual을 뺀 update 상대 RMS 최대 2.182%.
검사 상한 3%는 이 campaign의 smoke threshold이며 생물학적 정확도 보증이 아니다.
작은 update에서는 BF16 residual add의 반올림이 update 상대오차에 크게 보이므로
전체 출력과 update 오차를 둘 다 기록했다. upstream exact-tier 인증을 승계하지 않는다.

- CPU adapter 계약 테스트 **11 통과**.
- upstream GPU 테스트: **151 통과 / 1 실패 / 6 skip / 18 제외**였고,
  실패한 probe-cache 테스트는 fresh cache에서 **1 통과**. 기존 cache-hit 로그에
  `ratio_rms` 필드가 없어 생긴 테스트 실패로 기록을 보존했다.
- 고메모리 offset/rollout/rowpair 테스트는 제외했다. 6개 skip 이유는 JUnit에 보존했다.
- 추가 B=2 검사 **10개 통과**: 두 배치의 다른 마스크, TriMul outgoing/incoming,
  attention starting/ending, 입력 불변성, weight mutation 후 pack 갱신.
  Transition의 masked 표시는 다른 fixture seed일 뿐 mask API 검사가 아니다.
- `flash_sm90a` 재빌드의 BF16 65,536 패턴 SiLU bit 비교 **mismatch=0**.
- NCU **72개 profile 모두 수치 검사 통과**. Compute/HBM/L2 roofline, occupancy,
  scheduler 수집. NCU replay 시간은 아래 Graph latency와 다르다.

## 기존 Triton 대비 추론 시간

TriMul은 **단방향 outgoing 전체 모듈**, attention은 **전체 모듈**을 우선 표기한다.
단위 ms. 마지막 열은 기존 시간 / 원본 연결 시간이다.

| 연산 | L | C | 기존 Triton | Anthropic 연결 | 배속 |
|---|---:|---:|---:|---:|---:|''']
for fam,C,row,mod,label in [('trimul',128,'native_rebuilt',False,'TriMul outgoing'),('triattn',128,'block:triattn_native',True,'TriangleAttention 전체'),('transition',128,'v2',False,'Transition'),('transition',256,'esm_t16',False,'Transition'),('transition',384,'pf',False,'Transition')]:
 for L in (384,768):
  b=find(fam,L,C,'engine_triton',mod);u=find(fam,L,C,row,mod)
  lines.append(f"| {label} | {L} | {C} | {b['ms']:.4f} | {u['ms']:.4f} (`{row}`) | {b['ms']/u['ms']:.2f}× |")
lines.append('''
C=512 Transition은 이번 BF16 잔차 포함 구성에서 시험한 upstream optimized row가
지원하지 않아 기존 경로를 유지한다. C=384의 `pf`는 오히려 약 3.6–3.7배 느리다.
**기존 구현을 전부 지우거나 upstream을 모든 shape의 자동 기본값으로 바꾸지 않았다.**

### Attention: core 교체와 전체 융합을 구분

| L | 기존 전체 | core만 교체한 전체 | 원본 surround까지 연결한 전체 | 원본 core 단독 |
|---:|---:|---:|---:|---:|''')
for L in (384,768):
 vals=[find('triattn',L,128,row,mod)['ms'] for row,mod in [('engine_triton',True),('triattn_native',True),('block:triattn_native',True),('triattn_native',False)]]
 lines.append('| '+str(L)+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
lines.append('''
Core-only 경로에는 engine의 독립 LN/projection/gating과 layout 준비가 남는다.
권장 연결은 `anthropic_row="block:triattn_native"`: 원본 prologue `v3`, epilogue
`v2`를 함께 사용한다. 엔진의 입력 불변 계약을 지키려고 residual add는 별도다.
원본 `residual=True`는 입력을 덮어쓰므로 그대로 연결하지 않았다.

`triattn_native`라는 row 이름이 항상 CUDA라는 뜻은 아니다. **L384는 Gluon/Triton
`_fwd`, L768는 CUDA `triattn_m1_kernel`** 실행을 NCU에서 확인했다.

## NCU: 한계에 가까운가?

`SM %`를 roofline 도달률로 취급하지 않았다. HBM/L2 및 BF16 tensor의 각
sustained-peak 비율, 실제 kernel 시간 비중을 함께 봤다. 혼합 연산의 낮은 tensor
비율만으로 가능한 가속 배율을 계산할 수 없다. Softmax/SFU, 의존성, occupancy,
launch 비용을 구분하는 추가 실험이 필요하며 이번에는 구현 최적화를 하지 않았다.
NCU는 clock-control/cache-control none: 다음 수치는 현재 warm/cache 조건이다.

| 경로 | 관측 | 판단 |
|---|---|---|
| LN, L768 C128–512 | HBM 85.8–89.8%, L2 약 81–82% | 메모리 대역폭에 가까움. 독립 LN 재개발 우선순위 낮음 |
| LN, L384 C128 | HBM 70.3%, L2 87.3% | L2까지 봐야 함. 큰 개선을 가정하지 않음 |
| TriMul C128 K1/K3, L768 | HBM 약 64%, tensor 약 19.8%/9.5% | roofline 도달로 볼 근거 없음. 주변 연산 개선 후보 |
| Attention core L768 | HBM 20.8%, L2 56.8%, tensor 15.8%, issue 40.1% | HBM/compute roof 미도달. scheduling/latency 영향 추정, 세부 stall 증명은 추가 필요 |
| Transition C256 T16 L768 | tensor 33.1%, L2 57.0%, HBM 12.5% | 기존보다 빠르지만 이 수치로 최적이라고 결론 불가 |
| Transition C384 pf L384 | tensor 6.7%, HBM 1.2%, issue 17.2% | 이 shape에서는 채택하지 않음 |

### TriMul C128 내부 시간 분해

NCU replay 기준 µs / 전체 호출 내 비중. K1은 입력 LN+projection/gate,
K3는 출력 측 LN/projection/gate/residual이다. 중간 contraction은 cuBLAS를 유지한다.

| L | K1 | cuBLAS contraction | K3 | mask/변환 등 |
|---:|---:|---:|---:|---:|''')
for L in (384,768):
 p=next(p for p in ps if p['profile']==f'trimul-L{L}-C128-native_rebuilt');ks=p['kernels'];groups=[[k for k in ks if 'tmn_k1' in k['name']],[k for k in ks if 'nvjet' in k['name']],[k for k in ks if 'tmn_k3' in k['name']]];t=[sum(k['us'] for k in g) for g in groups];t.append(p['total_us']-sum(t))
 lines.append('| '+str(L)+' | '+' | '.join(f'{v:.2f} / {100*v/p["total_us"]:.1f}%' for v in t)+' |')
lines.append('''
Attention 전체의 원본 prologue/core/epilogue/별도 residual 비중은 L384에서
33.0/39.7/16.1/10.0%, L768에서 27.2/49.0/12.9/8.4%다.
Residual add 자체는 L768 HBM 90.9%지만 전체의 8.4%다. API를 유지하는 fusion은
별도 설계 문제이며, memcpy로 대체하거나 입력을 몰래 덮어써서 이득을 만들지 않는다.

## 나머지 연결·분석 범위

| 연산 표면 | 연결/검증 | 범위 |
|---|---|---|
| Pair-bias attention | provider wrapper + L384/768 + NCU | H16 D24, S=2, key mask |
| Atom-window | carried kernel + L384/768 + NCU | atom 수=8L, S=2, tail mask; IEEE와 tf32rn 별도 |
| Gather attention | carried kernel + L384/768 + NCU | shared indices/bias, sorted K=128 |
| Template embedding | carried kernel + L384/768 + NCU | packed template weights, 2 templates |
| DiT AdaLN/SwiGLU/gate-residual | carried kernels + L384/768 + NCU | R=5L; periodic conditioning/mask |
| LN-linear | carried kernel + L384/768 + NCU | 64L rows, C128 |
| MSA LN projection | operation accessor + L384/768 + NCU | two projections; output cat 비용 포함 |
| OPM output | operation accessor + L384/768 + NCU | CH=CE=32, CZ128; contraction 제외 |
| OPM contraction+projection | operation accessor + L384/768 + NCU | S64, CH32, CZ256 scalar_norm; input projections 제외 |
| MSA PWA | operation accessor + L384/768 + NCU | S64, CM64, CZ128, H8 D32; unchunked |
| Model-specific kits / Pallas | 전체 소스와 라이선스 보존 | 모델별 adapter 실행 및 NCU 미완료 |

Gather 기본 `ensure_sorted=True`는 GPU 결과를 host bool로 읽어 Graph capture가
실패했다. 원본 수정 없이 사전 정렬 + `ensure_sorted=False`로 연결해 Graph 검증을
통과했다. 매번 index가 바뀌는 모델의 총 시간에는 정렬 비용을 추가해야 한다.
Atom-window IEEE는 원본 기본 경로보다 훨씬 느리므로 둘을 혼동하지 않는다.

## 채택 판단과 다음 단계

1. **채택 후보**: TriMul native rebuild, 전체 triangle-attention block,
   C256 Transition T16. 명시적 backend/row로 접근 가능하다.
2. **유지/비교**: C128 Transition과 LN은 현재 구현과 거의 동률. C384/512
   Transition은 현재 engine 경로를 유지한다. 자동 선택 규칙은 추가 model-level 검증 후 결정한다.
3. **추가 분석 우선**: attention의 core 및 surround, TriMul K1/K3, T16.
   독립 LN은 낮은 우선순위. 새 kernel 구현·training 개발은 아직 시작하지 않는다.
4. **남은 범위**: 전체 MiniWorld inference 정확도/peak memory/latency,
   모든 dtype·shape·완전히 마스킹된 입력, fused bidirectional TriMul, 모델별
   genomics/JAX 경로의 실행 검증. 이번 kernel-level 통과를 이 범위의 통과로 확대하지 않는다.

## 원자료와 재현

- [모든 후보 시간 CSV](benchmarks/anthropic-h100-20260919/timings.csv)
- [개별 측정·selection·수치오차 JSON](benchmarks/anthropic-h100-20260919/all-results.json)
- [모든 NCU kernel 시간·비중·지표 CSV](benchmarks/anthropic-h100-20260919/ncu-kernels.csv)
- [NCU 요약 JSON](benchmarks/anthropic-h100-20260919/ncu-summary.json)
- [추가 모듈 검증](benchmarks/anthropic-h100-20260919/boundary-checks.json)
- [upstream GPU JUnit](benchmarks/anthropic-h100-20260919/upstream-gpu-tests.xml)
- [fresh-cache 재검사](benchmarks/anthropic-h100-20260919/fresh-probe-test.log)
- [소스 무결성/재빌드 변경 목록](../third_party/anthropic/LOCAL_BUILD.json)

원시 `.ncu-rep`, CSV 및 전체 build log는
`/home/psk6950/MiniWorld/runs/anthropic_adoption_20260919/`에 있다.
`native_rebuilt`는 source SHA를 대조한 CUDA12.9 build와 동반 testvectors를 사용한다.
`flash_sm90a`는 원본 요구 CUTLASS4.2.0, T16은 별도 기록한 CUTLASS4.8+NVRTC12.8
환경으로 재빌드했다. 재빌드 바이너리의 성능/수치 인증을 upstream의 CUDA13 원본
인증과 혼동하지 않는다. 바이너리·NCU 대용량 원자료는 Git 대상에서 제외한다.
''')
(E/'docs/anthropic-h100-audit.md').write_text('\n'.join(lines))
