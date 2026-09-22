from pathlib import Path
import csv,hashlib,html,json,re,subprocess
R=Path(__file__).resolve().parent;ROOT=R.parents[1];SITE=ROOT/'runs/anthropic_b1b4_pipeline_20260919/site-visuals/dist'
D={n:json.loads((R/('module-L%d.json'%n)).read_text()) for n in (384,768)}
rows=[]
for n,d in D.items():
 for scope,label in [('b1','B1–B4'),('backward','전체 BWD'),('forward_backward','전체 FWD+BWD')]:
  t=d['times'][scope];a,b=(t[k]['median_us'] for k in ('baseline','candidate'));rows.append([str(n),label,'%.3f μs'%a,'%.3f μs'%b,'%.3f×'%(a/b),'−%.2f%%'%(100*(1-b/a))])
headers=['L','범위','직전 saved-x_n 기준','새 공유 계산 커널','속도 배율','지연 감소']
def md(h,rs):return '\n'.join(['| '+' | '.join(h)+' |','|'+'---|'*len(h),*['| '+' | '.join(r)+' |' for r in rs]])
def table(h,rs):return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in h)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in r)+'</tr>' for r in rs)+'</tbody></table></div>'
metrics=['gpu__time_duration.sum','gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed','dram__bytes_read.sum','dram__bytes_write.sum','sm__throughput.avg.pct_of_peak_sustained_elapsed','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed','sm__warps_active.avg.pct_of_peak_sustained_active','smsp__issue_active.avg.pct_of_peak_sustained_active','smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio','smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio']
profile={};pr=[]
for n in D:
 for variant in ('baseline','candidate'):
  f=R/('ncu-%s-L%d-raw.csv'%(variant,n));rs=list(csv.DictReader(f.open()));units,values=rs[0],rs[-1];key='%s-L%d'%(variant,n)
  profile[key]={k:dict(value=float(values[k]),unit=units[k]) for k in metrics}
  def val(k):return float(values[k])
  def mb(k):return val(k)*({'byte':1e-6,'Kbyte':.001,'Mbyte':1,'Gbyte':1000}[units[k]])
  pr.append([str(n),variant,'%.1f MB'%mb('dram__bytes_read.sum'),'%.1f MB'%mb('dram__bytes_write.sum'),'%.1f%%'%val('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'),'%.1f%%'%val('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'),'%.1f%%'%val('sm__warps_active.avg.pct_of_peak_sustained_active')])
(R/'ncu-summary.json').write_text(json.dumps(profile,indent=2))
logs={}
for n,tool in [(384,'memcheck'),(768,'memcheck'),(384,'racecheck')]:
 p=R/('%s-L%d.log'%(tool,n));s=p.read_text();ok=('ERROR SUMMARY: 0 errors' in s) if tool=='memcheck' else ('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in s)
 assert ok,p;logs[p.name]=True
compiler={}
for n,d in D.items():
 p=Path(d['cubin']);log=p.with_suffix('.ptxas.log').read_text();sass=R/('sass-L%d.txt'%n)
 with sass.open('w') as f:subprocess.run(['/usr/local/cuda-12.9/bin/cuobjdump','--dump-sass',str(p)],stdout=f,check=True)
 ss=sass.read_text();assert 'HGMMA' in ss and 'UTMALDG' in ss
 compiler[n]=dict(cubin=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),ptxas=[x.strip() for x in log.splitlines() if 'spill' in x or 'Used ' in x],instructions={k:ss.count(k) for k in ('HGMMA','UTMALDG','UTMASTG','SETMAXNREG','STL','LDL')},build_metadata=json.loads(p.with_suffix('.json').read_text()))
(R/'compiler-summary.json').write_text(json.dumps(compiler,indent=2))
ph=['L','경로','DRAM 읽기','DRAM 쓰기','DRAM 처리율 / peak','Tensor active / peak','Occupancy']
readme='''# B1–B4: shared tile recomputation, saved input x_n

2026-09-21 · **개발 경로에 연결 완료. Production 승격 아님.**
Anthropic v5 TMA/WGMMA/LN primitives와 reference 연산 순서를 계승하고, Miniworld 학습 미분·공유 스케줄을 추가했다.
비교 대상은 **직전 Anthropic 파생 saved-x_n 학습 경로 `split_xn_pc1`**이다. 원본 Anthropic 추론이나 cuEquivariance와의 새 비교가 아니다.

## 같은 실행에서 직접 측정

'''+md(headers,rows)+'''

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

'''+md(ph,pr)+'''

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
'''
(R/'README.md').write_text(readme)
assets=SITE/'assets'
for n in D:
 for src,dst in [('module-L%d.json'%n,'b1-shared-module-L%d.json'%n)]: (assets/dst).write_bytes((R/src).read_bytes())
(assets/'b1-shared-report.md').write_text(readme);(assets/'b1-shared-ncu.json').write_bytes((R/'ncu-summary.json').read_bytes())
section='''<!-- B1_SHARED_BEGIN --><section class="panel" id="b1-shared"><span class="pill green">2026-09-21 · saved x_n · B1–B4 공통 계산 공유</span><h2>B1–B4 1.21× / 1.36× · 전체 학습 1.04× / 1.07×</h2><p>Anthropic 파생의 직전 학습 경로와 비교. <b>출력 LN·gate·projection을 타일당 한 번 계산</b>하고 dGate/dProj를 공유한다. 같은 커널의 후속 단계에서 이미 출력된 dGate를 읽어 dWgate를 계산한다.</p><p class="muted">node02 H100 · BF16 C128/H256 양방향 · dropout25%/mask/residual · 600회 교대 CUDA graph · 전체11개 gradient · Forward/B7 동일.</p>'''+table(headers,rows)+'''<h3>현재 배선 · 내부 계산은 수식과 PyTorch 코드로 표시</h3><p><a href="#current-wiring">확대 가능한 FWD/BWD 배선도 →</a></p><ol><li>Phase A: x_n / tri / dy → 출력 LN·P·G 한 번 → dGate·dProj → dWproj 부분합 + dTri/LN 미분.</li><li>Grid barrier 후 Phase B: HBM의 x_n / dGate 재읽기 → dWgate → 모든 partial reduction.</li></ol><p>한 CUDA launch이며 단계는 순차적이다. dNorm은 shared memory 안에만 있다. 로드 전용 producer warpgroup도 구현했지만 더 느려 제외했다. 선택한 경로는 Phase B에서 TMA double buffering을 사용한다.</p><h3>NCU: 아직 SoL90이 아니다</h3>'''+table(ph,pr)+'''<p>Scratch는5.60MB에서26.22MB로 증가했다(각각 write/read). 전체 재계산을 없애려다 register spill을 늘린 후보보다 두 단계 구조가 빨랐다. <b>1.7× 목표는 미달</b>이며, 낮은 occupancy와 동기화/메모리 대기를 더 줄여야 한다.</p><div class="notice">B1 dGate/dTri bit-exact · 나머지 출력 한도 내 · 변경 입력 graph/eager 일치 · 최종 memcheck 두 길이 오류0 · racecheck L384 hazard0.</div><div class="notice amber">개발 경로 연결 완료, production 승격 아님. 기존 L768 B7 dWL의 독립 기준 오차0.055569%가 한도0.05%를 넘는 문제는 그대로다. 새로운 B1과 기존 경로의 해당 dWL은 bit-exact.</div><p><a href="assets/b1-shared-report.md">설계·제외 후보·검증 보고서</a> · <a href="assets/b1-shared-module-L384.json">L384 원본</a> · <a href="assets/b1-shared-module-L768.json">L768 원본</a> · <a href="assets/b1-shared-ncu.json">NCU 지표</a></p></section><!-- B1_SHARED_END -->'''
p=SITE/'trimul.html';s=p.read_text()
if '<!-- B1_SHARED_BEGIN -->' in s:s=re.sub(r'<!-- B1_SHARED_BEGIN -->.*?<!-- B1_SHARED_END -->',lambda _:section,s,flags=re.S)
else:s=s.replace('<main>','<main>'+section,1)
if 'href="#b1-shared"' not in s:s=s.replace('<nav>','<nav><a href="#b1-shared">B1 공유 계산</a>',1)
# Update current-wiring label; older benchmark sections remain explicitly historical.
s=s.replace('현재 source-backed 경로: <code>split_xn_pc1</code>','현재 source-backed 경로: <code>shared B1 + split_xn_pc1 B7</code>')
p.write_text(s)
p=SITE/'index.html';s=p.read_text();new='<!-- TRAINING_LINK_BEGIN --><section class="notice"><b>B1–B4 공통 계산 공유</b> · <a href="trimul.html#b1-shared">커널1.21×/1.36× · 전체1.04×/1.07× →</a><br><a href="trimul.html#current-wiring">최신 수식·PyTorch·HBM 배선도</a> · 개발 경로, production 승격 아님.</section><!-- TRAINING_LINK_END -->';s=re.sub(r'<!-- TRAINING_LINK_BEGIN -->.*?<!-- TRAINING_LINK_END -->',lambda _:new,s,flags=re.S);p.write_text(s)
print(md(headers,rows))
