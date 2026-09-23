"""Render the recorded module measurements without substituting historical timings."""
import csv
import hashlib
import html
import json
from pathlib import Path
import subprocess

R=Path(__file__).resolve().parent
repo=R.parents[1]
arms=('pytorch','cuequiv','engine1','engine2')
headers=('PyTorch','cuEquivariance','Engine 1.0.0','Engine 2.0.0')
modules={'trimul':'양방향 TriMul','transition':'Transition','block':'MiniPairformer 1블록',
         'single':'단방향 TriMul (outgoing)','opm':'OPM','pwa':'PWA','msa':'MSA module 1블록','dit':'Token DiT'}
data={}
for f in R.glob('*.json'):
    if f.name.endswith('-trace.json') or f.name=='manifest.json':continue
    d=json.loads(f.read_text())
    if 'arm' in d:data[d['module'],d['L'],d['arm']]=d

for module in ('opm', 'pwa', 'msa'):
    for L in (384, 768):
        for arm in ('pytorch', 'engine1', 'engine2'):
            assert data[module,L,arm]['msa_depth'] == 1024, (module,L,arm)

intro='''# H100 모듈 비교 — 2026-09-23

Engine 2.0.0은 공개 v2.0.0 태그 뒤에 로컬에서 추가한 production 배선을 포함한 wheel이다. 아직 공개 태그나 실행 중인 학습 환경에 이 배선이 반영된 것은 아니다.
Engine 1.0.0은 실제 v1.0.0 태그 `850e160c`이다. 버전 문자열이 1.0.0인 9월 설치본 `1bc0803e` 및 과거 개별 커널 표와 구분한다.

조건: node01 H100 80GB, B=1, BF16 activation/projection, FP32 LN affine, static torch.compile + CUDA graph replay. 각 경우 7회 × 50 replay의 회당 시간 중앙값(ms). 컴파일·첫 튜닝·CPU 실행·optimizer 제외. 학습은 fwd+bwd, 입력 및 전체 parameter gradient, 실제 dropout RNG 포함. mask 적용. 양방향 TriMul 방향별 hidden=128, D=128, dropout=25%; Transition expansion=4. OPM/PWA/MSA module은 MSA depth=1024, MSA64/pair128, PWA head8×32/dropout15%. Token DiT는 sample1/single768/condition384/pair128/head16/expansion1536.

MSA module은 실제 `MiniMSAModuleBlock(last_block=False)` 1개: OPM → PWA → MSA Transition(D64) → 양방향 TriMul → Pair Transition(D128). 입력 embedding과 전체 4블록 stack은 제외하며, 학습에서는 MSA·pair 두 출력의 gradient를 모두 계산한다. MSA mask와 token mask를 함께 적용한다. 전용 cuEq MSA module이 없어 cuEq 열은 `—`이다. v1.0.0 측정은 현재 team-gm의 무관한 SWA import 의존성을 피하기 위해 같은 블록 클래스 본문을 그대로 읽어 과거 엔진 클래스에 직접 연결했다(연산·배선 변경 없음).

cuEquivariance 양방향 TriMul은 동일한 shared 2H output LN 수식을 유지하는 vendor primitive 조합이며, 단방향 public TMU 두 개의 합이 아니다. MiniPairformer 1블록은 양방향 TriMul + Transition이며 cuEq 열에서는 Transition을 PyTorch로 실행한다. 전용 cuEq 구현 없는 모듈은 `—`. v1.0.0에는 DiTBlock이 없어 해당 셀도 `—`. 오류는 다른 backend 숫자로 대체하지 않는다. 신규 전체 config 튜닝은 수행하지 않았으며 기본 경로의 첫 사용 튜닝 정책을 유지했다.
'''
md=[intro]
md.append('''
## 결론과 적용 범위

- OPM/PWA 표준 MSA depth를 1024로 수정하고 세 backend 모두 재측정했다. 기존 depth256 결과는 `msa-depth256/`에 보존했다.
- OPM/PWA의 Claude 개발 기록과는 baseline·compile·dropout·OPM 추론 residual/배선이 다르다. [원본 세션과 비교](../msa-claude-compare-20260923/README.md).
- MiniPairformer 1.520ms와 1.648ms 차이는 [동일 GPU 비용 분리 측정](block-gap.md)을 참고한다.

- 기본 `miniworld` / `auto`에서 8모듈 × 2길이 × 2모드 = 32건의 실행·유한값·커널 trace 검사를 완료했다. [검사 결과](dispatch-audit.json).
- 양방향 TriMul, Transition, OPM, PWA는 새 학습·추론 경로가 실행된다. 이 전체 비교표를 측정한 당시 단방향 TriMul과 Token DiT 학습은 기존 경로였다. 이후 단방향 D128 학습을 추가했으며 최신 별도 표를 참고한다.
- 양방향 TriMul 학습은 v1.0.0 태그 대비 L384 1.75배, L768 1.72배 빠르다. MiniPairformer는 cuEq TriMul + PyTorch Transition 대비 학습 2.13배/2.10배 빠르다.
- Token DiT sample1에서는 새 추론 경로가 PyTorch보다 느리다. 전체 shape의 최적 backend 선택이나 튜닝이 끝났다는 결론은 아니다. 단방향 TriMul 및 DiT 학습에서도 실제 cache miss가 관찰됐다.
- Transition의 native 빌드 확인이 Dynamo 안으로 들어가는 문제를 수정했다. narrow/wide CPU 회귀 검사 4건과 수정 후 GPU forward/backward 실행을 확인했다.
- v1.0.0 Transition/블록은 이 PyTorch 2.10 환경에서 CUDA assertion 또는 custom-op alias 제약으로 실패했다. 단방향 학습은 원래 parameter 일부의 gradient가 연결되지 않아 실패했다. 다른 revision/커널의 값으로 채우지 않았다.
- 아래는 pair D128 중심의 제한된 shape 비교다. D64/256/384/512의 추가 최적화 및 전체 cache 재튜닝 완료를 뜻하지 않는다. OPM은 pair residual을 포함한다.
''')
html_tables=[]
rows=[]
for mode,title in [('inference','추론 forward'),('training','학습 forward + backward')]:
    md+=['\n## '+title+'\n','| 모듈 | L | '+' | '.join(headers)+' |','|---|---:|---:|---:|---:|---:|']
    table=['<h2>'+title+' · ms</h2><div class="scroll"><table><thead><tr><th>모듈</th><th>L</th>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>']
    for module,label in modules.items():
        for L in (384,768):
            values=[]
            for arm in arms:
                d=data.get((module,L,arm),{});v=d.get('modes',{}).get(mode,{})
                text=f"{v['ms']:.3f}" if 'ms' in v else ('오류' if 'error' in v or 'error' in d else '—')
                values.append(text)
                rows.append(dict(module=module,L=L,msa_depth=d.get('msa_depth',''),mode=mode,backend=arm,ms=v.get('ms',''),status=text))
            md.append('| '+label+' | '+str(L)+' | '+' | '.join(values)+' |')
            table.append('<tr><td>'+label+'</td><td>'+str(L)+'</td>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
    table.append('</tbody></table></div>');html_tables.append(''.join(table))
md.append('\n## 실제 엔진 2 경로 (profiler kernel 이름)\n')
for module,label in modules.items():
    for mode in ('inference','training'):
        d=data.get((module,384,'engine2'),{}).get('modes',{}).get(mode,{})
        names=[n for n in d.get('kernels',{}) if any(s in n for s in ('tmn_k','b1_fused','b7_joint','save_k','infer_k','transition_fwd_fused','transition_bwd_fused','opm_','pwa_','wide_d64_','fused_','dit_'))]
        md.append('- '+label+' '+mode+': '+('; '.join(names[:12]) if names else '원본 JSON 참조'))
md.append('\n## 실패 기록\n')
for key,d in data.items():
    for mode,v in d.get('modes',{}).items():
        if 'error' in v:
            md.append('- '+str(key)+' '+mode+': `'+next((l for l in v['error'].splitlines() if 'Error:' in l or 'Unsupported:' in l),'실행 오류')+'`')
    if 'error' in d:md.append('- '+str(key)+': `'+d['error'].splitlines()[-1]+'`')
(R/'README.md').write_text('\n'.join(md)+'\n')
with (R/'results.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
wheel=repo/'.release-build/h100-dist/miniworld_engine-2.0.0-py3-none-any.whl'
manifest=dict(engine1_commit=subprocess.check_output(['git','rev-parse','v1.0.0^{commit}'],cwd=repo,text=True).strip(),
 engine2_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
 engine2_post_tag_local_wiring=True,wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
 measurements=len(data),benchmark_sha256=hashlib.sha256((R/'bench.py').read_bytes()).hexdigest())
package=repo/'.release-build/h100-installed/miniworld_engine'
manifest['benchmarked_transition_source_sha256']={name:hashlib.sha256((package/'kernels/transition/cuda'/name).read_bytes()).hexdigest() for name in ('fused_sm90a.py','fused_wide_sm90a.py')}
manifest['comparison_expected_records']=52
manifest['msa_block_source_sha256']=hashlib.sha256((repo.parent/'src/miniworld/modules/mini_msa_module.py').read_bytes()).hexdigest()
manifest['standard_msa_depth']=1024
manifest['previous_msa_measurements']='msa-depth256/'
manifest['native_compile_guard_regression']='tests/compile/test_transition_fake_availability.py: 4 passed'
manifest['timing_policy']='7 rounds x 50 CUDA graph replays; approximately 300ms warmup; median ms/replay'
(R/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Engine 2.0.0 H100 비교</title><style>body{font:16px/1.6 system-ui,sans-serif;background:#0e1724;color:#e5edf9;margin:24px auto;max-width:1200px;padding:0 18px}h1,h2{color:#92d9fa}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap;margin-bottom:28px}th,td{border-bottom:1px solid #34435a;padding:10px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#223149}td:last-child{color:#72edc2;font-weight:bold}a{color:#92d9fa}p{max-width:1000px}.note{padding:15px;background:#223149;border-radius:8px}</style><h1>Engine 2.0.0 · H100 모듈 실측</h1><p class="note">2.0.0은 공개 태그 이후 로컬 배선 수정 포함. 1.0.0은 실제 v1.0.0 태그(850e160c). 단위 ms, 작을수록 빠름.</p><p>B=1 · pair D128 · L384/768 · BF16 · static compile + CUDA graph · 학습은 fwd+bwd와 dropout RNG 포함. 최초 컴파일·튜닝·optimizer 제외. MiniPairformer = 양방향 TriMul + Transition. cuEq 블록의 Transition은 PyTorch. OPM/PWA/MSA module MSA depth1024. MSA module = 1블록(OPM → PWA → MSA Transition → 양방향 TriMul → Pair Transition), embedding 제외, 두 출력의 backward 포함. Token DiT sample1.</p>'''+''.join(html_tables)+'''<h2>Claude OPM/PWA 개발 기록과 대조 · L384 / S1024</h2><p>서로 다른 조건의 기록입니다. Claude 기준은 eager, 현재 표는 compile + graph. PWA 학습 dropout은 당시 0, 현재 0.15입니다. OPM 추론은 당시 residual 없음·추론 전용, 현재 residual 포함·학습 forward 재사용입니다.</p><div class="scroll"><table><tr><th>항목</th><th>Claude 최종 ms</th><th>원본 벤치 재현 ms</th><th>현재 비교표 ms</th></tr><tr><td>PWA 추론</td><td>0.411</td><td>0.412</td><td>0.407</td></tr><tr><td>PWA 학습</td><td>1.561</td><td>1.562</td><td>1.586</td></tr><tr><td>OPM 추론</td><td>0.633</td><td>0.636</td><td>0.703</td></tr><tr><td>OPM 학습</td><td>2.028</td><td>2.031</td><td>2.035</td></tr></table></div><p>기존 엔진 분모도 다릅니다: Claude는 개발 브랜치 own/eager, 이 표는 실제 v1.0.0 태그/compile+graph. 동일 GPU OPM 비교에서 residual 비용 약35–37µs, 현재 경로 추가 비용 약19–21µs. CUDA 소스 8개 SHA-256 동일.</p><p><a href=".engine-release-2.0.0/verdicts/msa-claude-compare-20260923/README.md">Claude OPM/PWA 개발 기록과 비교: 기준 경로·dropout·추론 배선 차이</a></p><p>OPM/PWA는 depth1024로 재측정. 이전 depth256 기록은 별도 보존. <a href=".engine-release-2.0.0/verdicts/version-compare-20260923/block-gap.md">MiniPairformer 1.520 → 1.648ms 차이 분석</a></p><p>—: 전용 구현 또는 당시 모듈 없음. 오류: 해당 조건에서 실행 실패. 임의 fallback 값으로 대체하지 않음.</p><p>배선: 양방향 TriMul·Transition·OPM·PWA 학습/추론 연결. 이전 전체표의 단방향·Token DiT 학습은 기존 경로. 단방향 D128 최신 결과는 상단 별도 표. 모든 shape/컨피그의 튜닝 완료를 의미하지 않음.</p><p><a href=".engine-release-2.0.0/verdicts/version-compare-20260923/README.md">측정 조건·커널 이름·실패 기록</a> · <a href=".engine-release-2.0.0/verdicts/version-compare-20260923/results.csv">CSV</a> · <a href="TRIMUL_STATUS.html">TriMul 현황</a></p></html>'''
fusion = R.parent/'msa-residual-dropout-20260923/results.csv'
if fusion.exists():
    with fusion.open() as f:
        fused_rows = list(csv.DictReader(f))
    recent = '<h2>최신 추가 개발 · OPM residual / PWA backward dropout 융합</h2><p>S1024 · dropout0.15(PWA 학습) · residual 포함 · 같은 H100에서 compile + CUDA graph 전후 비교. GPU 검사17건, memcheck/racecheck 오류0건.</p><div class="scroll"><table><tr><th>항목</th><th>L</th><th>수정 전 ms</th><th>수정 후 ms</th><th>배속</th></tr>'
    for r in fused_rows:
        if (r['module'],r['mode']) not in (('opm','inference'),('pwa','training')):
            continue
        label = 'OPM 추론' if r['module']=='opm' else 'PWA 학습 fwd+bwd'
        recent += f"<tr><td>{label}</td><td>{r['L']}</td><td>{float(r['before_ms']):.3f}</td><td>{float(r['after_ms']):.3f}</td><td>{float(r['speedup']):.3f}×</td></tr>"
    recent += '</table></div><p>OPM 전체 학습은 1% 미만 차이, PWA 추론은 기존과 비슷합니다. OPM에는 모델 정의상 dropout이 없습니다. 아래 전체 모듈 비교는 이번 추가 융합 전 기록입니다. <a href=".engine-release-2.0.0/verdicts/msa-residual-dropout-20260923/README.md">구현·전체 결과·검증</a></p><h2>추가 융합 전 · 전체 모듈 비교 기록</h2>'
    page = page.replace('<h1>Engine 2.0.0 · H100 모듈 실측</h1>', '<h1>Engine 2.0.0 · H100 모듈 실측</h1>'+recent)
prep = R.parent/'trimul-preparation-20260923/results.json'
if prep.exists():
    records = json.loads(prep.read_text())
    recent = '<h2>최신 · TriMul D128 파라미터 배치 및 중복 준비 제거</h2><p>동일 H100 전후 교대 측정 · 학습 fwd+bwd · dropout25%와 RNG 포함 · compile + CUDA graph. 네 입력 가중치를 backward용 배치로 저장하고, forward packing을 backward에서 재사용합니다.</p><div class="scroll"><table><tr><th>범위</th><th>L</th><th>변경 전 ms</th><th>변경 후 ms</th><th>단축 µs</th></tr>'
    for key, value in records.items():
        module, length = key.split('-')
        before, after = value['ms']['before'], value['ms']['after']
        label = '양방향 TriMul' if module == 'trimul' else 'MiniPairformer 1블록'
        recent += f'<tr><td>{label}</td><td>{length}</td><td>{before:.3f}</td><td>{after:.3f}</td><td>{(before-after)*1000:.1f}</td></tr>'
    recent += '</table></div><p>검증11건 통과. 구 optimizer 체크포인트의 모멘텀 배치는 재개 시 1회 변환이 필요합니다. 신규 optimizer는 불필요합니다. 아래 과거 전체 비교 표는 이번 변경 전 기록입니다. <a href=".engine-release-2.0.0/verdicts/trimul-preparation-20260923/README.md">구현·검증·optimizer 재개 방법</a></p>'
    page = page.replace('<h1>Engine 2.0.0 · H100 모듈 실측</h1>', '<h1>Engine 2.0.0 · H100 모듈 실측</h1>'+recent)
history = R.parent/'block-history-audit-20260923/summary.json'
if history.exists():
    evidence = json.loads(history.read_text())
    note = '<h2>1.520 → 1.607ms 원인 확인 · 동일 GPU 원본 재현</h2><p>과거/현재 코드를 같은 입력과 dropout25% 고정 마스크로 비교했습니다. 두 행의 타이머는 동일하며 함께 실행하는 비교 대상만 바뀝니다.</p><div class="scroll"><table><tr><th>실행 부하</th><th>과거 ms</th><th>현재 ms</th><th>SM 중앙값</th></tr>'
    for key, label in [('native_per_replay', 'CUDA 두 구현끼리 교대'), ('original_mix', '과거처럼 PyTorch/cuEq와 교대')]:
        t = evidence['timings_us'][key]
        clock = evidence['telemetry'][key]['sm_mhz']
        note += f"<tr><td>{label}</td><td>{t['historical']/1000:.3f}</td><td>{t['current_fixed']/1000:.3f}</td><td>{clock:.0f}MHz</td></tr>"
    note += '</table></div><p>같은 조건의 전체 차이는 약5µs(0.34%). 단독 교대 측정에서는 TriMul 약14µs 증가, Transition 약13µs 감소. 현재 경로의 추가 packing/복사·작업 버퍼 초기화 비용은 남지만, 87µs 전체를 코드 퇴행으로 볼 수 없습니다. <a href=".engine-release-2.0.0/verdicts/block-history-audit-20260923/README.md">원본·클럭·커널별 분석</a></p>'
    page = page.replace('<h1>Engine 2.0.0 · H100 모듈 실측</h1>', '<h1>Engine 2.0.0 · H100 모듈 실측</h1>'+note)
single = R.parent/'trimul-single-20260923/legacy-bench.json'
if single.exists():
    rows=json.loads(single.read_text())
    if len(rows)==4:
        note='<h2>최신 · 단방향 TriMul 학습도 native H100 연결</h2><p>D=hidden=128 · B1 · outgoing/ingoing · dropout25%/residual 포함 · static compile + CUDA graph · 고정 dropout mask로 RNG 제외. 같은 GPU에서 세 경로를 교대 측정. 다른 D의 단방향 학습은 아직 기존 경로입니다.</p><div class="scroll"><table><tr><th>L</th><th>방향</th><th>기존 Triton 학습 ms</th><th>기존 H100 학습 ms</th><th>새 CUDA 학습 ms</th><th>기존 H100 대비</th><th>새 학습 fwd ms</th></tr>'
        for r in rows:
            t=r['times_ms'];direction='outgoing' if r['outgoing'] else 'ingoing'
            note+=f"<tr><td>{r['L']}</td><td>{direction}</td><td>{t['triton']:.3f}</td><td>{t['legacy_cuda']:.3f}</td><td>{t['cuda']:.3f}</td><td>{t['legacy_cuda']/t['cuda']:.2f}×</td><td>{t['cuda_fwd']:.3f}</td></tr>"
        note+='</table></div><p>K1(x_n 저장) → 단일 cuBLAS → K3(LN·projection·gate·dropout·residual). Backward: fused B1 → cuBLAS 2회 → producer-consumer B7. D128에 대한 제한된 컨피그 탐색 완료; 전체 탐색/SoL90 완료라는 뜻은 아닙니다. Triton은 일부 cache miss에서 기본 24개 후보를 탐색했습니다.</p><p><a href=".engine-release-2.0.0/verdicts/trimul-single-20260923/README.md">구현·전체 성능·검증 기록</a></p>'
        page=page.replace('<h1>Engine 2.0.0 · H100 모듈 실측</h1>','<h1>Engine 2.0.0 · H100 모듈 실측</h1>'+note)
wide_section = R.parent/'trimul-wide-preparation-20260923/section.html'
if wide_section.exists():
    page = page.replace('<h1>Engine 2.0.0 · H100 모듈 실측</h1>', '<h1>Engine 2.0.0 · H100 모듈 실측</h1>'+wide_section.read_text())
(repo.parent/'ENGINE_2_COMPARISON.html').write_text(page)

# Keep a repository-local copy for readers of the pushed evidence.
(R/"index.html").write_text(page.replace(".engine-release-2.0.0/verdicts/", "../"))
