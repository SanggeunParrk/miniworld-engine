from pathlib import Path
import json,hashlib,html
R=Path(__file__).resolve().parent;ROOT=R.parent.parent
v=json.loads((R/'validation.json').read_text());s=json.loads((R/'stress_block.json').read_text());d=json.loads((R/'diagnose.json').read_text())
assert v['complete'] and v['fixed_strict_validation_passed'] and s['complete'] and d['complete']
entry=json.loads((R/'current_entry.json').read_text()) if (R/'current_entry.json').exists() else None
san=json.loads((R/'sanitizer.json').read_text()) if (R/'sanitizer.json').exists() else []
lnmax=max(x['relative_l2'] for c in v['checks'].values() for n,x in c['fixed']['graph'].items() if n.startswith(('dgamma','dbeta')))
blockmax=max(x['relative_l2'] for c in s['cases'].values() for n,x in c['block']['graph'].items() if n.startswith(('dgamma','dbeta','transition.ln_in.')))
old=v['times']['full']['latest_combined']['median_us'];new=v['times']['full']['fixed']['median_us']
cubin=v['cubins']['fixed']['b7'][0]
selected=dict(status='strict_validated_L384',scope='L384 bidirectional BF16 C128/H256 training',plan=str(R/'fixed/plan.py'),policy=str(R/'policy.py'),kwargs=v['selection']['kwargs'],environment=v['selection']['environment'],cubin=cubin,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'fixed/joint.cu',R/'fixed/single_wg.inc',R/'fixed/plan.py')},threshold_ln=5e-6,max_ln=lnmax,max_block_ln=blockmax,full_original_failures_retested=3,stress_cases=12,stress_scopes=['TriMul','MiniPairformer block'],current_entry=entry,sanitizer=san,production_engine_dispatch_changed=False)
(R/'selected.json').write_text(json.dumps(selected,indent=2))
rows=[]
for case,c in v['checks'].items():
 rows.append('| %s | %.3e | %.3e | %.3e | %.3e |'%(case,c['latest_combined']['graph']['dgamma_in']['relative_l2'],c['fixed']['graph']['dgamma_in']['relative_l2'],c['latest_combined']['graph']['dbeta_in']['relative_l2'],c['fixed']['graph']['dbeta_in']['relative_l2']))
santext='; '.join('%s: %s'%(x['tool'],'통과' if x['returncode']==0 else '실패') for x in san) or '대기 중'
entrytext='검증본과 전체 출력 bit-exact, 동일 cubin 확인' if entry else '연결 완료, GPU 대조 대기 중'
text='''# TriMul 입력 LN gradient 엄격 검증 해결 · 2026-09-22

**L384 수정 커널은 기존 한도 5×10⁻⁶을 그대로 유지한 전체 검증을 통과했다.**

## 원인과 수정

단일 B7의 gate 미분에서 FP32 곱셈 결합 순서를 바꾼 것이 원인이었다.

```python
# 기존 계약: da는 mask 적용 후 gradient, p/g는 재계산 projection/sigmoid gate
# dg를 BF16으로 저장하기 전 곱셈 순서를 유지해야 한다.
dp_fp32 = da * g
# 수정 전
# dg = bf16((dp_fp32 * p) * (1 - g))
# 수정 후
dg = bf16(((da * p) * g) * (1 - g))
dp = bf16(dp_fp32)
```

두 식은 실수 산술에서 같지만 FP32 중간 반올림이 달라진다. 이것이 다음 BF16 반올림 경계에서 차이를 만들고 dX_n GEMM과 LN 파라미터 gradient에 전달됐다. 원래 곱셈 순서를 복원했다. 허용 오차, 저장 정책, 단일 B7의 융합 구성, 타일/CTA/ring 설정은 변경하지 않았다.

## 원인 분리 증거

진단 커널은 dX_n과 affine 전 정규화값 xhat을 내보냈다. 정규화값은 전 원소 일치했다. 수정 전 dX_n 불일치는 3개 case에서 각각 150/140/145개였고, 수정 후 모두 **0개**였다.
기존·새 구현 모두 자신의 dX_n/xhat을 FP64로 곱하고 합산한 참조에 대해 LN gradient 오차가 약2–3×10⁻⁷이었다. 따라서 문제는 LN reduction 자체가 아니라 앞단 gate 미분에서 유입된 값 차이였다.
FP64 검사는 이 중간값의 독립 합산 참조이며 전체 네트워크 FP64 autograd 비교라고 주장하지 않는다.

## 기존 실패 3case 재검증

| case | 수정 전 dgamma | 수정 후 dgamma | 수정 전 dbeta | 수정 후 dbeta |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

입력 LN gradient 최대 상대L2: **%.3e**, 기존 한도5e-6. forward와 입력 dX는 기존 계약 참조와 bit-exact. 가중치 gradient도 기존5e-4 한도를 통과했다. graph/eager 전 출력 bit-exact.

## 추가 검증

- 12개 입력 조건 × TriMul 단독/새 CUDA Transition 연결 블록 = 24개 조합.
- random 입력·가중치·upstream gradient, 작은 입력, 0인 LN scale 채널, 비정규 affine 값, mask/dropout 변경, 전체 mask0, 전체 dropout scale0 포함.
- forward0, dX2e-5, LN gradient5e-6, 나머지 gradient5e-4: 기존 한도 유지. 모든 조건 통과, graph/eager bit-exact.
- MiniPairformer 연결 LN gradient 최대 상대L2 **%.3e**.
- 동일 수정 cubin sanitizer: **%s**.
- 현재 개발 진입점: **%s**.

## 성능

같은 실행 job%s에서 수정 전/후를 교차 측정했다.

| 구간 | 수정 전 | 수정 후 |
|---|---:|---:|
| B7 단독 | %.3fµs | %.3fµs |
| TriMul fwd+bwd | %.3fµs | %.3fµs |

전체 시간 변화 **%+.2f%%**. 성능 개선을 주장하지 않으며, 정확도를 고치면서 기존 성능을 거의 유지했다.
별도 job%s에서 MiniPairformer 블록 학습 전체는 수정 전 %.3fms, 수정 후 %.3fms였다. 서로 다른 job의 절대 시간을 섞어 비율을 계산하지 않는다.

## 적용 범위

`runs/trimul_training_current.py:Training`의 **L384 개발 경로**에 수정된 단일 B7을 연결했다. 다른 길이는 기존 개발 경로를 유지한다. **L768의 새 단일 B7 및 엔진 production auto-dispatch 승격은 이 작업 범위가 아니다.**
이전에 "LN gradient 검증 미완료"라고 표시했던 L384 제한은 해결됐다. 과거 벤치 숫자는 수정 전 커널의 역사 기록으로 보존하며, 당시 실패 결과를 지우거나 통과로 바꾸지 않는다.

## 재현·근거

- [수정 커널](fixed/joint.cu), [현재 선택 policy](policy.py), [선택·SHA-256](selected.json)
- [원인 분리](diagnose.json), [기존 실패 case 전체 검증과 시간표본](validation.json)
- [12case·MiniPairformer 연결](stress_block.json)
- [현재 개발 진입점](../trimul_training_current.py)
- `sbatch runs/trimul_ln_gradient_20260922/validate.sbatch`
- `sbatch runs/trimul_ln_gradient_20260922/stress_block.sbatch`
- `sbatch runs/trimul_ln_gradient_20260922/sanitize.sbatch`
- `sbatch runs/trimul_ln_gradient_20260922/verify_entry.sbatch`
'''%(lnmax,blockmax,santext,entrytext,v['job'],v['times']['b7']['latest_combined']['median_us'],v['times']['b7']['fixed']['median_us'],old,new,100*(new/old-1),s['job'],s['times']['block']['original_joint']['median_us']/1000,s['times']['block']['fixed']['median_us']/1000)
(R/'README.md').write_text(text)
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TriMul LN gradient · 해결</title><style>body{max-width:1100px;margin:24px auto;padding:0 18px;font:16px/1.7 system-ui;color:#24364c;background:#f3f6fa}section{padding:20px;background:white;margin:16px 0;border-radius:10px}.pass{padding:16px;background:#dff5e8;border-left:5px solid #23804c}pre{padding:15px;background:#eef4f8;overflow:auto}a{color:#1469a2}table{width:100%;border-collapse:collapse}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}</style><h1>TriMul 입력 LN gradient · 엄격 검증 해결</h1><p class="pass">L384 기존 한도5×10⁻⁶ 유지 · 수정 후 최대 LN 상대L2 MAXLN · 실패3case와 추가24조합 통과</p><section><h2>원인: gate 미분의 곱셈 순서</h2><pre>수정 전: dg = bf16(((da × g) × p) × (1 − g))
수정 후: dg = bf16(((da × p) × g) × (1 − g))</pre><p>FP32 중간 반올림이 다음 BF16 경계에서 값 차이를 만들었습니다. LN 합산 자체는 정상이었습니다. 곱셈 순서를 원래 계약에 맞췄고, 단일 커널 융합·저장 정책·타일 설정은 유지했습니다.</p></section><section><h2>중간값으로 원인 검증</h2><p>정규화 xhat은 원래부터 정확히 일치했습니다. dX_n은 수정 전 150/140/145개 원소가 달랐고, 수정 후 3case 모두 불일치 0개입니다. FP64 합산 참조로 LN reduction의 오차가 약2–3×10⁻⁷임을 확인했습니다.</p></section><section><h2>검증·성능·적용</h2><p>기존 실패3case, 추가12입력 × TriMul/전체 블록, graph/eager 통과. 입력 dX도 원래 참조와 bit-exact.</p><p>SANITIZER</p><p>ENTRY</p><p>같은 실행 TriMul 학습: BEFORE → AFTER µs. 약0.24% 시간 증가로 기존 성능을 거의 유지했습니다.</p><p>L384 개발 진입점에 적용했습니다. L768 새 경로와 엔진 production auto-dispatch는 이번 검증 범위가 아닙니다.</p></section><p><a href="README.md">상세 검증·재현</a> · <a href="selected.json">선택·SHA-256</a> · <a href="validation.json">실패 case 재검증</a> · <a href="stress_block.json">전체 블록 검증</a> · <a href="../../TRIMUL_STATUS.html">현황</a></p></html>'''
for k,val in [('MAXLN','%.3e'%lnmax),('SANITIZER',santext),('ENTRY',entrytext),('BEFORE','%.3f'%old),('AFTER','%.3f'%new)]:page=page.replace(k,html.escape(val))
(R/'index.html').write_text(page)
print('LN_FIXED',lnmax,blockmax,'sanitizer',santext,'entry',entrytext)
