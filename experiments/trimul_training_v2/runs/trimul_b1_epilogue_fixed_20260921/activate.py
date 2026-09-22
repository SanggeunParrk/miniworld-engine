from pathlib import Path
import json
R=Path(__file__).resolve().parent;root=R.parents[1]
s=json.loads((R/'summary.json').read_text());assert s['verification_complete'] and not s['production_ready']
p=root/'runs/trimul_training_current.py';s=p.read_text().replace('trimul_b1_sol90_selected_20260921/sol_policy.py','trimul_b1_epilogue_fixed_20260921/epilogue_policy.py').replace('trimul_b1_sol90_selected_20260921/README.md','trimul_b1_epilogue_fixed_20260921/README.md');p.write_text(s)
p=root/'runs/trimul_training_selection.json';s=json.loads(p.read_text());s['implementation']='runs/trimul_b1_epilogue_fixed_20260921/epilogue_policy.py:Training';s['validation']='runs/trimul_b1_epilogue_fixed_20260921/summary.json';s['b1_implementation'].update(param_reduction_shared=True,param_reduction_vectorized=False,dtri_store_overlaps_param_reduction=True,grid_barrier_before_gate=False,cta_barrier_before_raw_prefetch=True);s['config_by_length']={str(n):json.loads((R/('selected-L%d.json'%n)).read_text())['config'] for n in (384,768)};p.write_text(json.dumps(s,indent=2)+'\n')
p=root/'scripts/render_trimul_current.py';s=p.read_text().replace("ACTIVE=ROOT/'runs/trimul_b1_sol90_selected_20260921'","ACTIVE=ROOT/'runs/trimul_b1_epilogue_fixed_20260921'")
s=s.replace("[ACTIVE/k for k in ('sol_policy.py'","[ACTIVE/k for k in ('epilogue_policy.py'")
s=s.replace('trimul_b1_sol90_selected/{sol_policy.py,b1_fused.cu,lowreg_stats.inc,results-L*.json}','trimul_b1_epilogue_fixed/{epilogue_policy.py,b1_fused.cu,lowreg_stats.inc,results-L*.json}')
s=s.replace('같은 CTA들이 Phase A → grid barrier → Phase B → 최종 reduction을 순서대로 수행.','각 CTA는 Phase A → local fence → Phase B. 마지막 grid barrier 뒤 전체 partial 합산.')
s=s.replace('# SAME kernel launch, after a grid barrier.','# SAME launch: each CTA reads only its own Phase A rows.')
s=s.replace('B4: 재구성 x̂와 저장 rstd로 LN 미분','B4: LN 미분 + γ/β shared 합산')
s=s.replace('다음 tri 로드 ↔ 현재 LN 미분/쓰기','dTri TMA 쓰기 ↔ γ/β 합산')
s=s.replace('같은 호출 내부 · grid barrier · dGate의 global WRITE → READ','같은 호출 내부 · CTA별 Phase 전환 · dGate의 global WRITE → READ')
s=s.replace("  'x_n도 Phase B가 다시 읽음. 이 경계는 shared memory 전달이 아니라 global/L2 재읽기다.',", "  'x_n도 global/L2 재읽기. 각 CTA의 소유 행만 소비하므로 이 경계에 전체 grid barrier는 없음.',\n  '마지막에 모든 CTA의 parameter partial을 합치기 전에는 전체 grid barrier를 유지한다.',")
s=s.replace('shared_b1_tri_stats_split_prefetch_paired_dnorm_registers','shared_b1_tri_stats_param_reduce_async_store_local_gate');p.write_text(s)
print('Activated verified development candidate')
