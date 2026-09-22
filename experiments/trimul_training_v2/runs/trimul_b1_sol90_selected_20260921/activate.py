from pathlib import Path
import json
R=Path(__file__).resolve().parent;root=R.parents[1]
s=json.loads((R/'summary.json').read_text());assert s['verification_complete'] and not s['production_ready']
p=root/'runs/trimul_training_current.py';t=p.read_text().replace('trimul_b1_tri_opt_20260921/opt_policy.py','trimul_b1_sol90_selected_20260921/sol_policy.py').replace('trimul_b1_tri_opt_20260921/README.md','trimul_b1_sol90_selected_20260921/README.md');p.write_text(t)
p=root/'runs/trimul_training_selection.json';s=json.loads(p.read_text());s['implementation']='runs/trimul_b1_sol90_selected_20260921/sol_policy.py:Training';s['validation']='runs/trimul_b1_sol90_selected_20260921/summary.json';s['b1_implementation'].update(split_dnorm_prefetch=True,affine_both_warpgroups=True,pair_weight_gemm=True,pair_dnorm_gemm=True,dnorm_bf16_registers=True);s['config_by_length']={str(n):json.loads((R/('selected-L%d.json'%n)).read_text())['config'] for n in (384,768)};p.write_text(json.dumps(s,indent=2)+'\n')
p=root/'scripts/render_trimul_current.py';s=p.read_text().replace("ACTIVE=ROOT/'runs/trimul_b1_tri_opt_20260921'","ACTIVE=ROOT/'runs/trimul_b1_sol90_selected_20260921'")
s=s.replace("[ACTIVE/k for k in ('opt_policy.py'","[ACTIVE/k for k in ('sol_policy.py'")
s=s.replace('trimul_b1_tri_opt/{opt_policy.py,b1_fused.cu,lowreg_stats.inc,results-L*.json}','trimul_b1_sol90_selected/{sol_policy.py,b1_fused.cu,lowreg_stats.inc,results-L*.json}')
s=s.replace('dNorm = dProj @ Wp  # BF16 shared tile in CUDA','dNorm = dProj @ Wp  # BF16 register fragments')
s=s.replace('# Prefetch next tri/x_n/stats during dTri TMA store.','# Prefetch next tri/x_n/stats during LN backward.')
s=s.replace('dNorm GEMM: dProj 제자리 읽기','dNorm: 묶음 WGMMA → BF16 레지스터')
s=s.replace('다음 tri 로드 ↔ 현재 dTri 쓰기','다음 tri 로드 ↔ 현재 LN 미분/쓰기')
s=s.replace('(tri−저장 μ)×rstd → x̂ → affine','(tri−μ)×rstd → affine · 두 그룹 분담')
s=s.replace('shared_b1_tri_stats_direct_dproj_early_prefetch','shared_b1_tri_stats_split_prefetch_paired_dnorm_registers');p.write_text(s)
print('Activated verified development candidate; production dispatch unchanged')
