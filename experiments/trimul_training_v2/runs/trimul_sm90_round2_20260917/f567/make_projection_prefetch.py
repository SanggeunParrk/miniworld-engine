from pathlib import Path
r=Path.cwd();s=(r/'dropout_tma.py').read_text().replace('load_X2, _, _ = quack_copy.tma_get_copy_fn','load_X2, _, prefetch_X2 = quack_copy.tma_get_copy_fn',1)
pos=s.index('\n        thr_mma =')
s=s[:pos]+'''            for future in cutlass.range_constexpr(min(STAGES, P_LOOP)):
                cute.prefetch(tma_atom_X2, prefetch_X2[None, future])
'''+s[pos:]
s=s.replace('trimul_round2_dropout_tma','trimul_round2_projection_prefetch_initial');(r/'projection_prefetch_initial.py').write_text(s)
g=s.replace('''            warpgroup.commit_group()
            if cutlass.const_expr(k + STAGES < G_LOOP):''','''            warpgroup.commit_group()
            if cutlass.const_expr(STAGES + k < P_LOOP):
                if warp_idx == 0:
                    cute.prefetch(tma_atom_X2, prefetch_X2[None, STAGES + k])
            if cutlass.const_expr(k + STAGES < G_LOOP):''',1)
g=g.replace('''            warpgroup.commit_group()
            if cutlass.const_expr(k + STAGES < P_LOOP or k == P_LOOP - 1):''','''            warpgroup.commit_group()
            if cutlass.const_expr(min(P_LOOP, STAGES + G_LOOP) + k < P_LOOP):
                if warp_idx == 0:
                    cute.prefetch(tma_atom_X2, prefetch_X2[None, min(P_LOOP, STAGES + G_LOOP) + k])
            if cutlass.const_expr(k + STAGES < P_LOOP or k == P_LOOP - 1):''',1)
g=g.replace('trimul_round2_projection_prefetch_initial','trimul_round2_projection_prefetch_staged');(r/'projection_prefetch_staged.py').write_text(g)
for variant in ('initial','staged'):
 b=(r/'bench_dropout_tma.py').read_text().replace('import dropout_tma as mod',f'import projection_prefetch_{variant} as mod').replace('from miniworld_engine.kernels.trimul_inproj.cute import parity_f567 as base','import dropout_tma as base').replace('dropout_tma.json',f'projection_prefetch_{variant}.json')
 (r/f'bench_projection_prefetch_{variant}.py').write_text(b)
