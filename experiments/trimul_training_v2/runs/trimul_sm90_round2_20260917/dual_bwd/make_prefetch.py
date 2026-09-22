from pathlib import Path
base=Path('baseline.py').read_text()
for factor in (1,2):
 s=base.replace('loadf, _, _ = copy_utils.tma_get_copy_fn','loadf, _, f_prefetch_src = copy_utils.tma_get_copy_fn')
 s=s.replace('        thrg, thrf =',f'''        if warp == 0:
            for pre in cutlass.range_constexpr(self.stages, min({factor+1} * self.stages, self.front_tiles)):
                cute.prefetch(af, f_prefetch_src[None, pre])
        thrg, thrf =''')
 s=s.replace('                warpgroup.commit_group()\n                warpgroup.wait_group(0)',f'''                warpgroup.commit_group()
                if cutlass.const_expr(kind == 1):
                    future = step + {factor+1} * self.stages
                    if warp == 0 and future < self.front_tiles:
                        cute.prefetch(af, f_prefetch_src[None, future])
                warpgroup.wait_group(0)''')
 s=s.replace('name="trimul_input_dual_bwd_sm90"',f'name="trimul_round2_prefetch{factor}_sm90"')
 Path(f'prefetch{factor}.py').write_text(s)
