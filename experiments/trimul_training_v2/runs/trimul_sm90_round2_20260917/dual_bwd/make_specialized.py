from pathlib import Path
s=Path('baseline.py').read_text()
s=s.replace('self.mma_mgroups = min(num_warps // 4, BLOCK_M1 // 64)','self.mma_mgroups = 1').replace('self.mma_ngroups = num_warps // 4 // self.mma_mgroups','self.mma_ngroups = 1')
s=s.replace('cute.struct.MemRange[cutlass.Int64, 2 * self.stages]','cute.struct.MemRange[cutlass.Int64, 3 * self.stages]')
s=s.replace('for stage in cutlass.range_constexpr(2 * self.stages):','for stage in cutlass.range_constexpr(3 * self.stages):')
start=s.index('        thrg, thrf =')
end=s.index('    @cute.jit\n    def __call__',start)
old=s[start:end]
init=old[:old.index('        # Each stage')]
init=init.replace('mg.get_slice(tidx), mf.get_slice(tidx)','mg.get_slice(tidx % 128), mf.get_slice(tidx % 128)')
epilogue=old[old.index('        coords ='):]
epilogue=epilogue.replace('cute.arch.barrier()','cute.arch.barrier(barrier_id=1, number_of_threads=128)').replace('if warp == 0:','if warp == 4:').replace('.get_slice(tidx)','.get_slice(tidx % 128)')
epilogue=epilogue.replace('            so = storage.sg.get_tensor(lo.outer, swizzle=lo.inner)\n','')
body='''        so = storage.sg.get_tensor(lo.outer, swizzle=lo.inner)
        if warp < 4:
            cute.arch.setmaxregister_decrease(24)
            if warp == 0:
                for kind in cutlass.range_constexpr(2):
                    count = self.gate_tiles if kind == 0 else self.front_tiles
                    offset = 0 if kind == 0 else self.gate_tiles
                    for step in cutlass.range(count):
                        absolute = offset + step
                        stage = absolute % self.stages
                        if absolute >= self.stages:
                            cute.arch.mbarrier_wait(bar + 2 * self.stages + stage, ((absolute // self.stages) - 1) & 1)
                        full = bar + kind * self.stages + stage
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(full, (self.pm + self.bn) * self.bk * 2)
                        if cutlass.const_expr(kind == 0):
                            loadg(src_idx=step, dst_idx=stage, tma_bar_ptr=full)
                            loadw(src_idx=step, dst_idx=stage, tma_bar_ptr=full)
                        else:
                            loadf(src_idx=step, dst_idx=stage, tma_bar_ptr=full)
                            loadv(src_idx=step, dst_idx=stage, tma_bar_ptr=full)
        else:
            cute.arch.setmaxregister_increase(232)
'''
body+=''.join('    '+line+'\n' for line in init.rstrip().splitlines())
body+='''            for kind in cutlass.range_constexpr(2):
                count = self.gate_tiles if kind == 0 else self.front_tiles
                offset = 0 if kind == 0 else self.gate_tiles
                if cutlass.const_expr(kind == 0):
                    accg.fill(0)
                else:
                    accf.fill(0)
                for step in cutlass.range(count):
                    stage = (step + offset) % self.stages
                    cute.arch.mbarrier_wait(bar + kind * self.stages + stage, (step // self.stages) & 1)
                    cute.arch.fence_view_async_shared()
                    warpgroup.fence()
                    for k in cutlass.range_constexpr(cute.size(xg.shape[2])):
                        if cutlass.const_expr(kind == 0):
                            cute.gemm(mag, accg, xg[None,None,k,stage], wg[None,None,k,stage], accg)
                        else:
                            cute.gemm(maf, accf, xf[None,None,k,stage], vf[None,None,k,stage], accf)
                    warpgroup.commit_group()
                    warpgroup.wait_group(0)
                    cute.arch.barrier(barrier_id=1, number_of_threads=128)
                    if warp == 4:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(bar + 2 * self.stages + stage)
                if cutlass.const_expr(kind == 0):
                    gate_saved.store(accg.load().to(BFloat16))
'''
body+=''.join('    '+line+'\n' for line in epilogue.rstrip().splitlines())+'\n'
s=s[:start]+body+s[end:]
s=s.replace('    if bm < 64:','    if config["num_warps"] != 8 or bm != 64 or bn > 128:\n        return "bounded warp-specialized prototype requires BM64, BN<=128, 8 physical warps"\n    if bm < 64:')
s=s.replace('name="trimul_input_dual_bwd_sm90"','name="trimul_round2_specialized_sm90"')
Path('specialized.py').write_text(s)
