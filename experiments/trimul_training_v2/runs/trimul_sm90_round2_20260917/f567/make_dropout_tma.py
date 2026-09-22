from pathlib import Path
r=Path.cwd();s=Path('/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/trimul_inproj/cute/parity_f567.py').read_text()
s=s.replace('        atomR: cute.CopyAtom,','        atomR: cute.CopyAtom,\n        atomD: cute.CopyAtom,\n        tD_tma: cute.Tensor,',1)
s=s.replace('                cpasync.prefetch_descriptor(atomR)','                cpasync.prefetch_descriptor(atomR)\n                cpasync.prefetch_descriptor(atomD)',1)
anchor='''            store_op = sm90h.get_smem_store_op(LayoutEnum.ROW_MAJOR, BFloat16, Float32)'''
pre='''            a_tiles = max(cute.cosize(sX1_layout), cute.cosize(sX2_layout)) // cute.cosize(sO_layout)
            b_tiles = max(cute.cosize(sW1_layout), cute.cosize(sW2_layout)) // cute.cosize(sO_layout)
            use_tma_ds = cutlass.const_expr(self.L % TILE_M == 0 and self.N % TILE_N == 0 and a_tiles + b_tiles >= 4)
            if cutlass.const_expr(use_tma_ds):
                if cutlass.const_expr(a_tiles >= 3):
                    sD = cute.make_tensor(sP.iterator + 2 * cute.cosize(sO_layout), sO_layout.outer)
                elif cutlass.const_expr(a_tiles >= 2 and b_tiles >= 2):
                    sD = cute.make_tensor(sG.iterator + cute.cosize(sO_layout), sO_layout.outer)
                else:
                    sD = cute.make_tensor(sG.iterator + 2 * cute.cosize(sO_layout), sO_layout.outer)
                gD_tma = cute.local_tile(tD_tma, (TILE_M, TILE_N), (m_block % (self.L // TILE_M), n_block))
                load_D, _, _ = quack_copy.tma_get_copy_fn(atomD, 0, cute.make_layout(1), gD_tma, sD, single_stage=True)
'''
s=s.replace(anchor,pre+anchor,1)
s=s.replace('''            cute.arch.barrier()
            cute.copy(store_op, copyC.retile(pfrag), copyC.partition_D(sP))''','''            cute.arch.barrier()
            if cutlass.const_expr(use_tma_ds):
                if warp_idx == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar_full_ptr, TILE_M * TILE_N * 2)
                    load_D(tma_bar_ptr=mbar_full_ptr)
            cute.copy(store_op, copyC.retile(pfrag), copyC.partition_D(sP))''',1)
s=s.replace('''            cg0 = copyC.retile(acc_G)''','''            if cutlass.const_expr(use_tma_ds):
                cute.arch.mbarrier_wait(mbar_full_ptr, Int32(cute.ceil_div(G_LOOP, STAGES) % 2))
            cg0 = copyC.retile(acc_G)''',1)
s=s.replace('''            if cutlass.const_expr(self.L % TILE_M == 0 and self.N % TILE_N == 0):
                gD =''','''            if cutlass.const_expr(use_tma_ds):
                cd0 = load_c.partition_S(sD)
                cd = cute.group_modes(cd0, 1, cute.rank(cd0))
            elif cutlass.const_expr(self.L % TILE_M == 0 and self.N % TILE_N == 0):
                gD =''',1)
s=s.replace('''                if cutlass.const_expr(self.L % TILE_M == 0 and self.N % TILE_N == 0):
                    cute.copy(global_atom, cd[None, epi_idx], dc)''','''                if cutlass.const_expr(use_tma_ds):
                    cute.copy(load_atom, cd[None, epi_idx], dc)
                elif cutlass.const_expr(self.L % TILE_M == 0 and self.N % TILE_N == 0):
                    cute.copy(global_atom, cd[None, epi_idx], dc)''',1)
s=s.replace('''        tx_bytes_total = (TILE_M + TILE_N) * TILE_K * 2''','''        atomD, tD_tma = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mD, sO_layout, (TILE_M, TILE_N))
        tx_bytes_total = (TILE_M + TILE_N) * TILE_K * 2''',1)
s=s.replace('''            atomR,
            sO_layout,''','''            atomR,
            atomD,
            tD_tma,
            sO_layout,''',1)
s=s.replace('name="trimul_parity_f567_sm90"','name="trimul_round2_dropout_tma"')
(r/'dropout_tma.py').write_text(s)
