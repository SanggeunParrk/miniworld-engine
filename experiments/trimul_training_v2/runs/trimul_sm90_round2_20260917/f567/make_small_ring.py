from pathlib import Path
s=Path('projection_prefetch_initial.py').read_text()
s=s.replace('trimul_parity_f567_prefetch_initial_sm90','trimul_parity_f567_smallring_sm90')
a=s.index('        first_m = (pid // (self.group_m * nn))')
b=s.index('        warp_idx =',a)
s=s[:a]+'''        first_m = (pid // nn) * self.group_m
        n_block = pid % nn
'''+s[b:]
mark='        mbar_full_ptr = storage.mbar_full.data_ptr()'
s=s.replace(mark,'''        sP_base = storage.sX1.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        sG_base = storage.sW1.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        at = max(cute.cosize(sX1_layout), cute.cosize(sX2_layout)) // cute.cosize(sO_layout)
        bt = max(cute.cosize(sW1_layout), cute.cosize(sW2_layout)) // cute.cosize(sO_layout)
        has_ds = cutlass.const_expr(at >= 1 and bt >= 1 and max(at, bt) >= 2 and at + bt >= 4 and self.L % TILE_M == 0 and self.N % TILE_N == 0)
'''+mark)
s=s.replace('sP = storage.sX1.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)','sP = sP_base').replace('sG = storage.sW1.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)','sG = sG_base')
s=s.replace('mbar_full_ptr + stage, Int32((k // STAGES) % 2)', 'mbar_full_ptr + stage, Int32((iteration * (cute.ceil_div(G_LOOP - stage, STAGES) + (1 if has_ds and stage == 0 else 0)) + k // STAGES) % 2)')
s=s.replace('mbar_full_ptr + STAGES + stage, Int32((k // STAGES) % 2)', 'mbar_full_ptr + STAGES + stage, Int32((iteration * cute.ceil_div(P_LOOP - stage, STAGES) + k // STAGES) % 2)')
s=s.replace('mbar_full_ptr + 2 * STAGES, Int32(0)', 'mbar_full_ptr + 2 * STAGES, Int32(iteration % 2)')
s=s.replace('Int32(cute.ceil_div(G_LOOP, STAGES) % 2)', 'Int32((iteration * (cute.ceil_div(G_LOOP, STAGES) + 1) + cute.ceil_div(G_LOOP, STAGES)) % 2)')
a=s.index('        gX1 = cute.local_tile')
b=s.index('\n    @cute.jit',a)
body=s[a:b]
s=s[:a]+'        for iteration in cutlass.range(min(self.group_m, nm - first_m), unroll=1):\n            m_block = first_m + iteration\n'+''.join('    '+line if line.strip() else line for line in body.splitlines(True))+s[b:]
s=s.replace('m_blocks = cute.ceil_div(M, TILE_M) * cute.ceil_div(self.N, TILE_N)', 'm_blocks = cute.ceil_div(cute.ceil_div(M, TILE_M), self.group_m) * cute.ceil_div(self.N, TILE_N)')
# Ensure a unique opaque registration regardless prior generated spelling.
s=s.replace('projection_prefetch_initial','small_ring')
Path('small_ring.py').write_text(s)
