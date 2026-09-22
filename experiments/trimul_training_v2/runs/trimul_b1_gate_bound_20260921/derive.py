from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'trimul_b1_epilogue_fixed_20260921'
for p in [*P.glob('*.cuh'),*P.glob('*.inc'),P/'b1_fused.cu',P/'replace_plan.py']:(R/p.name).write_bytes(p.read_bytes())
p=R/'b1_fused.cu';s=p.read_text()+'''
// Diagnostic only: same gate phase and CTA shared-memory reservation.
// This does not implement full B1 and is never a dispatch candidate.
extern "C" __global__ __launch_bounds__(256,1) void b1_gate_only(__grid_constant__ const Params p){
 extern __shared__ __align__(1024) uint8_t sm[];__shared__ uint64_t bars[2];
 if(threadIdx.x==0){for(int i=0;i<2;++i)mbar_init(bars+i,1);fence_barrier_init();}allsync();
 gate_phase(p,sm,bars);
}
''';p.write_text(s)
p=R/'replace_plan.py';s=p.read_text().replace("u.kernel('b1_fused')","u.kernel('b1_gate_only')");p.write_text(s)
