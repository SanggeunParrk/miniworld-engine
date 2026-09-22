from pathlib import Path
R=Path(__file__).resolve().parent
D=R.parent/'trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/trimul_inproj/cuda'
s=(D/'anthropic_saved_front.cu').read_text()
s=s.replace('// Prenormalized x_n: input LN and its saves stay in their existing kernel.','for (int i=tid;i<CZ;i+=G::NTHR) { sGamma[i]=p.gamma[i]; sBeta[i]=p.beta[i]; }')
s=s.replace('true,0,false>(p)','true,1,true,true>(p)')
# Store statistics in existing separate contiguous arrays without a reorder kernel.
s=s.replace('p.stats[2 * ((size_t)iA * p.N + jA)]','p.stats[(size_t)iA * p.N + jA]').replace('p.stats[2 * ((size_t)iA * p.N + jA) + 1]','p.stats[(size_t)p.N*p.N + (size_t)iA * p.N + jA]')
s=s.replace('p.stats[2 * ((size_t)iB * p.N + jB)]','p.stats[(size_t)iB * p.N + jB]').replace('p.stats[2 * ((size_t)iB * p.N + jB) + 1]','p.stats[(size_t)p.N*p.N + (size_t)iB * p.N + jB]')
s=s.replace('// Changes: consume existing normalized input (LNM=0), preserve original training','// Experiment: restore original ln_fragment; save normalized input + mean/rstd.\n// Preserve original training')
(R/'fused_front.cu').write_text(s)
