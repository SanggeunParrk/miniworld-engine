from pathlib import Path
R=Path(__file__).resolve().parent;E=R.parent/'trimul_sm90_parity_20260917/engine';S=E/'src/miniworld_engine/kernels/trimul_inproj/cuda'
s=(S/'anthropic_saved_front.cu').read_text().replace('CUtensorMap tm_gate,tm_proj;','CUtensorMap tm_gate,tm_proj; float* rstd;').replace('sizeof(SavedFrontParams)==768','sizeof(SavedFrontParams)==832')
s=s.replace('// Prenormalized x_n: input LN and its saves stay in their existing kernel.','for (int i = tid; i < CZ; i += G::NTHR) { sGamma[i] = p.gamma[i]; sBeta[i] = p.beta[i]; }')
s=s.replace('p.stats[2 * ((size_t)iA * p.N + jA)]','p.stats[(size_t)iA * p.N + jA]').replace('p.stats[2 * ((size_t)iA * p.N + jA) + 1]','tp.rstd[(size_t)iA * p.N + jA]')
s=s.replace('p.stats[2 * ((size_t)iB * p.N + jB)]','p.stats[(size_t)iB * p.N + jB]').replace('p.stats[2 * ((size_t)iB * p.N + jB) + 1]','tp.rstd[(size_t)iB * p.N + jB]')
s=s.replace('SavedFrontCfg,true,0,false>','SavedFrontCfg,true,MW_FUSED,MW_FUSED,MW_FUSED>')
(R/'front.cu').write_text(s)
s=(R.parent/'anthropic_ln_inference_20260919/separate_ln_tma.cu').read_text();pos=s.rfind('\n}')
s=s[:pos]+'''\n if(q==0){if(ra<p.m){p.mean[ra]=st.mA;p.rstd[ra]=st.rA;}if(rb<p.m){p.mean[rb]=st.mB;p.rstd[rb]=st.rB;}}\n'''+s[pos:]
(R/'ln.cu').write_text(s)
