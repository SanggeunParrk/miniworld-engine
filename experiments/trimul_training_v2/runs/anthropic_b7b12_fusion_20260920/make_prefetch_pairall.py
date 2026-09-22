from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'front_prefetch_lnpair.cu').read_text()
a=s.index('    float xa=');b=s.index('    float ha=',a)
s=s[:a]+'''    uint32_t xp=pair_get(lnsm+wi*8192,r,c);
    float xa=__fmul_rn(__fsub_rn(bf16lo(xp),mu[rr]),rs[rr]),xb=__fmul_rn(__fsub_rn(bf16hi(xp),mu[rr]),rs[rr]);
'''+s[b:]
a=s.index('    float xaa=');b=s.index('    float da=',a)
s=s[:a]+'''    uint32_t xap=pair_get(lnsm+wi*8192,ra,c),xbp=pair_get(lnsm+wi*8192,rb,c);
    float xaa=__fmul_rn(__fsub_rn(bf16lo(xap),mu[0]),rs[0]),xab=__fmul_rn(__fsub_rn(bf16hi(xap),mu[0]),rs[0]);
    float xba=__fmul_rn(__fsub_rn(bf16lo(xbp),mu[1]),rs[1]),xbb=__fmul_rn(__fsub_rn(bf16hi(xbp),mu[1]),rs[1]);
'''+s[b:]
name='front_prefetch_lnpairall';(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/'front_kindprefetch.launch.json').read_text())
