"""Bounded first tuning sweep; full public candidate grid remains in variants.py."""
def seeds(variant,d):
    bk=(1<<(d-1).bit_length()) if variant=='full_k' else 64
    ng=2 if d>=384 else 1
    base=dict(bk=bk,bn=32,bo=64,mgroups=1,ngroups=ng,stages=2,min_blocks=1)
    rows=[dict(base,bn=bn,stages=s) for bn in (32,64,128) for s in (1,2,3)]
    rows += [dict(base,bn=64,bo=128),dict(base,mgroups=2),dict(base,bn=64,ngroups=3-ng)]
    if variant=='streamed_k':
        rows += [dict(base,bk=k,bn=64) for k in (128,256) if k<d]
    elif d==384:
        # CuTe C++ can tile the exact width; Triton arange required padding to 512.
        rows += [dict(base,bk=384,bn=bn,stages=s) for bn in (32,64) for s in (1,2)]
    out=[]
    for c in rows:
        if c not in out:out.append(c)
    return out

def smem(variant,d,c,backward=False):
    g=1 if backward else c['ngroups'];bm=64*c['mgroups'];s=c['stages'];bk=c['bk'];bn=c['bn'];bo=c['bo']
    pad=((d+g*bo-1)//(g*bo))*bo*g
    elems=(1 if variant=='full_k' else s)*bm*bk+2*s*bn*bk
    elems+=(4 if backward else (0 if g==1 or bk>=bm else 1))*bm*bn
    if not backward:elems+=pad*bn
    return max(2*elems,0 if backward else 2*bm*pad)+16*(s+2)
