def candidates(v,d):
 c=dict(bk=(1<<(d-1).bit_length()) if v=='full_k' else 64,bn=32,bo=64,mgroups=2,ngroups=2,stages=1,min_blocks=1)
 yield c
 if v=='full_k' and d==384:yield dict(c,bk=384)
