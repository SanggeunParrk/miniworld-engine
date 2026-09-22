from pathlib import Path
r=Path(__file__).resolve().parent
base=(r/'dual_balanced.cu').read_text()

def shared_acc(s):
    s=s.replace('tmp[w*512+c]=dga;tmp[w*512+c+1]=dgb;tmp[w*512+256+c]=dba0;tmp[w*512+256+c+1]=dbb0;',
                'tmp[w*512+c]+=dga;tmp[w*512+c+1]+=dgb;tmp[w*512+256+c]+=dba0;tmp[w*512+256+c+1]+=dbb0;')
    start=s.index(' allsync(); // Publish all four warps')
    end=s.index(' fence_proxy_async();sync_group();',start)
    s=s[:start]+s[end:]
    old=' for(int j=threadIdx.x;j<512;j+=256)p.partln[split*512+j]=reinterpret_cast<float*>(sm+229376)[j];'
    assert old in s
    s=s.replace(old,''' // Every warp owns its own parameter partials across the input loop.
 allsync();
 float* tmp=reinterpret_cast<float*>(sm+65536);int c=threadIdx.x;
 p.partln[split*512+c]=(tmp[c]+tmp[512+c])+(tmp[1024+c]+tmp[1536+c]);
 p.partln[split*512+256+c]=(tmp[256+c]+tmp[768+c])+(tmp[1280+c]+tmp[1792+c]);''')
    return s.replace(' // Publish initialized slot barriers', ''' if(!dw)for(int j=threadIdx.x;j<2048;j+=256)reinterpret_cast<float*>(sm+65536)[j]=0;
 // Publish initialized slot barriers''')

def cache_xh(s):
    s=s.replace('uint32_t fx[2][4][4],dn[2][4][4];','uint32_t fx[2][4][4],dn[2][4][4];float xh[2][4][8];')
    needle='    float ha=bf16lo(dn[nlocal][q][j])*gam[c]'
    assert needle in s
    s=s.replace(needle,'    xh[nlocal][q][j*2]=xa;xh[nlocal][q][j*2+1]=xb;\n'+needle)
    start=s.index('    float xaa=__fmul_rn')
    end=s.index('    float daa=',start)
    return s[:start]+'''    float xaa=xh[nl][q][j*2],xab=xh[nl][q][j*2+1];
    float xba=xh[nl][q][(j+1)*2],xbb=xh[nl][q][(j+1)*2+1];
'''+s[end:]

for name,s in [('dual_ln_shared_acc',shared_acc(base)),
               ('dual_ln_cachexh',cache_xh(base)),
               ('dual_ln_shared_xh',cache_xh(shared_acc(base)))]:
    (r/(name+'.cu')).write_text('// Experiment: '+name+'\n'+s)
    (r/(name+'.py')).write_text('from dual_experiment import Experiment\nclass Plan(Experiment):\n    def __init__(self,d,dy,saved,count=132,part=2):\n        super().__init__(d,dy,saved,count,part,'+repr(name)+')\n')
