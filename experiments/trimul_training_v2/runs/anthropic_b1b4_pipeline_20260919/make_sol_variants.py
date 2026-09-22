"""Isolated B1-B4 experiments based on the validated balanced kernel."""
from pathlib import Path

root = Path(__file__).resolve().parent
base = (root / 'dual_balanced.cu').read_text()

def soa(s):
    old = '''if(lane%4==0){stats[wi*128+ra*2]=s1[0];stats[wi*128+ra*2+1]=s2[0];stats[wi*128+rb*2]=s1[1];stats[wi*128+rb*2+1]=s2[1];}'''
    new = '''if(lane%4==0){stats[wi*128+ra]=s1[0];stats[wi*128+64+ra]=s2[0];stats[wi*128+rb]=s1[1];stats[wi*128+64+rb]=s2[1];}'''
    assert old in s
    s = s.replace(old, new)
    s = s.replace('stats[ra*2]+stats[128+ra*2]', 'stats[ra]+stats[128+ra]')
    s = s.replace('stats[rb*2]+stats[128+rb*2]', 'stats[rb]+stats[128+rb]')
    s = s.replace('stats[ra*2+1]+stats[128+ra*2+1]', 'stats[64+ra]+stats[192+ra]')
    return s.replace('stats[rb*2+1]+stats[128+rb*2+1]', 'stats[64+rb]+stats[192+rb]')

def direct(s):
    start = s.index(' float *mus=stats+256')
    end = s.index(' uint32_t fx[2][4][4]', start)
    s = s[:start] + ''' // Read saved row statistics directly; gamma is published once per CTA.
 float* gam=reinterpret_cast<float*>(sm+73728);
 int ra=w*16+lane/4,rb=ra+8;
 float mu[2]={p.mean[m0+ra],p.mean[m0+rb]};
 float rs[2]={p.rs[m0+ra],p.rs[m0+rb]},s1[2]={},s2[2]={};
''' + s[end:]
    old = ' // Publish initialized slot barriers and LN sums before issuing local TMA.'
    assert old in s
    return s.replace(old, ''' // DX-only persistent gamma: 73728..74752 never overlaps either input slot.
 if(!dw)reinterpret_cast<float*>(sm+73728)[threadIdx.x]=p.gamma[threadIdx.x];
''' + old)

for name, source in [('dual_stats_soa', soa(base)),
                     ('dual_ln_direct', direct(base)),
                     ('dual_ln_direct_soa', soa(direct(base)))]:
    (root / (name + '.cu')).write_text('// Experiment: ' + name + '\n' + source)
    (root / (name + '.py')).write_text(
        'from dual_experiment import Experiment\n'
        'class Plan(Experiment):\n'
        '    def __init__(self, d, dy, saved, count=132, part=2):\n'
        f'        super().__init__(d, dy, saved, count=count, part=part, source={name!r})\n')
