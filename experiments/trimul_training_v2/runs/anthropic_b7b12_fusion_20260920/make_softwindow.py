from pathlib import Path
import json
p=Path(__file__).resolve().parent
s=(p/'front_window256.cu').read_text();a=s.index('TMN_DEVI void window_sync');b=s.index('TMN_DEVI void weight_role',a)
helper='''TMN_DEVI void window_sync(const Params& p,int completed){
 int role=blockIdx.x<DWCOUNT?0:1,window=completed-1;
 allsync();if(threadIdx.x==0){atomicAdd(p.counts+2+window*2+role,1u);if(window>=WINDOW_GAP){unsigned want=role==0?DXCOUNT:DWCOUNT;while(atomicAdd(p.counts+2+(window-WINDOW_GAP)*2+(1-role),0u)<want)__nanosleep(32);}}allsync();
}
'''
s=s[:a]+helper+s[b:];s=s.replace('atomicExch(p.counts+2,0u);','for(int i=0;i<2*((p.tiles+WINDOW-1)/WINDOW);++i)atomicExch(p.counts+2+i,0u);')
s=s.replace('group=blockIdx.x/DW_SPLITS,split=blockIdx.x%DW_SPLITS','group=blockIdx.x%8,split=blockIdx.x/8')
for w in [64,128,256,512,1024]:
 for gap in [1,2]:
  v=s.replace('#define WINDOW 256','#define WINDOW %d\n#define WINDOW_GAP %d'%(w,gap));name='front_softw%d_g%d'%(w,gap);(p/(name+'.cu')).write_text(v);cf=json.loads((p/'front_window256.launch.json').read_text());cf['extra_counts']=2*((9216+w-1)//w);(p/(name+'.launch.json')).write_text(json.dumps(cf)+'\n')
