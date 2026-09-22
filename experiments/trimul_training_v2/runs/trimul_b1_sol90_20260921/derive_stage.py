from pathlib import Path
P=Path(__file__).resolve().parent;R=P/'stage';R.mkdir(exist_ok=True)
for p in [*P.glob('*.cuh'),*P.glob('*.inc'),P/'b1_fused.cu',P/'replace_plan.py']:(R/p.name).write_bytes(p.read_bytes())
s=(R/'replace_plan.py').read_text().replace("OLD=R.parent/'trimul_split_bwd_20260921'","OLD=R.parent.parent/'trimul_split_bwd_20260921'").replace('torch.zeros(2+count,','torch.zeros(512+132*32,');(R/'replace_plan.py').write_text(s)
s=(R/'b1_fused.cu').read_text()
s=s.replace('TMN_DEVI void shared_role(','''TMN_DEVI void stage_tick(const Params& p,unsigned long long& last,int st){
 if(threadIdx.x==0){auto now=clock64();reinterpret_cast<unsigned long long*>(p.counts+512)[blockIdx.x*16+st]+=now-last;last=now;}
}
TMN_DEVI void shared_role(''')
s=s.replace('int wi=threadIdx.x/128,round=0,mi=0;','int wi=threadIdx.x/128,round=0,mi=0;unsigned long long last=clock64();')
s=s.replace('  prepare_shared(p,sm,bars,slot,tile*64,round&1,mask,mi);','  stage_tick(p,last,0);prepare_shared(p,sm,bars,slot,tile*64,round&1,mask,mi);stage_tick(p,last,1);')
s=s.replace('  if(threadIdx.x==0)tma_store_wait_all();allsync();','  if(threadIdx.x==0)tma_store_wait_all();allsync();stage_tick(p,last,2);')
s=s.replace('  lowreg_dgrad(p,sm,bars,tile*64,slot);allsync();','  lowreg_dgrad(p,sm,bars,tile*64,slot);allsync();stage_tick(p,last,3);')
s=s.replace(' shared_role(p,sm,bars);',' unsigned long long mainlast=clock64();shared_role(p,sm,bars);stage_tick(p,mainlast,4);')
s=s.replace(' gate_phase(p,sm,bars);',' stage_tick(p,mainlast,5);gate_phase(p,sm,bars);stage_tick(p,mainlast,6);')
s=s.replace(' for(int i=blockIdx.x*256+threadIdx.x;i<49664;i+=UCOUNT*256)reduce_at(p,i);',' stage_tick(p,mainlast,7);for(int i=blockIdx.x*256+threadIdx.x;i<49664;i+=UCOUNT*256)reduce_at(p,i);stage_tick(p,mainlast,8);')
(R/'b1_fused.cu').write_text(s)
