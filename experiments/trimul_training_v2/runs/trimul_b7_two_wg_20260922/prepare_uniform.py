from pathlib import Path
S=Path(__file__).resolve().parent
R=S.parent/'trimul_b7_two_wg_uniform_20260922'
for name in ('plan.py','single_wg.inc','precompile.py','probe.py','ncu_profile.py','sanitize.py','run.slurm','sanitize.slurm','profile.slurm','sweep.py'):
 (R/name).write_text((S/name).read_text().replace(S.name,R.name))
s=(S/'joint.cu').read_text().replace('if(sub==0&&valid)mbar_wait','if(sub==0)mbar_wait').replace('if(valid){uint8_t* ds=','{uint8_t* ds=').replace('if(valid)mbar_arrive(bar+10','mbar_arrive(bar+10')
(R/'joint.cu').write_text(s)
s=(S/'pair_producer.inc').read_text().replace('   if(!valid)continue;','')
s=s.replace('if((plane&1)==0&&lane<4)', 'if(valid&&(plane&1)==0&&lane<4)')
old='if(lane==0){mbar_arrive_expect_tx(bar+8+g*4+slot,16384);bulk_load(sm+g*32768+slot*16384,p.ring+(cid*RINGS+rs)*131072+phase*16384,16384,bar+8+g*4+slot);}'
new='''if(valid){
     if(lane==0){mbar_arrive_expect_tx(bar+8+g*4+slot,16384);bulk_load(sm+g*32768+slot*16384,p.ring+(cid*RINGS+rs)*131072+phase*16384,16384,bar+8+g*4+slot);}
    }else{
     for(int i=lane*16;i<16384;i+=512)*reinterpret_cast<uint4*>(sm+g*32768+slot*16384+i)=make_uint4(0,0,0,0);
     fence_proxy_async();__syncwarp();if(lane==0)mbar_arrive(bar+8+g*4+slot);
    }'''
assert old in s;s=s.replace(old,new)
s=s.replace('publish(flags+SOURCES+1,sequence+1);','if(valid)publish(flags+SOURCES+1,sequence+1);')
(R/'pair_producer.inc').write_text(s)
