from pathlib import Path
r=Path(__file__).resolve().parent
outs=','.join(f'%{i}' for i in range(32));args=','.join(f'"+f"(d[{i}])' for i in range(32))
head=f'''// SM90 BF16 WGMMA SS, both operands MN-major (transpose flags 1,1).
// Instruction signature follows NVIDIA CUTLASS MMA_64x64x16_F32BF16BF16_SS.
TMN_DEVI void mma_ss(float (&d)[32],uint64_t a,uint64_t b,int accumulate){{
 asm volatile("{{ .reg .pred p; setp.ne.b32 p, %34, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {{{outs}}}, %32, %33, p, 1, 1, 1, 1; }}"
 : {args} : "l"(a),"l"(b),"r"(accumulate));
}}
'''
s=(r/'fused.cu').read_text();a=s.index('TMN_DEVI void wgrad(');b=s.index('TMN_DEVI void wgrad_full',a);v=s[a:b].replace('void wgrad(','void wgrad_ss(')
v=v.replace('uint8_t *sy=sm,*sg=sm+8192,*sp=sm+16384,*sx=sm+24576,*sb=sm+32768;','uint8_t *sy=sm,*sg=sm+8192,*sp=sm+16384,*sx=sm+24576;')
a=v.index(' auto load_tile');b=v.index(' float* part=');old=v[a:b];transform=old[old.index(' auto transform_tile'):old.index(' if(PREFETCH && first<end)')]
# lambda takes a buffer; use local pointers to avoid reassignment hazards.
transform=transform.replace('auto transform_tile = [&](int it){','auto transform_tile = [&](int it,uint8_t* buf){uint8_t *sy=buf,*sg=buf+8192,*sp=buf+16384;')
loop=''' auto load_tile = [&](int it,uint8_t* buf,uint64_t* barrier){int row=it*64,cc=wg?nch:mch;
  if(tid==0){mbar_arrive_expect_tx(barrier,wg?32768:24576);tma_load_2d(buf,&p.dy,barrier,cc,row);tma_load_2d(buf+8192,&p.gate,barrier,cc,row);if(wg)tma_load_2d(buf+16384,&p.proj,barrier,cc,row);tma_load_2d(buf+24576,wg?&p.xn:&p.norm,barrier,wg?mch:nch,row);}
 };
'''+transform+'''
 if(WSS>=2){if(tid==0){mbar_init(bar+1,1);fence_barrier_init();}sync_group();}
 if(first<end)load_tile(first,sm,bar);
 for(int it=first;it<end;++it){
  int stage=WSS>=2?(it-first)&1:0;uint8_t* buf=sm+stage*32768;uint64_t* barrier=bar+stage;
  sync_group();mbar_wait(barrier,((it-first)/(WSS>=2?2:1))&1);
  transform_tile(it,buf);sync_group();fence_proxy_async();sync_group();
  uint8_t* sa=wg?buf+24576:buf;uint8_t* sb=wg?buf:buf+24576;
  fence_regs(acc);wgmma_fence();
  static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;
   uint64_t da=smem_desc(smem_u32(sa+k*2048),16,1024,1),db=smem_desc(smem_u32(sb+k*2048),16,1024,1);
   mma_ss(acc,da,db,it>first||k>0);
  });wgmma_commit();
  if(WSS>=2&&it+1<end)load_tile(it+1,sm+(1-stage)*32768,bar+1-stage);
  wgmma_wait<0>();fence_regs(acc);fence_proxy_async();sync_group();
  if(WSS==1&&it+1<end)load_tile(it+1,sm,bar);
 }
'''
v=v[:a]+loop+v[b:];(r/'wgrad_ss.cuh').write_text(head+v)
