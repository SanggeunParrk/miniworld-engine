from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'front_kindunroll8_inter.cu').read_text();a=s.index('   for(int h=0;h<4;++h){int slot=h&1;');b=s.index('\n   load_p(',a)
g=r'''   for(int base=0;base<4;base+=2){fence_regs(acc);wgmma_fence();
    static_for<2>([&](auto si){constexpr int slot=decltype(si)::value;int h=base+slot;uint8_t* s=sm+slot*DX_SLOT;mbar_wait(bar+slot,1^(h/2)^slot);glu_small(p,s,s+16384,sm+81920+h*8192,row);
     static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_input64(acc,smem_desc(smem_u32(s+16384+k*2048),16,1024,1),smem_desc(smem_u32(s+24576+wi*8192+k*32),16,1024,1),side>0||h>0||k>0);});wgmma_commit();
    });wgmma_wait<0>();fence_regs(acc);allsync();if(base==0){load_g(p,sm,bar,row,side,2);load_g(p,sm,bar,row,side,3);}
   }'''
s=s[:a]+g+s[b:];a=s.index('   for(int h=0;h<4;++h){int slot=h&1;',a);b=s.index('\n  }\n  // B10',a)
g=r'''   for(int base=0;base<4;base+=2){fence_regs(acc);wgmma_fence();
    static_for<2>([&](auto si){constexpr int slot=decltype(si)::value;int h=base+slot;uint8_t* s=sm+slot*DX_SLOT;mbar_wait(bar+slot,1^(h/2)^slot);
     static_for<4>([&](auto ki){constexpr int k=decltype(ki)::value;mma_input64(acc,smem_desc(smem_u32(sm+81920+h*8192+k*2048),16,1024,1),smem_desc(smem_u32(s+wi*8192+k*32),16,1024,1),1);});wgmma_commit();
    });wgmma_wait<0>();fence_regs(acc);allsync();if(base==0){load_p(p,sm,bar,side,2);load_p(p,sm,bar,side,3);}
   }'''
s=s[:a]+g+s[b:];(p/'front_kindpair.cu').write_text(s);(p/'front_kindpair.launch.json').write_text((p/'front_kindunroll8_inter.launch.json').read_text())
