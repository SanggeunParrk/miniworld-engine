from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'front_cached_tail12_r40_216.cu').read_text()
s=s.replace('*reinterpret_cast<uint32_t*>(s+32768+swz128(c,r*2))=g;', '''*reinterpret_cast<uint32_t*>(s+32768+swz128(c,r*2))=g;
#if DEBUG_SAVE
   int gc=(group/2)*512+(group%2)*128+c;*reinterpret_cast<uint32_t*>(p.debugdc+(size_t)gc*p.M+row+r)=g;*reinterpret_cast<uint32_t*>(p.debugdc+(size_t)(gc+256)*p.M+row+r)=pp[q];
#endif''')
s=s.replace('wgmma_wait<0>();fence_regs(gate);static_for<16>', '''wgmma_wait<0>();fence_regs(gate);
#if DEBUG_SAVE
   static_for<8>([&](auto qi){constexpr int q=decltype(qi)::value;int rr=q*8+2*(lane%4),cc=wi*64+cr;p.debugxn[(size_t)(row+rr)*128+cc]=__float2bfloat16_rn(gate[q*4]);p.debugxn[(size_t)(row+rr+1)*128+cc]=__float2bfloat16_rn(gate[q*4+1]);p.debugxn[(size_t)(row+rr)*128+cc+8]=__float2bfloat16_rn(gate[q*4+2]);p.debugxn[(size_t)(row+rr+1)*128+cc+8]=__float2bfloat16_rn(gate[q*4+3]);});
#endif
static_for<16>''')
s=s.replace('put(out,rr,cr,acc[q*4]);', '''
#if DEBUG_SAVE
   p.partw[(size_t)(row+rr)*128+wi*64+cr]=acc[q*4];p.partw[(size_t)(row+rr+1)*128+wi*64+cr]=acc[q*4+1];p.partw[(size_t)(row+rr)*128+wi*64+cr+8]=acc[q*4+2];p.partw[(size_t)(row+rr+1)*128+wi*64+cr+8]=acc[q*4+3];
#endif
   put(out,rr,cr,acc[q*4]);''')
(p/'front_cached_debug.cu').write_text(s);(p/'front_cached_debug.launch.json').write_text('{"direct_weights":true}\n')
p=p/'cached_dx_plan.py';s=p.read_text().replace("source='front_cached_dx'):","source='front_cached_dx',debug=0):").replace('splits=1,source=source)','splits=1,source=source,debug=debug)').replace('self.partln=torch.empty((count,256),device=self.dx.device);self.bind','self.partln=torch.empty((count,256),device=self.dx.device)\n  if debug:self.partw=torch.empty_like(self.dx,dtype=torch.float32)\n  self.bind');p.write_text(s)
