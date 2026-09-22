"""Generate a source- and NCU-backed TriMul review, without changing kernels."""
import csv,json,re,html,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];E=R.parent/'trimul_sm90_parity_20260917/engine'
ps=json.loads((R/'ncu-summary.json').read_text());rs=json.loads((R/'all-results.json').read_text())
manifest=json.loads((R/'native_rebuilt/build/manifest.json').read_text());unit=manifest['units']['tmn90_z128_h128/sm_90a']
assert hashlib.sha256((R/'native_rebuilt/build'/unit['cubin']).read_bytes()).hexdigest()==unit['cubin_sha256']
proof={}
for part in ('k1','k3'):
 s=(R/f'trimul-{part}-selected.sass').read_text()
 proof[part]={op:len(re.findall(r'\b'+op+r'\b',s)) for op in ('UTMALDG','UTMASTG','HGMMA','STG','LDSM','STSM')}
assert proof['k1']['UTMALDG'] and not proof['k1']['UTMASTG'] and proof['k1']['HGMMA']
assert proof['k3']['UTMALDG'] and proof['k3']['UTMASTG'] and proof['k3']['HGMMA']
(R/'trimul-sass-summary.json').write_text(json.dumps(dict(cubin_sha256=unit['cubin_sha256'],static_instruction_counts=proof),indent=2))
sections=[]
def section(title,body):sections.append((title,body))
section('1. 정확히 무엇이 연결됐나', '''분석 대상은 `native_rebuilt`, BF16, C_z=C_h=128, B=1, residual 포함, dropout=0이다.
원본 커밋 `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`의 CUDA 소스를 수정하지 않은
CUDA12.9 재빌드다. 호출은 engine module → integration → native face → ops → K1/cuBLAS/K3다.

| 범위 | 현재 상태 |
|---|---|
| 단방향 outgoing / incoming inference module | 연결 및 수치·CUDA Graph 검증 |
| 배치 B=2, 서로 다른 mask, weight 변경 | 추가 검증 통과 |
| 기본값 | implementation=anthropic만 주면 v4. 이번 분석 경로는 anthropic_row=native_rebuilt 명시 필요 |
| 양방향 inference fusion | 새 native 연결 없음. 두 단방향 호출은 별도 |
| training / backward / dropout | 새 native 경로 미구현. engine는 grad-enabled 호출 거절 |
| MiniWorld 전체 모델 | 아직 통합 성능·정확도 검증 미완료 |

이 문서는 새 timing 실험이 아니다. 앞선 H100 결과를 소스, ptxas, 이번에 추출한 실제 cubin SASS와 대조했다.''')
section('2. 수식과 세 개의 핵심 단계', '''x = LN_in(z)
a = mask × sigmoid(x W_agᵀ) × (x W_apᵀ)
b = mask × sigmoid(x W_bgᵀ) × (x W_bpᵀ)

outgoing: X[c,i,j] = Σ_k a[c,i,k] b[c,j,k]
incoming: X[c,i,j] = Σ_k a[c,k,i] b[c,k,j]

y = z + sigmoid(LN_in(z) W_ogᵀ) × (LN_out(X) W_oᵀ)

| 단계 | 융합 내용 | HBM에 남기는 결과 |
|---|---|---|
| K1 | LN_in + a/b 양쪽 projection/gate + mask + channel-major 배치 | ab[2H,Np,Np], BF16 |
| cuBLAS | 채널별 triangle contraction, FP32 accumulate | X[H,Np,Np], BF16 |
| K3 | LN_in 재계산 + output gate, LN_out + output projection, gate×projection + residual | y[N,N,C] |

핵심 kernel은 3개지만 engine mask 생성·dtype 변환까지 포함한 전체 호출이 3 launch라는 뜻은 아니다.
mask는 contraction 입력 a/b에만 적용되고 output gate에는 적용하지 않는다.
여기서 a/b 두 projection은 **한 방향의 두 operand**이며 outgoing+incoming 양방향 실행이 아니다.

Incoming 기본 kt 경로는 K1의 TMA tensor-map stride를 바꿔 transposed plane을 바로 만든다.
따로 큰 transpose kernel을 실행하지 않고 outgoing과 같은 NT GEMM을 쓴다.
L384/768은 16의 배수이므로 Np=ceil16(N)에 따른 padding 낭비가 없다.''')
section('3. H100 구현: 실제 선택된 설정', '''| 항목 | K1 | K3 |
|---|---|---|
| 선택 kernel | t6x32_s8k2_m1_l2_v0 | t2x64_s8a2_l1 |
| 한 tile의 pair 위치 | 6×32 = 192개 | 2×64 = 128개 |
| producer / consumer warpgroups | 1 / 3 | 1 / 2 |
| CTA threads | 512 | 384 |
| persistent grid | 132 CTA | 132 CTA |
| CTA당 dynamic shared memory | 206,080 B (201.25 KiB) | 149,760 B (146.25 KiB) |
| compile register allocation | 128/thread | 168/thread |
| runtime producer / consumer budget | 40 / 152 regs | 24 / 240 regs |
| weight ring | 8 slots × 16 KiB | 8 slots × 8 KiB |
| 선택 C128에서 weight residency | 전체 W1을 CTA shared에 유지 | W_og + W_o를 CTA shared에 유지 |
| WGMMA | m64n64k16, register A / shared B | m64n32k16, register A / shared B |
| 입력 이동 | TMA | TMA |
| 출력 이동 | shared stmatrix → vector global store | shared stmatrix → TMA store |
| ptxas spill loads / stores | 0 / 0 | 0 / 0 |

K1 헤더의 ‘producer 1 + consumer 2’ 설명만 읽으면 틀린다. 현재 선택된 tile은 consumer가 3개다.
또 K1은 BJ=32이므로 `BJ % 64 == 0`인 TMA-store 조건을 만족하지 않는다.
실제 cubin에서 **K1: UTMALDG + HGMMA + STG, UTMASTG 없음**, **K3: UTMALDG + HGMMA + UTMASTG**를 확인했다.
SASS opcode의 정적 개수는 실행 횟수/성능 비중이 아니다.

둘 다 producer가 TMA와 mbarrier를 관리하고 consumer는 ldmatrix로 읽은 fragment를
register 안에서 정규화한 뒤 WGMMA의 A operand로 직접 사용한다.
따라서 normalized activation을 global memory에 써서 다시 읽을 필요가 없다.
128-byte shared swizzle, weight resident ring, producer/consumer register 재분배,
MMA와 gate/store의 겹침이 함께 설계돼 있다. 단순히 cp.async만 TMA로 바꾼 커널이 아니다.
K1 SCHED=0, K3 NACC=2로 두 accumulator 세트를 사용해 다음 MMA와 현재 epilogue를 겹친다.''')
section('4. 기존 Triton과 무엇이 다른가', '''| 항목 | 현재 비교한 engine Triton | Anthropic native |
|---|---|---|
| 입력 LN | 별도 kernel, x_n 저장 | K1 내부 register LN |
| gate 입력 | 저장된 x_n 재사용 | K3에서 원본 z를 읽고 LN 재계산 |
| front | a/b projection+gate+mask | 여기에 LN과 직접 plane 출력을 통합 |
| contraction | cuBLAS | cuBLAS, 같은 주요 GEMM 이름 확인 |
| back | LN_out + projection + gate + residual fusion | 같은 큰 수식에 LN_in 재계산과 H100 pipeline을 결합 |
| weight 준비 | 호출 경로에 작은 stack/cat/transpose 복사가 남음 | weight packing은 최초 1회, 이후 cache |

따라서 이번 비교는 **융합 알고리즘이 완전히 같은 Triton vs CUDA 비교가 아니다.**
입력 LN 저장/재계산 정책, packing 비용, 메모리 배치, Hopper 구현이 함께 다르다.
‘향상 전체가 TMA 덕분’이라고 분해할 수 없으며 이를 입증하려면 별도 ablation이 필요하다.

Triton NCU에 남은 CatArray kernel은 주로 weight packing이다. pair activation의 대형
concat과 혼동하면 안 된다. 이번 작업에서 이를 수정하지는 않았다.''')
lines=['NCU replay 기준 µs. Graph latency와 분리해서 읽는다.','', '| L | 구간 | 기존 Triton | Anthropic |','|---:|---|---:|---:|']
for L in (384,768):
 d={}
 for row in ('engine_triton','native_rebuilt'):
  p=next(p for p in ps if p['profile']==f'trimul-L{L}-C128-{row}')
  n=lambda cond:sum(k['us'] for k in p['kernels'] if cond(k['name']))
  vals=[n(lambda s:'layer_norm' in s or '_bidir_front' in s or 'tmn_k1' in s),n(lambda s:'nvjet' in s),n(lambda s:'_back_kernel' in s or 'tmn_k3' in s)]
  vals.append(p['total_us']-sum(vals));vals.append(p['total_us']);d[row]=vals
 for i,name in enumerate(['입력 LN + front / K1','cuBLAS contraction','출력 fusion / K3','mask·packing 등','합계']):lines.append(f'| {L} | {name} | {d["engine_triton"][i]:.2f} | {d["native_rebuilt"][i]:.2f} |')
lines += ['', 'L768에서는 K1 측 약 149µs, K3 약 166µs가 줄었다. cuBLAS는 거의 그대로다.',
          '따라서 이득의 중심은 앞뒤 fused kernel이다. K1/K3를 먼저 분석하고 cuBLAS를 유지하는 판단이 현재 데이터와 맞는다.']
section('5. 성능 이득은 어디서 나오나','\n'.join(lines))
section('6. Roofline과 메모리: 아직 여지가 있나', '''| L768 | K1 | K3 |
|---|---:|---:|
| 전체 native 호출 내 시간 비중 | 33.8% | 34.2% |
| HBM throughput / sustained peak | 64.1% | 64.0% |
| L2 throughput / sustained peak | 59.6% | 62.4% |
| BF16 Tensor throughput / sustained peak | 19.8% | 9.5% |
| issue-active / active cycles | 57.0% | 45.9% |
| eligible warps / scheduler cycle | 1.06 | 0.71 |

둘 다 register/shared-memory 제약상 1 CTA/SM이다. 하지만 ‘occupancy가 낮으니 실패’가 아니다.
weight 재사용과 fusion을 얻기 위해 자원을 많이 쓰는 설계다. spill은 없다.
K3는 낮은 eligible-warp 및 issue 비율이 보여 의존성·scheduling을 더 볼 가치가 있다.
이것만으로 특정 stall 원인이나 가능한 개선 배율을 확정할 수는 없다.

동일 작업량에서 HBM을 100% 쓰는 이상적 모델만 가정하면 64% 이용률은 약 1.56×의
kernel 속도 여유에 대응한다. 그러나 혼합 연산이므로 이것은 구현 가능 이득도,
알고리즘 전체의 엄밀한 상한도 아니다. 실제 15% 개선은 별도 실험이 필요하다.
K1/K3가 합쳐 약 68%이므로 둘 다 1.15×가 돼도 전체는 대략 1.10×다.
전체 1.15×를 원하면 나머지가 고정일 때 두 kernel이 약 1.24×여야 한다(Amdahl 계산).

| C=H=128 | L384 | L768 |
|---|---:|---:|
| ab plane | 72 MiB | 288 MiB |
| X contraction 결과 | 36 MiB | 144 MiB |
| ab + X workspace | 108 MiB | 432 MiB |
| 저장하지 않는 x_n 크기 | 36 MiB | 144 MiB |
| upstream workspace 정책 | cache에 유지 | 256 MiB 기준 초과, transient allocation |

x_n의 global write와 front/back read를 피하지만 K3에서 z를 다시 읽는다.
실제 DRAM 절감량은 L2 재사용과 residual의 shared 재사용도 포함해 달라지므로
위 tensor 크기를 그대로 HBM byte 감소량이라고 주장하지 않는다.
L768의 transient는 framework caching allocator 정책이며, kernel 컴파일 cache miss를 뜻하지 않는다.''')
section('7. 살릴 설계와 주의할 계약', '''**계승할 핵심:** register LN → WGMMA RS 연결, channel-major plane 직접 출력,
incoming의 descriptor transpose, CTA 내 weight residency, K3의 gate/projection/residual
융합, shape별 명시적 tile table 및 binary/source provenance 관리.

**후속 실험 우선순위:** K3 scheduling/epilogue, K1 6×32와 TMA-store 가능한 tile의
전체 비용 비교, 작은 wrapper packing 비용 정리. K1의 TMA store 미사용은 결함으로
단정하지 않는다. tile 모양을 바꾸면 locality·레지스터·파이프라인도 함께 변한다.

**학습으로 가져갈 때:** API의 save_intermediates는 현재 명시적으로 거절된다.
K1 stats 저장용 낮은 수준 hook이 있다는 것과 학습 지원은 다르다.
Backward가 쓸 a/b, X, LN stats, gate/projection 값을 저장할지 재계산할지 설계하고
mask/dropout/residual gradient 및 모든 파라미터 gradient를 새로 검증해야 한다.

**Cache 계약:** 원본 pack은 기본적으로 주소 기반이고 weight in-place 변경 시
caller가 cache를 무효화해야 한다. engine adapter는 parameter version/교체/epsilon을
키에 포함시켜 재생성한다. workspace key에는 CUDA stream이 없으므로 같은 module/cache의
동시 multi-stream 재진입은 이번에 검증하지 않았고 별도 cache/event 설계가 필요하다.
B>1은 각 plane 순차 호출 후 torch.stack이므로 완전히 fused batched kernel이 아니다.''')
section('8. 원본·증거 위치', '''원본 pinned revision: https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e

- `native_rebuilt/python/trimul_native/ops.py`: pack_weights(59), trimul_plane(191), trimul(258)
- `native_rebuilt/python/trimul_native/kernel.py`: tile/SMEM(66), K1 launch(307), K3 launch(362)
- `native_rebuilt/csrc/tmn_kernels.cuh`: K1Cfg(82), K1 body(448), TMAST 조건(535), K3 body(708)
- `native_rebuilt/csrc/tmn_ptx.cuh`: TMA load/store(93/111), WGMMA(158/213)
- `native_rebuilt/csrc/common/tmn_math.cuh`: LN/gate/residual 반올림 수식
- `trimul-k1-selected.sass`, `trimul-k3-selected.sass`, `trimul-sass-summary.json`: 이번 cubin 대조 결과
- `ncu/trimul-L384-C128-native_rebuilt.csv`, `ncu/trimul-L768-C128-native_rebuilt.csv`: 실행 config/지표
- `all-results.json`: 후보별 수치 검사 및 CUDA Graph 시간

상대 경로의 기준은 `runs/anthropic_adoption_20260919/`이다.
원본 소스·커널은 수정하지 않았다. 이 문서는 후속 최적화나 학습 개발 완료 주장이 아니다.''')
md='# Anthropic TriMul 심층 분석\n\n2026-09-19 · source + actual cubin + NCU 대조\n\n'+'\n\n'.join('## '+h+'\n\n'+b for h,b in sections)+'\n'
(E/'docs/anthropic-trimul-analysis.md').write_text(md)
# Minimal deterministic renderer for this report: paragraphs and GFM-style tables.
def inline(s):
 s=html.escape(s);s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s);s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s);return s

def render(text):
 out=[]
 for block in text.split('\n\n'):
  ls=block.splitlines()
  if ls[0].startswith('|'):
   table=[]
   for i,line in enumerate(ls):
    cells=[c.strip() for c in line.strip().strip('|').split('|')]
    if i==1:continue
    tag='th' if i==0 else 'td';table.append('<tr>'+''.join(f'<{tag}>{inline(c)}</{tag}>' for c in cells)+'</tr>')
   out.append('<div class="table-wrap"><table>'+''.join(table)+'</table></div>')
  else:out.append('<p>'+ '<br>'.join(inline(line) for line in ls)+'</p>')
 return '\n'.join(out)
styles=re.search(r'<style>(.*?)</style>',(ROOT/'ANTHROPIC_STATUS.html').read_text(),re.S).group(1)
page='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Anthropic TriMul · 심층 분석</title><style>'+styles+'p{overflow-wrap:anywhere}.table-wrap{margin:18px 0}td{vertical-align:top}</style></head><body><header><div class="eyebrow">SOURCE · CUBIN · NCU</div><h1>Anthropic TriMul을 뜯어보면</h1><p>K1 → cuBLAS → K3. 융합 수식, H100 구현, 실측 병목과 학습으로 계승할 설계.</p></header><nav><a href="index.html">← 전체 현황판</a><a href="#s3">H100 설정</a><a href="#s5">성능 분해</a><a href="#s7">후속 설계</a></nav><main>'
page+=''.join('<section id="s'+str(i)+'" class="panel"><h2>'+html.escape(h)+'</h2>'+render(b)+'</section>' for i,(h,b) in enumerate(sections,1))
page+='</main></body></html>'
for p in (ROOT/'ANTHROPIC_TRIMUL.html',R/'web/trimul.html',R/'site/dist/trimul.html'):p.write_text(page)
# Add a stable local/hosted entry point; idempotent on repeated renders.
for p in (ROOT/'ANTHROPIC_STATUS.html',R/'web/index.html',R/'site/dist/index.html'):
 s=p.read_text()
 if 'id="trimul-deep-link"' not in s:
  target='ANTHROPIC_TRIMUL.html' if p.parent==ROOT else 'trimul.html'
  s=s.replace('<section id="wiring">','<div id="trimul-deep-link" class="notice"><b>추가 분석</b> · <a href="'+target+'">Anthropic TriMul 심층 분석: 소스·실제 cubin·NCU 대조 →</a></div><section id="wiring">')
  p.write_text(s)
print('TriMul report + local and hosted HTML prepared; cubin SHA and instruction claims checked.')
