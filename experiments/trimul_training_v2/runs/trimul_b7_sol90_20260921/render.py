from pathlib import Path
import html,json
R=Path(__file__).resolve().parent
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1660" height="1150" viewBox="0 0 1660 1150"><rect width="1660" height="1150" fill="#f4f8fb"/><style>text{font-family:Arial, sans-serif;fill:#183546}.code{font-family:monospace;font-size:16px}.sub{fill:#506b7b}</style>']
def text(x,y,s,size=19,cls=''):
 parts.append('<text x="%s" y="%s" font-size="%s" class="%s">%s</text>'%(x,y,size,cls,html.escape(s)))
def box(x,y,w,h,fill):parts.append('<rect x="%s" y="%s" width="%s" height="%s" rx="10" fill="%s" stroke="#bed0db"/>'%(x,y,w,h,fill))
def lines(x,y,ss,cls='',step=29):
 for i,s in enumerate(ss):text(x,y+i*step,s,17,cls)
text(32,43,'B7–B12 · H100 학습 · 실제 HBM 입출력과 커널 내부 계산',29)
text(32,77,'L384 / L768 · C128 · 양방향 H256 · BF16 · dropout 25% · B1 v51 및 forward 저장 정책 유지',19,'sub')
text(32,106,'개선 후보: dW 두 행 타일 중첩 + dX 전용 TMA producer + 두 커널 모두 2 CTA/SM',18)
text(32,139,'HBM payload는 논리적 버퍼 크기다. 재읽기는 L2에서 처리될 수 있으며 실측 DRAM 전송량과 다르다.',16,'sub')
for y,title,reads,left,code,writes in [
 (176,'CUDA ① dW · 256 CTA · 128 producer + 128 consumer · 2 CTA/SM · shared 112 KiB',
  ['HBM → TMA / register','x_n: [M,128] BF16','W1: [1024,128] BF16','dLeft / dRight: [256,M] ×2','pair mask: [M] BF16'],
  ['네 단계 TMA 입력 버퍼','x_n은 이미 정규화됨','projection/gate 재계산','GP(row1) ↔ GLU(row0) 중첩','두 행 K128 dW → FP32 누적','CTA partial → 전체 합산','중복 x_n shared 재저장 제거'],
  ['# Each side and hidden-channel tile','p = xn @ Wp','s = sigmoid(xn @ Wg)','d = bf16(dSide * mask)','dp = bf16(d * s)','dg = bf16(d * p * s * (1-s))','partial_dWp += xn.T @ dp','partial_dWg += xn.T @ dg','dW = bf16(sum(partials))'],
  ['HBM outputs','dWL / dWLg / dWR / dWRg','4 × [128,256] BF16','합계 262.14 kB','scratch: FP32 partial dW','16.78 MB write + read','dConcat HBM 저장 없음']),
 (572,'CUDA ② dX + LN 미분 + residual · 264 CTA · 128 producer + 128 consumer · 2 CTA/SM · shared 112 KiB',
  ['HBM → TMA / register','x_n, raw x: [M,128] ×2','dLeft / dRight: [256,M] ×2','dGate, dy: [M,128] ×2','WT 4개, Wgate, gamma_in','pair mask: [M] BF16'],
  ['전용 producer: TMA 데이터 공급','mask는 한 번 읽고 재사용','projection/gate 재계산','L768: 다음 행 x_n·첫 G 선행 읽기','raw x → mean / rstd 계산','raw fragment 레지스터 유지','LN 미분 + dy residual'],
  ['dp, dg = same_GLU_bwd(xn, dSide)','dxn = sum(dp @ Wp.T + dg @ Wg.T)','dxn = bf16(dxn + bf16(dGate @ Wgate.T))','mu, rs = stats(raw_x)','xhat = (raw_x - mu) * rs','dx, dgamma, dbeta = ln_bwd(dxn, xhat, rs)','dx = bf16(bf16(dx) + dy)','param_grad = sum(CTA_partials)'],
  ['HBM outputs','dx: [M,128] BF16','L384: 37.75 MB','L768: 150.99 MB','dgamma_in, dbeta_in: 1.02 kB','LN scratch: 270.34 kB','dxn HBM 저장 없음'])]:
 text(32,y,title,21)
 box(32,y+20,300,330,'#e6f2fc');box(355,y+20,910,330,'#e3f4ed');box(1288,y+20,340,330,'#fff0df')
 parts.append('<path d="M332 %d H355 M1265 %d H1288" stroke="#527485" stroke-width="3"/>'%(y+160,y+160))
 parts.append('<path d="M670 %d V%d" stroke="#83b29a"/>'%(y+34,y+335))
 lines(48,y+51,reads);lines(370,y+51,left);lines(686,y+51,code,'code',28);lines(1305,y+51,writes)
text(32,975,'동일 GPU 교대 CUDA-event 측정 · 두 CUDA launch를 합친 B7–B12 구간',21)
text(32,1011,'L384: 544.928 → 395.584 μs  (1.378×)        L768: 2154.080 → 1486.816 μs  (1.449×)',22)
text(32,1049,'전체 FWD+BWD: 1200.352 → 1048.768 μs / 4980.608 → 4265.904 μs. 구간별 측정값을 합산하지 않는다.',18)
text(32,1087,'SoL90 미달. 수식은 의미 설명용 PyTorch 의사코드이며 BF16 반올림·실행 순서의 완전한 명세가 아니다.',17,'sub')
text(32,1121,'실제 검증·기준 오차·NCU 해석은 함께 제공하는 보고서 참조. B1, cuBLAS triangle matmul, 저장 정책은 이번 변경 대상 밖.',17,'sub')
parts.append('</svg>');(R/'b7-current.svg').write_text(''.join(parts))
