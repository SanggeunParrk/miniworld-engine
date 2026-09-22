"""Render the local report from completed measurements; never infer a pass."""
from pathlib import Path
import json
import re

R = Path(__file__).resolve().parent
c = json.loads((R / 'confirmation.json').read_text())
assert 'times' in c and all(v['valid'] for case in c['checks'].values() for v in case.values())
t = c['times']
chosen = 'g10u10p64'
san_path = R / 'sanitizer-L384.json'
san = json.loads(san_path.read_text()) if san_path.exists() else []
san_ok = len(san) == 2 and all(x['returncode'] == 0 for x in san)
selected = dict(status='experimental_validated' if san_ok else 'numerically_checked_sanitizer_pending',
    scope='Bidirectional TriMul B7-B12 only, L384 C128 projection width256',
    plan='plan.py', kwargs=dict(clusters=10, mode=4),
    environment=dict(B7_CONSUMERS='10', B7_PRODUCER_REGS='64'),
    ring_chunk_bytes=16384, ring_shared_slots=4, ctas=260,
    dynamic_shared_bytes=114688, producer_registers=64, compute_registers=192,
    benchmark='confirmation.json', sanitizer='sanitizer-L384.json',
    times_us=t, source_sha256=c['source_sha256'][chosen],
    target_us=252, target_met=t[chosen] <= 252, sol90_verified=False,
    production_dispatch_changed=False)
(R / 'selected.json').write_text(json.dumps(selected, indent=2) + '\n')
names = [('split', '현재 분리형'), ('old_400', '재개 출발점: 약 400 µs 통합형'),
         ('ring_baseline', '직전 개선 통합형: 약 388 µs'),
         (chosen, '이번 선택: 16 KiB × 4, producer 64 regs'),
         ('g10u10p48', '비교 후보: producer 48 regs')]
rows = ''.join(f'<tr><td>{label}</td><td>{t[n]:.2f} µs</td><td>{(1-t[n]/t["ring_baseline"])*100:.2f}%</td></tr>' for n,label in names)
validation = 'memcheck·racecheck 오류 0건' if san_ok else 'memcheck·racecheck 최종 검증 대기'
speed = (1-t[chosen]/t['ring_baseline'])*100
split_speed = (1-t[chosen]/t['split'])*100
readme = f'''# B7–B12 단일 CUDA 커널: 16 KiB ring pipeline

L384 / C128 / projection width256 / H100 / 단일 cooperative launch.
동일 프로세스에서 입력을 공유하고 순서를 교대로 바꾸며 5×160회 측정.
각 커널의 초기화·최종 reduction을 포함한 B7–B12 범위다.

## 결과

| 구현 | µs |
| --- | ---: |
''' + ''.join(f'| {label} | {t[n]:.3f} |\n' for n,label in names) + f'''
직전 통합형보다 {speed:.2f}%, 현재 분리형보다 {split_speed:.2f}% 시간 단축.
252 µs 및 SoL90 미달/미입증. 전체 학습이나 L768 결과로 외삽하지 않는다.
최종 반복 측정에서 producer 64 regs가 48 regs보다 소폭 빨랐다. 선택값은 64 regs다.

## 구현

- source CTA가 projection/gate를 재계산하고 dp/dg를 한 번 생성한다.
- 같은 shared dp/dg를 dW WGMMA가 소비하는 동안 global ring으로 bulk store한다.
- ring을 left dg / left dp / right dg / right dp 네 개 32 KiB plane으로 배치했다.
- dX CTA는 16 KiB씩 네 shared slot으로 읽으며 다음 전송과 현재 WGMMA를 겹친다.
- ring-load warp와 weight-load warp를 분리한다. 가중치는 16 KiB 두 버퍼다.
- 기존 GEMM 합산 및 BF16 반올림 순서를 유지한다. LN 미분과 residual도 같은 커널이다.
- 끝에서 cooperative grid sync 후 dW와 LN 파라미터 partial을 합산한다.
- global ring 크기와 논리적 전송량은 그대로다. global memory ring은 L2 재사용을 유도하며 shared-only 전송이 아니다.
- 8 KiB는 전송/동기화 증가를 상쇄하지 못했다. 3D TMA store·early gate·CTA/레지스터 배분도 비교했다.

## 검증과 한계

5개 입력/가중치/마스크 변형에서 7개 출력의 기존 허용오차 통과.
scratch poison, eager/graph 일치, counter/flag 재사용 검사 통과. {validation}.
원본: [confirmation.json](confirmation.json), [selected.json](selected.json).
컴파일러 로그 및 cubin은 build/에 있다. 현재 후보는 spill 0 bytes다.
NCU Tensor activity는 SoL 수치가 아니며 단독 프로파일 latency를 graph latency와 섞지 않는다.
Anthropic 유래 Apache-2.0 TMA/WGMMA primitives를 바탕으로 한 학습 확장이다.
실험 후보이며 production dispatch는 변경하지 않았다.
'''
(R / 'README.md').write_text(readme)
html = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B7–B12 · 16 KiB pipeline</title><style>
body{{max-width:1100px;margin:30px auto;padding:0 18px;font:17px/1.7 system-ui;color:#20374b;background:#f4f7fb}}section{{background:white;padding:20px;margin:20px 0;border-radius:12px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ccd6df;text-align:left}}.flow{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}}.box{{padding:15px;border-radius:10px;background:#e7f4ed}}.ring{{background:#f4e9fa}}.io{{background:#e7effa}}a{{color:#1564aa}}small{{color:#526779}}@media(max-width:720px){{.flow{{grid-template-columns:1fr}}}}
</style><h1>B7–B12 · 16 KiB 전송으로 계산과 로드 겹치기</h1><p>L384 · H100 · 단일 CUDA kernel · 2026-09-22</p>
<section><h2>같은 실행에서 비교</h2><table><thead><tr><th>구현</th><th>시간</th><th>직전 통합형 대비 시간 단축</th></tr></thead><tbody>{rows}</tbody></table><p>선택 후보: 직전 통합형 대비 {speed:.2f}%, 분리형 대비 {split_speed:.2f}% 시간 단축.</p><small>5×160회 교차 측정. 약 400/388 µs는 과거 식별용 이름이며 위 시간은 이번 실행 실측. B7–B12만 비교. 252 µs·SoL90 목표는 달성하지 못했습니다.</small></section>
<section><h2>단일 커널 안의 데이터 이동</h2><div class="flow"><div class="box"><b>160 source CTAs</b><p class="io">입력: x_n, dleft/dright, mask, projection/gate 가중치</p><p>p/g 재계산 → dp/dg 생성</p><p>① dW WGMMA가 shared dp/dg 소비<br>② 동시에 dp/dg를 ring으로 bulk store</p><p>출력: dW partial</p></div><div class="box ring"><b>Global ring · L2 재사용 유도</b><p>left dg · left dp<br>right dg · right dp</p><p>각 32 KiB plane<br>각 타일 총 128 KiB</p><p>16 KiB × 4 shared slot으로 읽기<br>전송 중 다음 WGMMA 실행</p><small>ring 총 바이트 수는 동일합니다. HBM 왕복을 없앤 변경이 아닙니다.</small></div><div class="box"><b>100 dX CTAs</b><p class="io">입력: ring dp/dg, 가중치, dGate, raw X, residual gradient, LN gamma/beta</p><p>WGMMA → dX_n<br>gate gradient 합산<br>LN 미분 + residual</p><p>출력: dX, LN parameter partial</p></div></div><p>마지막에 같은 커널 안에서 grid sync → dW / dgamma / dbeta partial 합산.</p></section>
<section><h2>검증</h2><p>5개 변형 입력 · 7개 출력 정확도 · buffer poison · eager/graph 반복 · flag 재사용 검사 통과. {validation}.</p><p>8 KiB, 3D TMA store, early gate, CTA 및 레지스터 배분도 비교했습니다. 전체 학습 배선은 아직 실험 후보로 교체하지 않았습니다.</p><p><a href="confirmation.json">측정 원본</a> · <a href="selected.json">선택 설정</a> · <a href="README.md">상세 기록</a> · <a href="../../TRIMUL_STATUS.html">전체 현황</a></p><small>Anthropic 유래 Apache-2.0 TMA/WGMMA primitives를 사용한 학습 확장.</small></section></html>'''
(R / 'index.html').write_text(html)
root = R.parent.parent / 'TRIMUL_STATUS.html'
s = root.read_text()
s = re.sub(r'<aside id="b7-ring-16k-20260922">.*?</aside>', '', s, flags=re.S)
note = f'<aside id="b7-ring-16k-20260922"><b>최신 B7–B12 단일 커널:</b> 같은 실행에서 직전 통합형 {t["ring_baseline"]:.2f} → 16 KiB 후보 {t[chosen]:.2f} µs, {speed:.2f}% 단축. 분리형 {t["split"]:.2f} µs. {validation}. L384 한정, 252 µs·SoL90 미달. <a href="runs/trimul_b7_ring_balance_20260922/index.html">최신 표·데이터 이동·검증</a></aside>'
root.write_text(s.replace('</header>', '</header>' + note, 1))
print(t, validation)
