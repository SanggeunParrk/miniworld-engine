"""Rebuild the local report using checked measurements, not cached claims."""
from pathlib import Path
import csv
import hashlib
import json
import re

R = Path(__file__).resolve().parent
c = json.loads((R / 'confirmation.json').read_text())
s = json.loads((R / 'sanitizer-L384.json').read_text())
assert all(x['valid'] for case in c['checks'].values() for x in case.values())
assert all(case['depth12']['bit_exact_to_prior_ring'] for case in c['checks'].values())
assert len(s) == 2 and all(x['returncode'] == 0 for x in s)
manifest = c['cubin_manifest']['depth12']
assert hashlib.sha256(Path(manifest['path']).read_bytes()).hexdigest() == manifest['sha256']
for name in ('memcheck-L384.log', 'racecheck-L384.log'):
    assert manifest['path'] in (R / name).read_text()
for f, h in c['source_sha256']['depth12'].items():
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest() == h

t = c['times']
prior = (1 - t['depth12'] / t['ring_baseline']) * 100
split = (1 - t['depth12'] / t['split']) * 100
original = (1 - t['depth12'] / t['old_400']) * 100
selection = dict(status='experimental_validated', scope='Bidirectional TriMul B7-B12, L384 C128 projection width256',
    plan='plan.py', kwargs=dict(clusters=10, mode=52),
    environment=dict(B7_CONSUMERS='10', B7_PRODUCER_REGS='64', B7_RING_DEPTH='12'),
    ctas=260, source_ctas=160, dx_ctas=100, ring_bytes=10*12*131072,
    ring_depth=12, shared_ring_chunk_bytes=16384, shared_ring_slots=4,
    dynamic_shared_bytes=114688, producer_registers=64, compute_registers=192,
    transpose_dw_partial=True, native_layout_reducer=True,
    source_sha256=c['source_sha256']['depth12'], cubin=manifest,
    times_us=t, benchmark='confirmation.json', sanitizer='sanitizer-L384.json',
    target_us=252, target_met=False, sol90_verified=False, production_dispatch_changed=False)
(R / 'selected.json').write_text(json.dumps(selection, indent=2) + '\n')

keys = ['gpu__time_duration.sum', 'dram__bytes_read.sum', 'dram__bytes_write.sum',
        'lts__t_sectors.sum', 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active',
        'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum', 'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']
profiles = {}
for label, path in [('prior', R.parent/'trimul_b7_ring_balance_20260922/ncu-L384-mode4-p64.csv'),
                    ('new', R/'ncu-L384-mode52-p64-r12.csv')]:
    rows = list(csv.DictReader(path.open()))
    profiles[label] = {k: dict(value=rows[-1][k], unit=rows[0][k]) for k in keys}
(R / 'profile-summary.json').write_text(json.dumps(profiles, indent=2) + '\n')

names = [('old_400','재개 출발점 통합형'), ('split','현재 분리형'),
         ('ring_baseline','직전 선택 통합형'), ('reduced','dW 합산 배치 개선'),
         ('depth12','새 선택: 합산 개선 + ring 12타일')]
table = ''.join(f'| {label} | {t[n]:.3f} |\n' for n,label in names)
readme = f'''# B7–B12: dW 합산 배치 + ring depth 12

H100 / L384 / C128 / projection width256 / 단일 cooperative CUDA launch.
Anthropic 유래 Apache-2.0 TMA/WGMMA primitives를 사용한 학습 확장이다.

## 동일 실행 비교

| 구현 | 시간 (µs) |
| --- | ---: |
{table}
직전 통합형보다 {prior:.2f}%, 현재 분리형보다 {split:.2f}%, 재개 출발점보다 {original:.2f}% 시간 단축.
5개 변형 입력을 검증한 뒤 같은 입력·프로세스에서 순서를 교대로 바꾸며 5×160회 측정했다.
B7–B12의 초기화와 최종 reduction을 포함한다. 전체 학습 / L768 결과가 아니다.
252 µs 목표는 미달이며 SoL90은 입증하지 않았다.

## 변경

1. source의 dW partial을 [2,128,32] 순서로 shared에서 재배치한 뒤 연속 global store한다.
2. 최종 reducer도 partial의 연속 주소를 담당하도록 바꿨다. 기존 값과 합산 순서는 유지한다.
3. 그룹당 global ring을 8 → 12타일로 늘렸다. dX CTA 10개에 전달할 데이터를 source가 더 앞서 생성할 수 있다.
4. dX의 shared ring은 기존과 동일하게 16 KiB × 4이다. global ring 크기 변경과 구분해야 한다.

source 160 CTA / dX 100 CTA / producer 64 registers / compute 192 registers.
global ring은 10 → 15 MiB. 총 논리적 dp/dg 크기는 그대로이며 실제 HBM 트래픽은 별도다.

## 시도와 선택

- source dW/다음 projection rolling: 421.62 µs vs 같은 실행 기준 366.78 µs, 제외.
- dX WGMMA rolling: 404.08 µs vs 369.86 µs, 제외. 결합 후보도 더 느렸다.
- 첫 dp/dg 타일을 먼저 전달: 381.79–387.87 µs vs 368.05 µs, 제외.
- dW partial/reducer 배치 세 방식: 모두 정확도 통과, 결합 방식을 선택.
- ring depth 4/8/10/12/14/16/20 비교: 12 선택. 4는 심한 대기, 과도한 크기는 이득 감소.
- 새 depth에서 CTA/레지스터 배분 재조정: G10/U10/P64 유지.

서로 다른 실행의 절대 시간을 합쳐 속도 향상을 계산하지 않는다. 최종 비교는 confirmation.json이다.

## 검증

5개 입력/가중치/마스크 변형에서 7개 출력 모두 기존 허용오차 통과.
선택 후보는 직전 통합형과 5건 모두 bit-exact. eager/graph, scratch poison, counter/flag 재사용 통과.
선택한 동일 cubin의 memcheck/racecheck 오류 0건. NCU local spill traffic 0.

NCU 별도 실행에서 새 후보 HBM read 303.24 MB / write 144.61 MB,
이전 후보 read 303.76 MB / write 56.80 MB다. ring 확대는 실제 쓰기를 늘리는 비용이 있다.
pipeline 대기 감소의 이득이 이 비용을 넘었다. HBM 왕복을 줄여 얻은 성능이라고 해석하면 안 된다.
Tensor active 37.42%는 SoL 수치가 아니다. 프로파일 latency 340.19 µs는 graph 교차 측정과 별도다.

## 재현

선택 설정: selected.json. 소스/cubin SHA-256: confirmation.json 및 selected.json.
최종 측정: `sbatch confirm.slurm --depths 12 --cases 5` (저장소 루트의 전체 경로로 실행).
검증 환경: B7_CONSUMERS=10, B7_PRODUCER_REGS=64, B7_RING_DEPTH=12, MODE=52.
production dispatch 및 전체 모델 배선은 변경하지 않았다.
'''
(R/'README.md').write_text(readme)

rows = ''.join(f'<tr><td>{label}</td><td>{t[n]:.2f} µs</td></tr>' for n,label in names)
html = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>B7–B12 · ring 12타일</title><style>
body{{max-width:1100px;margin:30px auto;padding:0 18px;font:17px/1.7 system-ui;color:#20374b;background:#f4f7fb}}section{{background:white;padding:20px;margin:20px 0;border-radius:12px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #ccd6df;text-align:left}}.flow{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px}}.box{{padding:15px;border-radius:10px;background:#e7f4ed}}.ring{{background:#f4e9fa}}.io{{background:#e7effa}}a{{color:#1564aa}}small{{color:#526779}}@media(max-width:720px){{.flow{{grid-template-columns:1fr}}}}</style>
<h1>B7–B12 · {t['depth12']:.2f} µs 단일 커널</h1><p>L384 · H100 · 2026-09-22 · 개발 후보</p>
<section><h2>같은 실행에서 비교</h2><table><tr><th>구현</th><th>시간</th></tr>{rows}</table><p>직전 통합형 대비 {prior:.2f}%, 분리형 대비 {split:.2f}% 시간 단축.</p><small>5×160회 교차 측정. B7–B12 초기화·reduction 포함. 전체 학습 수치가 아닙니다. 252 µs·SoL90 목표 미달.</small></section>
<section><h2>한 CUDA 커널 안의 데이터 이동</h2><div class="flow"><div class="box"><b>Source · 160 CTA</b><p class="io">x_n, dleft/dright, mask, projection/gate weights 읽기</p><p>p/g 재계산 → shared dp/dg 생성</p><p>① dW WGMMA가 소비<br>② 동시에 global ring으로 전달</p><p><b>변경:</b> dW partial을 shared에서 [2,128,32]로 재배치 → 연속 global store</p></div><div class="box ring"><b>Global ring · L2 재사용 유도</b><p>그룹당 <b>8 → 12타일</b><br>전체 <b>10 → 15 MiB</b></p><p>left dg / left dp / right dg / right dp</p><p>먼저 생성한 타일을 둘 공간을 늘려 source 대기를 줄입니다.</p><small>shared buffer는 별개이며 16 KiB × 4를 유지합니다.</small></div><div class="box"><b>dX · 100 CTA</b><p class="io">ring dp/dg, weights, dGate, raw X, residual gradient, LN parameters 읽기</p><p>16 KiB 전송과 WGMMA 겹치기 → dX_n → LN 미분 + residual</p><p>출력: dX, LN parameter partial</p></div></div><p><b>같은 커널 마지막:</b> grid sync → 연속 주소에서 dW partial 읽기·합산 → dW 출력. LN parameter partial도 합산.</p></section>
<section><h2>NCU와 검증</h2><p>5개 변형 입력의 7개 출력 모두 허용오차 통과. 직전 통합형과 bit-exact. memcheck/racecheck 오류 0건. local spill traffic 0.</p><p>NCU: HBM 읽기 303.24 MB / 쓰기 144.61 MB. 이전 쓰기 56.80 MB보다 늘었습니다. ring 확대는 대기를 줄이는 대신 실제 HBM 쓰기를 늘리는 비용이 있습니다.</p><p>Tensor active 37.42%는 SoL이 아닙니다. 340.19 µs 프로파일 시간은 위 graph 교차 측정과 별도입니다.</p><p><a href="confirmation.json">최종 측정</a> · <a href="selected.json">설정·SHA-256</a> · <a href="profile-summary.json">NCU 지표</a> · <a href="README.md">시도·상세 기록</a> · <a href="../../TRIMUL_STATUS.html">전체 현황</a></p><small>Anthropic 유래 Apache-2.0 TMA/WGMMA primitives를 사용한 학습 확장. production dispatch는 변경하지 않았습니다.</small></section></html>'''
(R/'index.html').write_text(html)
root = R.parent.parent/'TRIMUL_STATUS.html'
text = root.read_text()
text = re.sub(r'<aside id="b7-ring-depth-20260922">.*?</aside>', '', text, flags=re.S)
note = f'<aside id="b7-ring-depth-20260922"><b>최신 B7–B12:</b> dW 합산 배치 개선 + ring 12타일. 같은 실행에서 직전 {t["ring_baseline"]:.2f} → 새 통합형 {t["depth12"]:.2f} µs ({prior:.2f}% 단축), 분리형 {t["split"]:.2f} µs. 정확도·memcheck·racecheck 통과. L384 한정, 252 µs 미달. <a href="runs/trimul_b7_ring_depth_20260922/index.html">최신 비교·데이터 이동·NCU</a></aside>'
root.write_text(text.replace('</header>', '</header>'+note, 1))
print(t, dict(prior_reduction_pct=prior, split_reduction_pct=split, original_reduction_pct=original))
