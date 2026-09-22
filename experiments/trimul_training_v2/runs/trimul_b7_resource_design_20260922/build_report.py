from pathlib import Path
import hashlib
import json

R = Path(__file__).resolve().parent
root = R.parents[1]
selected = json.loads((R.parent / 'trimul_b7_l2_reuse_20260922/selected.json').read_text())
for path, expected in selected['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected

M = 384**2
tiles = M // 64
groups = 10
paired_tiles = sum(((tiles - 1 - group) // groups + 2) // 2 for group in range(groups))
rows = [
    ('source x_n 읽기', M * 128 * 2 * 16, M * 128 * 2 * 16),
    ('dp/dg ring 쓰기+읽기', tiles * 131072 * 2, tiles * 131072 * 2),
    ('dX 주 가중치 읽기', tiles * 4 * 128 * 256 * 2, paired_tiles * 4 * 128 * 256 * 2),
    ('output-gate 가중치 읽기', tiles * 128 * 128 * 2, paired_tiles * 128 * 128 * 2),
    ('dW partial 쓰기+읽기', groups * 4 * 128 * 256 * 4 * 2, groups * 4 * 128 * 256 * 4 * 2),
]
assert tiles == 2304 and paired_tiles == 1154
assert 128 * (32 + 104 + 104) == 384 * 80 == 30720
assert 128 * (24 + 192 + 24) == 30720
data = {
    'status': 'design_only_not_benchmarked',
    'scope': 'L384 bidirectional B7-B12 single launch',
    'selected_source_sha256': selected['source_sha256'],
    'row_tiles': tiles,
    'paired_row_tiles_in_10_groups': paired_tiles,
    'traffic_kind': 'logical payload, not measured HBM or L2 bytes',
    'traffic': [dict(name=n, current_bytes=a, proposed_bytes=b) for n, a, b in rows],
    'register_budget_per_cta': 30720,
    'resource_budget_verified_by_compiler': False,
    'new_gpu_measurements': False,
}
(R / 'cost_model.json').write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
table = ''.join('<tr><td>{}</td><td>{:.1f}</td><td>{:.1f}</td></tr>'.format(n, a/2**20, b/2**20) for n, a, b in rows)
html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>B7–B12 자원 수명과 가중치 공유 설계</title><style>
body{max-width:1080px;margin:24px auto;padding:0 20px;font:16px/1.65 system-ui;color:#203448;background:#f6f8fc}h1{line-height:1.3}section{background:white;border:1px solid #d4dfec;border-radius:12px;padding:20px;margin:18px 0}a{color:#175da0}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:9px;border-bottom:1px solid #dde5ef}.banner{background:#fff1d8;padding:15px}.flow{display:grid;grid-template-columns:1fr 1fr;gap:14px}.wide{grid-column:1/-1}.box{background:#e9f4f0;border:1px solid #a3cabe;border-radius:8px;padding:16px}.load{background:#e8f0fa;border-color:#a6bdd8}.ring{background:#f4eafa;border-color:#c5a8d6}.arrow{text-align:center;font-weight:bold}.muted{color:#52677e}code{font-family:monospace} @media(max-width:640px){.flow{grid-template-columns:1fr}table{font-size:13px}}
</style><header><a href="../../TRIMUL_STATUS.html">← TriMul 현황</a><h1>재사용이 실제 이득이 되도록<br>레지스터·동기화·버퍼를 함께 설계</h1>
<p class="banner"><b>설계 검토 · 새 벤치마크 없음.</b> 현재 선택 및 361.62 µs 기록은 그대로입니다. 아래 후보의 성능·SoL·occupancy는 아직 측정하지 않았습니다.</p></header>
<section><h2>핵심: 값은 짧게 보관하고, 가중치는 두 계산 그룹이 공유</h2><p>먼저 기존 BF16 반올림 뒤의 값을 두 개씩 포장합니다. 그다음 한 CTA 안에서 각 계산 그룹이 한 행 타일의 누산기를 따로 소유하게 합니다.</p>
<div class="flow"><div class="box ring wide"><b>기존 source CTA 유지</b><br>x_n + dleft/dright → projection/gate 재계산 → dp/dg → 장기 dW 누적<br>dp/dg는 기존 global ring으로 전달. 이 왕복을 없앤 설계는 아닙니다.</div>
<div class="arrow wide">↓ 행 A/B의 독립 derivative 입력</div><div class="box load wide"><b>dX CTA의 producer warpgroup</b><br>가중치 타일 한 번 TMA load → 공용 shared 버퍼<br>각 행의 derivative 버퍼는 독립 관리</div>
<div class="box"><b>compute WG A · 128 threads</b><br>행 A: dX 누산기 64 FP32<br>→ gate 합산 후 BF16x2 32개<br>→ LN 미분 + residual → dX</div>
<div class="box"><b>compute WG B · 128 threads</b><br>행 B: dX 누산기 64 FP32<br>→ gate 합산 후 BF16x2 32개<br>→ LN 미분 + residual → dX</div>
<div class="wide muted">두 WG 모두 WGMMA를 마친 뒤 공용 W 슬롯을 재사용합니다. 두 번째 행의 dX 누산기를 shared에 옮겼다 읽는 이전 방식은 사용하지 않는 설계입니다.</div></div></section>
<section><h2>자원 예산부터 통과해야 합니다</h2><table><tr><th>항목</th><th>설계 조건</th></tr>
<tr><td>dX CTA 레지스터</td><td>producer 32 + compute 104 + compute 104 = 30,720 registers</td></tr>
<tr><td>source CTA 레지스터</td><td>producer 24 + compute 192 + inactive 24 = 30,720 registers</td></tr>
<tr><td>주 GEMM shared</td><td>행별 32 KiB × 2 + 공용 W 32 KiB = 96 KiB</td></tr>
<tr><td>gate/LN shared</td><td>행별 raw-x/dGate 32 KiB × 2 + 공용 Wgate 32 KiB = 96 KiB</td></tr>
<tr><td>실행 목표</td><td>spill 없이 2 CTA/SM. 실제 컴파일·occupancy API 확인 필요</td></tr></table>
<p><b>주의:</b> 384-thread launch는 source에도 적용됩니다. source producer 축소의 손해와 비활성 WG 비용을 먼저 측정해야 합니다. dX 부분만 좋아진 것으로 채택하지 않습니다.</p></section>
<section><h2>줄이려는 전송량</h2><p>단위 MiB. 소스 수준 논리 payload입니다. HBM·L2 실측값이나 예상 속도 향상률이 아닙니다. 꼬리 타일을 포함해 계산했습니다.</p>
<table><tr><th>항목</th><th>현재</th><th>설계 후보</th></tr>''' + table + '''</table></section>
<section><h2>이전 실패를 반복하지 않기 위한 확인 순서</h2><ol>
<li>현재 커널에서 BF16 포장만 변경: 정확도·시간·실제 live register 확인.</li>
<li>384-thread source 대조: producer 축소와 추가 WG의 손해 확인.</li>
<li>두-WG dX: register/shared 예산, WGMMA serialization, occupancy 확인.</li>
<li>L384 동일 프로세스 교차 timing → 정확도·sanitizer → 같은 cubin NCU.</li></ol>
<p>추가 gate 명령, 얕아진 derivative pipeline, 두 WG 사이 대기 비용이 공유 이득을 지울 수 있습니다. 252 µs·SoL90을 달성한다고 단정하지 않습니다.</p>
<p><a href="README.md">수식·버퍼 수명·동기화·구현 순서 상세</a> · <a href="cost_model.json">계산 원본</a> · <a href="../trimul_b7_l2_reuse_20260922/index.html">현재 선택의 실측 결과</a></p>
<p>H100 자원 한도: <a href="https://docs.nvidia.com/cuda/hopper-tuning-guide/">NVIDIA Hopper Tuning Guide</a>. Warpgroup register 재분배: <a href="https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/primitives.html">NVIDIA CUTLASS primitives</a>.</p></section></html>'''
(R / 'index.html').write_text(html)
status = root / 'TRIMUL_STATUS.html'
s = status.read_text()
marker = 'b7-resource-design-20260922'
if marker not in s:
    block = '<aside id="' + marker + '"><b>B7–B12 다음 구조 설계:</b> BF16 값의 레지스터 수명 단축 + 두 계산 warpgroup의 가중치 공유. 자원 예산과 이전 실패 원인 대조까지 완료; 새 성능 실측은 아직 없습니다. <a href="runs/trimul_b7_resource_design_20260922/index.html">배선·자원·전송량·검증 순서</a></aside>'
    assert '</header>' in s
    status.write_text(s.replace('</header>', '</header>' + block, 1))
print(json.dumps(data, indent=2, ensure_ascii=False))
