"""Update B1 only; preserve the existing B5-B12 diagram and geometry."""
from pathlib import Path
import xml.etree.ElementTree as ET
R=Path(__file__).resolve().parents[2]
changes={
 'partial dWproj: FP32 · 17.30 MB':'dWproj: 레지스터 유지 → B 종료 후 저장',
 '위 Phase A에서 출력한 dGate를 아래 Phase B가 다시 읽는다. 새 dGate 복사 버퍼를 만들지 않는다.':'dGate: L2 보존 힌트로 출력 → Phase B에서 재사용 → 소비한 타일의 캐시 우선순위를 낮춤.',
 'partial dWgate: FP32 · 8.65 MB':'partial dWg / dWp: B 끝에 저장',
 '# CTA end: store acc_dWg; final grid barrier.':'# CTA end: store acc_dWg AND acc_dWp; barrier.',
 'A+B partial: dW 25.95 MB + LN 270.34 kB':'B 종료 후 dW 25.95 MB + A의 LN 270.34 kB',
}
for name in ('TRIMUL_BACKWARD.svg','TRIMUL_BACKWARD_L768.svg'):
 p=R/name;s=p.read_text()
 for old,new in changes.items():
  assert old in s or new in s,(name,old)
  s=s.replace(old,new)
 ET.fromstring(s);p.write_text(s)
print('B1 deferred writes and cache reuse updated in both SVGs.')
