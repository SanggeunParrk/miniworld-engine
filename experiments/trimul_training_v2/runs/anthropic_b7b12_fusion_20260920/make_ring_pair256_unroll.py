"""Bounded unroll experiments for the two-row input-gradient role.

Candidates retain operation order and the original numerical limits. The loader
must reject any candidate with a stack frame or local-memory spills.
"""
from pathlib import Path
import json

R = Path(__file__).resolve().parent
base = (R / 'front_ring_pair256.cu').read_text()
config = json.loads((R / 'front_ring_pair256.launch.json').read_text())
for factor in (2, 4, 8):
    name = 'front_ring_pair256_lnq' + str(factor)
    src = base.replace('#pragma unroll 1\n  for(int q=0;q<8;',
                       '#pragma unroll %d\n  for(int q=0;q<8;' % factor)
    src = src.replace('#pragma unroll 1\n for(int q=0;q<16;',
                      '#pragma unroll %d\n for(int q=0;q<16;' % factor)
    assert src != base
    (R / (name + '.cu')).write_text(src)
    (R / (name + '.launch.json')).write_text(json.dumps(config))
    print(name)
