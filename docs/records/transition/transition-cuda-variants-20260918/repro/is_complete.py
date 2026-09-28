import json,sys
from pathlib import Path
kind,d,l=sys.argv[1:]
p=Path(__file__).parent/f'{kind}-D{d}-L{l}.json'
try:
 r=json.loads(p.read_text());assert len(r['rows'])==(20 if kind=='module' else 5)
 if kind=='breakdown':assert 'isolated_common_backward_ms' in r
except (OSError,ValueError,KeyError,AssertionError):sys.exit(1)
