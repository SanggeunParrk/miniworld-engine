#!/bin/bash
# ncu_one.sh KERNEL_REGEX <launch_one.py args...>  -> median gpu__time_duration (us) of launches 2..4 at base clocks
X=$(cd "$(dirname "$0")" && pwd); re=$1; shift
/usr/local/cuda-12.9/bin/ncu --clock-control base -k "regex:$re" --launch-skip 1 --launch-count 3 --metrics gpu__time_duration.sum --csv \
  python -u $X/launch_one.py "$@" 2>/dev/null | python3 -c "
import csv,sys,statistics
v=[float(r['Metric Value'].replace(',','')) for r in csv.DictReader(l for l in sys.stdin if l.startswith('\"')) if r.get('Metric Name')=='gpu__time_duration.sum']
print('%.1f us (n=%d, spread %.1f)' % (statistics.median(v)/1e3 if v else -1, len(v), (max(v)-min(v))/1e3 if v else -1))"
