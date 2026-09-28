#!/usr/bin/env bash
set -euo pipefail
out=${1:?result directory}
shift
mkdir -p "$out"
completed() {
  python - "$1" "$2" <<'PY'
import hashlib, json, pathlib, sys
try:
    r = json.loads(pathlib.Path(sys.argv[1]).read_text())
    assert r['complete']
    d128_only = {'src/miniworld_engine/kernels/transition/cuda/fused_sm90a.py',
                 'src/miniworld_engine/kernels/transition/cuda/transition_fused_bwd_sm90a_kernel.cu'}
    for name, digest in r['source_sha256'].items():
        if r['D'] != 128 and name in d128_only:
            continue
        assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() == digest
    assert 'ERROR SUMMARY: 0 errors' in pathlib.Path(sys.argv[2]).read_text()
except (OSError, ValueError, KeyError, AssertionError):
    sys.exit(1)
PY
}
for d in "$@"; do
  for tool in memcheck racecheck synccheck; do
    # Long shape exercises persistent CTA refills; the full short shape is also
    # checked for memory safety separately below.
    if [[ "$tool" == racecheck && "$d" != 128 ]]; then
      filters=()
      stem=race-once
      harness=experiments/transition_shapes_vast/race_once.py
      export RACE_KERNEL_FILTER=all
      if (( d >= 384 )); then
        filters=(--kernel-name kns=wide_)
        stem=race-isolated-v2
        harness=experiments/transition_shapes_vast/race_kernels.py
        export RACE_KERNEL_FILTER=kns=wide_
      fi
      /workspace/tools/sanitizer132/bin/compute-sanitizer --tool racecheck --error-exitcode 86 "${filters[@]}" \
        python "$harness" --width "$d" \
        --out "$out/$stem" > "$out/$stem-D$d-L768.log" 2>&1
    elif ! completed "$out/$tool/D$d-L768.json" "$out/$tool-D$d-L768.log"; then
      /workspace/tools/sanitizer132/bin/compute-sanitizer --tool "$tool" --error-exitcode 86 \
        python experiments/transition_shapes_vast/bench.py --sanitize --width "$d" --length 768 \
        --out "$out/$tool" > "$out/$tool-D$d-L768.log" 2>&1
    fi
  done
  /workspace/tools/sanitizer132/bin/compute-sanitizer --tool memcheck --error-exitcode 86 \
    python experiments/transition_shapes_vast/bench.py --sanitize --width "$d" --length 384 \
    --out "$out/memcheck" > "$out/memcheck-D$d-L384.log" 2>&1
done
