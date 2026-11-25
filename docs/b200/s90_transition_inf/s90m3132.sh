#!/bin/bash
# s90m.sh TAG BACKENDS MODE LENGTHS...: measure.speed of every Transition requirement case, one exclusive dev.sh process per length (s90-transition-inf tree)
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
tag=$1; be=$2; mode=$3; shift 3
o=$W/b2511/agents/s90-transition-inf/out; mkdir -p $o
for L in "$@"; do
  c=$o/m_${tag}_r3132_${mode}_$L.yaml
  cat > $c <<EOY
modules: [Transition]
backends: [$be]
external: []
rows: [31, 32]
dtypes: []
modes: [$mode]
lengths: [$L]
depths: min
samples: min
timing: [graph]
warmup: 5
iterations: 50
flush_l2: true
device: cuda
seed: 0
fixed: {bfloat16: 0.008, master: 0.008, float32: 1.0e-05}
EOY
  $W/b2511/dev.sh -x -g ${GPU:-2} -t 2400 s90-transition-inf python -m foldjet_engine.measure.speed $c --out $o/m_${tag}_r3132_${mode}_$L.jsonl > $o/m_${tag}_r3132_${mode}_$L.log 2>&1
  echo "$(date +%T) $tag $mode L$L rc=$? n=$(grep -c median_ms $o/m_${tag}_r3132_${mode}_$L.jsonl)"
done
