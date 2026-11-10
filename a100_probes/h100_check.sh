set -x
ruff check src/miniworld_engine/kernels/integrations/augattn_sm80.py src/miniworld_engine/modules/token_dit.py src/miniworld_engine/modules/smooth_lddt.py src/miniworld_engine/kernels/smooth_lddt/cuda/sm100/__init__.py src/miniworld_engine/kernels/augmented_attention/cuda/prebuild_sm80.py
rm -rf /tmp/w3base && mkdir -p /tmp/w3base && git archive e67d3fb5d src | tar -x -C /tmp/w3base
for a in sm90a sm80; do
  python -m miniworld_engine.kernels._prebuild --arch $a --list > /tmp/w3_new_$a.txt 2>&1
  PYTHONPATH=/tmp/w3base/src python -m miniworld_engine.kernels._prebuild --arch $a --list > /tmp/w3_base_$a.txt 2>&1
  echo "== $a lines new=$(wc -l < /tmp/w3_new_$a.txt) base=$(wc -l < /tmp/w3_base_$a.txt)"; diff /tmp/w3_base_$a.txt /tmp/w3_new_$a.txt && echo "$a IDENTICAL"
done
python -m miniworld_engine.judge --gpu h100_sm90 owed 2>&1 | tail -15
PYTHONPATH=/tmp/w3base/src python -m miniworld_engine.judge --gpu h100_sm90 owed 2>&1 | tail -5
python -m miniworld_engine.judge --gpu a100_sm80 owed 2>&1 | tail -15
