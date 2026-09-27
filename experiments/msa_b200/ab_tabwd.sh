# ablations of the triattn backward: TA_ABL="NORED NODS ..." -> one build per flag set
cd $(dirname $0)
for f in ${TA_ABL:-base}; do
  src=../src/miniworld_engine/integrations/csrc/sm100/triattn_sm100.cu
  out=/NHNHOME/WORKSPACE/26mohw002_A/psk6950/.tmp/ta_$f.cu
  { [ "$f" = base ] || for x in ${f//_/ }; do echo "#define TA_$x 1"; done; cat $src; } > $out
  echo "== $f"; TA_SRC=$out TA_TAG=abl_$f python k_triattn_bwd.py 2>&1 | grep -E "sm100" 
done
