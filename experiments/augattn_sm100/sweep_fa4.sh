#!/bin/bash
# sweep_fa4.sh [L ...] -- FA4-style core variants (exp ping-pong PP, polynomial exp share PCNT/PMOD) against the current kernels.
# Refuses to measure while any other process holds GPU 6 (the colleague's jobs share the box).
set -u
cd "$(dirname "$0")"
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950
source $W/miniworld-engine/.venv/bin/activate
export PATH=/usr/local/cuda/bin:$PATH TMPDIR=$W/.tmp XDG_CACHE_HOME=$W/.cache
uuid6=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==6{print $2}')
busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$uuid6")
if [ "$busy" != "0" ]; then echo "GPU 6 is busy ($busy processes): not measuring"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$uuid6"; exit 1; fi
for k in attn_fwd2 attn_dqb attn_dkv; do
  OUT=${k}_nopp bash build.sh $k -DPP=0 >/dev/null
  OUT=${k}_pp bash build.sh $k >/dev/null
  for pc in 1 2; do OUT=${k}_pp_p${pc} bash build.sh $k -DPCNT=$pc >/dev/null; done
  OUT=${k}_pp_m8p3 bash build.sh $k -DPMOD=8 -DPCNT=3 >/dev/null
done
for L in ${@:-384 768}; do
  for kind in fwd dqb dkv; do
    k=attn_$kind; [ $kind = fwd ] && k=attn_fwd2
    CUDA_VISIBLE_DEVICES=6 L=$L python t_core.py $kind build/${k}_nopp.cubin build/${k}_pp.cubin build/${k}_pp_p1.cubin build/${k}_pp_p2.cubin build/${k}_pp_m8p3.cubin build/${k}_nopp.cubin
  done
done
