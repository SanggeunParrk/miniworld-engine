#!/bin/bash
cd /NHNHOME/WORKSPACE/26mohw002_A/psk6950/b2511/agents/n2-ife
F="features=esmfold2;swa_window_size=1000000;token_pool=relu"
E=miniworld_engine.modules.embedder; A=miniworld_engine.modules.atom_dit; B=miniworld_engine.kernels.swa_dit.cuda.sm100.tf32_bwd
for spec in "$E._FEAT_DWK=False,True" "$E._ROPE_SIDE=False,True" "$E._SCATTER8=False,True" "$B.DWM_KERNEL=False,True" "$B.SILU_ONCE=False,True" "$A._B200_MOD0_MAIN=False,True" "$A._B200_EARLY_FORMS=False,True"; do
  echo "### $spec"
  /NHNHOME/WORKSPACE/26mohw002_A/psk6950/b2511/dev.sh -x -t 600 n2-ife python n1cfg/ifeab.py n2out/t8192.yaml "$spec" "$F" 2>&1 | grep -v "USDT\|Warning\|return Var\|build" | sed "s/== InputFeatureEmbedder.*swa_wi/==/" | cut -c1-300 | tail -1
done
echo DONE
