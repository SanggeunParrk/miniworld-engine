#!/bin/bash
cd /NHNHOME/WORKSPACE/26mohw002_A/psk6950/b2511/agents/n2-ife
F="features=esmfold2;swa_window_size=1000000;token_pool=relu"
B=foldjet_engine.kernels.swa_dit.cuda.sm100.tf32_bwd
for spec in "DWM_SPLIT=True,False" "DCG=True,False" "DXP=True,False" "DWS=True,False" "KSPLIT_DX=1,2,4" "KSPLIT_DC=1,2,3,4" "EARLY_DC=False,True" "DCX=False,True"; do
  echo "### $spec"
  /NHNHOME/WORKSPACE/26mohw002_A/psk6950/b2511/dev.sh -x -t 600 n2-ife python n1cfg/ifeab.py n2out/t4096.yaml "$B.$spec" "$F" 2>&1 | grep -v "USDT\|Warning\|return Variable" | cut -c1-300
done
echo DONE
