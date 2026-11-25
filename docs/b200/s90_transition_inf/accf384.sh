#!/bin/bash
W=/NHNHOME/WORKSPACE/26mohw002_A/psk6950; . $W/b2511/env.sh
cd $W/b2511/agents/s90-transition-inf
$W/b2511/dev.sh -g 2 -t 1200 s90-transition-inf python -m foldjet_engine.measure.accuracy b200cfg/acc3132.yaml --out out/acc3132_f.jsonl > out/acc3132_f.log 2>&1
echo "acc rc=$?"; grep -o '"light": "[A-Z]*"' out/acc3132_f.jsonl | sort | uniq -c
CUDA_VISIBLE_DEVICES= $ENV/bin/python -I - <<EOT
import json
for l in open("out/acc3132_f.jsonl"):
    r=json.loads(l); c=r.get("case",r); res=r.get("result",r)
    if c.get("backend")=="cuda": print(c.get("row"), c.get("length"), res.get("light"), round(res.get("err_backend",0)/max(res.get("tol",1),1e-30),3))
EOT
