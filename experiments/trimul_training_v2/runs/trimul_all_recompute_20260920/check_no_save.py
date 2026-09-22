"""Small driver for compute-sanitizer on the new no-save K3 epilogue."""
import argparse
import json
import torch
import adapter as A

ap=argparse.ArgumentParser();ap.add_argument('--length',type=int,required=True)
n=ap.parse_args().length
with torch.no_grad():
    d=A.C.setup(n)
    cfg=json.loads((A.P/('training-k3-audit-L%d.json'%n)).read_text())['winner']
    for condition in ('dropout25','dropout_zero_output','changed_mask_input_weights'):
        if condition=='dropout_zero_output':d['ds'].zero_()
        if condition=='changed_mask_input_weights':
            d['ds'].fill_(1);d['mask'].copy_(1-d['mask']);d['x'].mul_(.91)
            d['leaves'][1].add_(.003);d['leaves'][5].mul_(.94)
        ref,_=A.F.forward(d,k3=cfg)
        for _ in range(3):
            y=A.forward_no_save(d);torch.cuda.synchronize()
            assert torch.equal(y,ref),condition
            if condition=='dropout_zero_output':assert torch.equal(y,d['x'])
        print('PASS',condition,n,flush=True)
