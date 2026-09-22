from pathlib import Path
base=Path('/home/psk6950/MiniWorld/runs/trimul_sm90_parity_20260917/engine/src/miniworld_engine/kernels/trimul_inproj/triton')
for name in ('bidirectional','unidirectional'):
 p=base/(name+'.py');s=p.read_text()
 if name=='bidirectional':
  s=s.replace('residual, dropscale=None):\n        B, L, _, D', 'residual, dropscale=None, output_backend="triton"):\n        B, L, _, D',1)
 else:
  s=s.replace('mask=None, residual=None, dropscale=None):\n        B, L, _, D','mask=None, residual=None, dropscale=None, output_backend="triton"):\n        B, L, _, D',1)
 marker='        if x_n.dtype == torch.bfloat16:\n            te_xn, mean_out, rstd_out'
 replacement='''        if output_backend == "anthropic_cuda":
            from miniworld_engine.kernels.trimul_inproj.cuda.anthropic_training import fused_output
            y, te_xn, mean_out, rstd_out, proj, gate = fused_output(
                tri, x_n, Wp, Wg, ln_out_w, ln_out_b, residual, dropscale, eps)
        elif x_n.dtype == torch.bfloat16:
            te_xn, mean_out, rstd_out'''
 assert marker in s;s=s.replace(marker,replacement,1)
 # Additional non-tensor argument in backward return.
 if name=='bidirectional':
  old='d_residual, None)'
 else: old='d_residual, None)'
 assert old in s;s=s.replace(old,'d_residual, None, None)',1)
 # public signature, optional keyword and explicit refusal before work
 marker='    dropscale=None,              # drop_row scale [B,1,L,D] (== mask/(1-p)); training only\n):'
 assert marker in s;s=s.replace(marker,marker[:-2]+'    *, output_backend="triton",\n):',1)
 marker='    d = pair.shape[-1]\n'
 check='''    if output_backend not in ("triton", "anthropic_cuda"):
        raise ValueError("Unknown TriMul training output backend")
    if output_backend == "anthropic_cuda" and (not torch.is_grad_enabled() and dropscale is None):
        raise ValueError("anthropic_cuda is an explicit training prototype; use the inference adapter for inference")
'''
 assert marker in s;s=s.replace(marker,check+marker,1)
 s=s.replace('        residual_flat, ds_2d,\n    )','        residual_flat, ds_2d, output_backend,\n    )',1)
 p.write_text(s)
