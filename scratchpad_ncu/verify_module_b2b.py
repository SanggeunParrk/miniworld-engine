"""Module-level correctness: transition module CUDA-b2b inference path vs torch reference."""
import os
import torch
from miniworld_kernels.modules.transition.module import Transition
from miniworld_kernels.modules.exceptions import ImplementationType

dev = "cuda"
torch.manual_seed(0)
B, L, d = 1, 1024, 128  # M = B*L*L = 1048576 (%128==0)

m = Transition(d_hidden=d, n=4, implementation=ImplementationType.TRITON).to(dev).eval()
# squeeze is zero-init by default -> randomize ALL weights so the test is non-degenerate
with torch.no_grad():
    m.expand_a.weight.normal_(0, 1.0 / d**0.5)
    m.expand_b.weight.normal_(0, 1.0 / d**0.5)
    m.squeeze.weight.normal_(0, 1.0 / (d * 4) ** 0.5)
    m.ln_in.weight.normal_(1.0, 0.1)
    m.ln_in.bias.normal_(0.0, 0.1)
for p in m.parameters():
    p.data = p.data.bfloat16()
x = torch.randn(B, L, L, d, device=dev, dtype=torch.bfloat16)

with torch.no_grad():
    ref = m._torch_forward(x).float()
    os.environ["MINIWORLD_TRANSITION_CUDA_B2B"] = "1"
    out_cuda = m._inference_forward(x).float()
    os.environ["MINIWORLD_TRANSITION_CUDA_B2B"] = "0"
    out_triton = m._inference_forward(x).float()

def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()

print(f"shape ref={tuple(ref.shape)} cuda={tuple(out_cuda.shape)}")
print(f"cos(cuda_b2b, torch)   = {cos(out_cuda, ref):.6f}")
print(f"cos(triton_b2b, torch) = {cos(out_triton, ref):.6f}")
print(f"cos(cuda_b2b, triton)  = {cos(out_cuda, out_triton):.6f}")
print(f"max|cuda-torch|        = {(out_cuda-ref).abs().max().item():.4e}")
