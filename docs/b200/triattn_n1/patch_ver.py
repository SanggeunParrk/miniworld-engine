import sys
root = sys.argv[1]; base = sys.argv[2]
p = root + "/src/miniworld_engine/modules/triangle.py"
s = open(base + "/src/miniworld_engine/modules/triangle.py").read()   # start again from the base file
b = '        "cuda:b200_sm100": 4,  # 4 ('
assert s.count(b) == 1
s = s.replace(b, '        "cuda:b200_sm100": 5,  # 5 (2026-11-18, N1 triattn, training): the LayerNorm forward / backward and gate backward (C = 128) on B200 kernels (b200_sources/lnbwd_sm100.cu, shared with the single-direction bias-only path of TriangleAttention), the dW / dWo GEMMs as graph branches at L <= 256 (dW alone above), outputs and gradients bitwise equal;\n        # 4 (', 1)
old = '    backend_versions: ClassVar[dict[str, int]] = {**_TriangleAttentionBase.backend_versions, "cuda:a100_sm80": 7, "cuda:h100_sm90": 10}'
assert s.count(old) == 1
new = '''    backend_versions: ClassVar[dict[str, int]] = {
        **_TriangleAttentionBase.backend_versions,
        "cuda:a100_sm80": 7,
        "cuda:h100_sm90": 10,
        # 71 (2026-11-18, N1 triattn, training; the base entry stays 69: only the single-direction bias-only form changes): the bias-only paths' LayerNorm
        # forward / backward and gate backward (C = 128, K = 128) on B200 kernels (b200_sources/lnbwd_sm100.cu: loads issued together and a row step ahead,
        # 2-4 CTAs an SM, d gamma / d beta column sum in one launch), the dW / dWo GEMMs as graph branches at L <= 256 (dW alone above): row 188 training
        # step -4.2 .. -11.5 % (measurer, L128-768), dx / dO / dG / y bitwise equal, gradients bitwise equal at module level.
        "cuda:b200_sm100": 71,
    }'''
s = s.replace(old, new)
open(p, "w").write(s); print("ok")
