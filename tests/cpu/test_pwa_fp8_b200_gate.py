"""pwa_fp8_b200: the B200 PWA e4m3 route is OFF unless FOLDJET_ENGINE_PWA_FP8_B200=1 (FoldJet F3 RoseTTAFold3 n = 12 / 16 non-finite under CUDA graphs with
this route alone, 2026-11-24), and its scale cache is keyed on the weights' versions and shapes, not on the wv pointer alone. CPU only (no kernel runs)."""

import torch

from foldjet_engine.kernels.integrations import pwa_fp8_b200 as F


def _args(n=2048, s=64, d_msa=256, c=32):
    m = torch.zeros(1, s, n, d_msa, dtype=torch.bfloat16)
    lnw, lnb = torch.ones(d_msa), torch.zeros(d_msa)
    wv = torch.randn(8 * c, d_msa) * 0.05
    return (m, None, None, lnw, lnb, wv)


def test_off_by_default(monkeypatch):
    monkeypatch.setattr(F, "B200_ON", False)
    args = _args()
    w16 = torch.zeros(8, 2048, 2048, dtype=torch.bfloat16)
    v = torch.zeros(8, 2048, 64 * 32, dtype=torch.bfloat16)
    assert F.serves(args, w16, v) is None
    assert F.serves_fused(args, 2048) is None
    assert F.serves_fused_wide(args, 2048) is None


def test_scale_cache_follows_the_weight_version(monkeypatch):
    monkeypatch.setattr(F, "_CACHES", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)  # CPU job: no CUDA runtime to ask
    args = _args()
    c1 = F._scales(args, torch.device("cpu"))
    assert F._scales(args, torch.device("cpu")) is c1  # same tensors, same versions: the cached entry
    with torch.no_grad():
        args[5].mul_(4.0)  # in place: new version, a larger bound
    c2 = F._scales(args, torch.device("cpu"))
    assert c2 is not c1 and float(c2["sv"]) != float(c1["sv"])
