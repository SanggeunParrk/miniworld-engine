"""nvcc choice for the JIT build: the torch CUDA release, else the same major, never silently another major (a CUDA 13 build next to a
cu129 torch links libcublasLt.so.13 and crashed cublasLtMatmulAlgoGetHeuristic: TriangleAttention out_gemm_residual, RTX 5090)."""

import warnings

from foldjet_engine.kernels import _nvcc


def _t(rel):
    return f"Cuda compilation tools, release {rel}, V{rel}.61"


def test_exact_release_wins_over_order():
    found = [("/c/13.1", _t("13.1")), ("/c/12.9", _t("12.9")), ("/c/12.8", _t("12.8"))]
    assert _nvcc.pick_nvcc(["12", "9"], found) == "/c/12.9"


def test_same_major_beats_a_first_listed_other_major():
    found = [("/usr/local/cuda-13.1", _t("13.1")), ("/usr/local/cuda-12.8", _t("12.8")), ("/usr/local/cuda", _t("13.1"))]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _nvcc.pick_nvcc(["12", "9"], found) == "/usr/local/cuda-12.8"


def test_closest_minor_of_the_same_major():
    found = [("/c/12.4", _t("12.4")), ("/c/12.8", _t("12.8"))]
    assert _nvcc.pick_nvcc(["12", "9"], found) == "/c/12.8"


def test_only_another_major_warns_and_keeps_the_first():
    import pytest

    with pytest.warns(UserWarning, match="no nvcc of CUDA 12"):
        assert _nvcc.pick_nvcc(["12", "9"], [("/c/13.1", _t("13.1"))]) == "/c/13.1"


def test_nothing_found():
    assert _nvcc.pick_nvcc(["12", "9"], []) is None
