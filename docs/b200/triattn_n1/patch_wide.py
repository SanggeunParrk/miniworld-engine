import sys
p = sys.argv[1] + "/src/foldjet_engine/kernels/triangle_attention/cuda/b200_triattn.py"; s = open(p).read()
old = '''    if TALLK and R <= TALLK_MAX_ROWS and C <= 256:
        G, dwo = _tallk().tallk(Dall, xh, dys, u)
    else:
        G = _mm32(Dall.t(), xh)
        dwo = _mm32(dys.t(), u)
    dt = dtypes
    fp32 = dt[2] == torch.float32
    dw, dwb, dwo, dgb = w.tri_wfinish(G, wcat, lnw, lnb, dwo, n_head, head_dim, fp32)
'''
new = '''    side = _side_mod() if WIDE_SIDE else None
    s_dwo = s_bias = None
    if TALLK and R <= TALLK_MAX_ROWS and C <= 256:
        G, dwo = _tallk().tallk(Dall, xh, dys, u)
    else:
        s_dwo = side.fork(x.device) if side is not None else None
        if s_dwo is not None:  # (N1) the dWo GEMM beside the G GEMM: independent, both split-K cuBLAS calls that leave SMs idle
            with _on(side, s_dwo):
                dwo = _mm32(dys.t(), u)
            G = _mm32(Dall.t(), xh)
            side.join(x.device, s_dwo)
        else:
            G = _mm32(Dall.t(), xh)
            dwo = _mm32(dys.t(), u)
    # the bias / second out-projection GEMMs read only dys, xh, u2: a branch beside the G GEMM and tri_wfinish, joined where their results are used
    ob_col = dwo2_raw = None
    s_bias = side.fork(x.device) if (side is not None and (ob is not None or out_mode == 2)) else None
    with _on(side, s_bias):
        if ob is not None:
            ob_col = _mm32(dys.t(), xh[: dys.shape[0], C : C + 1]).view(-1)  # (xh's ones column: any B L^2 of its rows)
        if out_mode == 2:
            dwo2_raw = _mm32(dys.t(), u2)
    dt = dtypes
    fp32 = dt[2] == torch.float32
    dw, dwb, dwo, dgb = w.tri_wfinish(G, wcat, lnw, lnb, dwo, n_head, head_dim, fp32)
'''
assert s.count(old) == 1
s = s.replace(old, new)
old2 = '''    if ob is not None:
        dbias[n_head * head_dim :] = _mm32(dys.t(), xh[: dys.shape[0], C : C + 1]).view(-1)  # (xh's ones column: any B L^2 of its rows)
    dwo2 = x.new_empty(0)
    if out_mode == 2:  # dWo2 = dy_s^T u2 (the padded channels dropped), in Wo's dtype
        dwo2 = _mm32(dys.t(), u2).view(C, n_head, S)[:, :, :head_dim].reshape(C, n_head * head_dim).to(dwo.dtype)
'''
new2 = '''    if s_bias is not None:
        side.join(x.device, s_bias)
    if ob is not None:
        dbias[n_head * head_dim :] = ob_col
    dwo2 = x.new_empty(0)
    if out_mode == 2:  # dWo2 = dy_s^T u2 (the padded channels dropped), in Wo's dtype
        dwo2 = dwo2_raw.view(C, n_head, S)[:, :, :head_dim].reshape(C, n_head * head_dim).to(dwo.dtype)
'''
assert s.count(old2) == 1
s = s.replace(old2, new2)
anchor = "BWD_PRE = os.environ.get(\"MINIWORLD_TRIATTN_B200_BWD_PRE\", \"1\") != \"0\"\n"
assert anchor in s
s = s.replace(anchor, anchor + "#: (N1) the wide backward's weight-gradient GEMMs (dWo, the bias column sum, dWo2) as branches of the captured graph beside the G GEMM / tri_wfinish.\n#: MINIWORLD_TRIATTN_B200_WIDE_SIDE=0 turns it off.\nWIDE_SIDE = os.environ.get(\"MINIWORLD_TRIATTN_B200_WIDE_SIDE\", \"1\") != \"0\"\n", 1)
open(p, "w").write(s); print("ok")
