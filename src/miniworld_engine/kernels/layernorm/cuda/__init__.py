"""CUDA implementations of LayerNorm: ``general`` (any width, forward and backward) and ``pipe`` (the pipelined H100 forward /
backward). The legacy warp-per-row backward (``layer_norm_cuda_kernel.cu``) was removed: no requirements class reached it."""


def prebuild(target, rows):
    """Every build of the LayerNorm family (kernels/_prebuild.py): the pipelined and the general-width extensions, each when a row
    needs one of their kernels (the forward of the main kernels is Triton)."""
    if target.arch == "sm80":   # A100: the sm_80 row kernels (sm80.py), for the LayerNorm / RMSNorm rows
        from . import sm80

        if any(row["module"] in ("LayerNorm", "RMSNorm") for row in rows):
            yield sm80.ext
        return
    from . import general, pipe

    if any(row["kernels"] & {"layernorm_pipe_direct_fwd", "layernorm_pipe_direct2_fwd", "layernorm_pipe_bulk_fwd", "layernorm_pipe_bwd",
                             "layernorm_pipe_reduce_bwd", "layernorm_pipe_fwd"} for row in rows):
        yield pipe._ext
    if any(row["kernels"] & {"layernorm_general_fwd", "layernorm_general_bwd", "layernorm_general_bwd_reduce"} for row in rows):
        yield general._ext
