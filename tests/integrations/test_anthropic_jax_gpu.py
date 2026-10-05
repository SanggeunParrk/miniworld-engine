"""Original Pallas rows, including genuine backward kernels, through engine entry points.

Run in the isolated JAX 0.10.2/CUDA12 environment; no PyTorch autograd substitute.
"""
import os

import numpy as np
import pytest

from miniworld_engine.integrations import anthropic as A

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not os.environ.get("MINIWORLD_ANTHROPIC_ROOT"), reason="needs pinned upstream")]
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


@pytest.fixture(autouse=True)
def runtime():
    A.configure()
    assert jax.devices()[0].platform == "gpu"


def rand(*shape, scale=1):
    return jnp.asarray(np.random.default_rng(sum(shape)).normal(size=shape) * scale, dtype=jnp.bfloat16)


def close(actual, expected, tol=.04):
    a, b = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    assert np.isfinite(a).all()
    assert np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-8) < tol


def test_pallas_layer_norm_forward_and_backward():
    x, w, b = rand(128, 128), rand(128, scale=.05).astype(jnp.float32)+1, rand(128, scale=.05).astype(jnp.float32)
    serve = A.provider("pallas")
    from opt_core.kernels.pallas import serve as S
    fn = lambda x, w, b: A.pallas_call("layer_norm", x, w, b, word="cd_ln", direction="fwdbwd")
    ref = S.reference_layer_norm
    close(jax.jit(fn)(x,w,b), jax.jit(ref)(x,w,b))
    dy=rand(*x.shape)
    for got,want in zip(jax.vjp(fn,x,w,b)[1](dy),jax.vjp(ref,x,w,b)[1](dy),strict=True):
        close(got,want)
    assert serve.has_backward("cd_ln")


@pytest.mark.parametrize("row", ["cd_transition", "mlp_transition"])
def test_pallas_transition(row):
    from opt_core.kernels.pallas import serve as S
    x=rand(128,128)
    p={"ln_scale":jnp.ones(128),"ln_offset":jnp.zeros(128),"w1":rand(128,256,scale=128**-.5),
       "b1":rand(256,scale=.1),"w2":rand(256,128,scale=256**-.5),"b2":rand(128,scale=.1)}
    fn=lambda x,p:A.pallas_call("transition",x,p,activation="relu",word=row,
                               direction="fwdbwd" if row=="cd_transition" else "fwd")
    ref=lambda x,p:S.reference_transition(x,p,activation="relu")
    close(jax.jit(fn)(x,p),jax.jit(ref)(x,p))
    if row=="cd_transition":
        dy=rand(*x.shape)
        got=jax.vjp(fn,x,p)[1](dy)
        want=jax.vjp(ref,x,p)[1](dy)
        for g,w in zip(jax.tree.leaves(got),jax.tree.leaves(want),strict=True):
            close(g,w,.08)


@pytest.mark.parametrize("row", ["cd_triatt", "pallas_attn", "fpf_core", "triattn_xla"])
def test_jax_attention(row):
    from opt_core.kernels.pallas import serve as S
    q,k,v=(rand(2,128,4,32,scale=s) for s in (1,.8,.6))
    bias=rand(4,128,128,scale=.2).astype(jnp.float32)
    mask=jnp.broadcast_to(jnp.arange(128)%7!=0,(2,128))
    if row == "fpf_core":
        # The measured N199 bucket refuses this row for its unaligned size.
        # Qualify the explicitly named stage at the actual aligned N128 shape.
        fn=lambda q,k,v,b:A.pallas_components().attention_core(q,k,v,b,mask,cc="8.0")
    else:
        fn=lambda q,k,v,b:A.pallas_call("attention",q,k,v,b,mask,word=row,
                                       direction="fwdbwd" if row in ("cd_triatt","pallas_attn") else "fwd")
    ref=lambda q,k,v,b:S.reference_attention(q,k,v,b,mask)
    close(jax.jit(fn)(q,k,v,bias),jax.jit(ref)(q,k,v,bias))
    if row in ("cd_triatt","pallas_attn"):
        dy=rand(*q.shape)
        for g,w in zip(jax.vjp(fn,q,k,v,bias)[1](dy),jax.vjp(ref,q,k,v,bias)[1](dy),strict=True):
            close(g,w,.06)


def test_pallas_glut():
    from tokamax._src.ops.gated_linear_unit.pallas_triton import Config

    x=rand(128,128,128)
    w=rand(128,2,128,scale=128**-.5)
    mask=jnp.broadcast_to(jnp.arange(128)%7!=0,(128,128))
    cfg=Config(block_m=64,block_n=64,block_k=32,num_warps=4,num_stages=2)
    fn=lambda x,w:A.pallas_call("glu_transposed_masked",x,w,mask,word="glut",config=cfg)
    projected=jnp.einsum("ijc,coh->ijoh",x,w)
    ref=(jax.nn.swish(projected[:,:,0,:])*projected[:,:,1,:]).transpose(2,0,1)*mask[None]
    close(jax.jit(fn)(x,w),ref)


@pytest.mark.parametrize("row", ["cd_trimul", "fpf_trimul_xlabwd"])
def test_pallas_trimul_forward_and_backward(row):
    from opt_core.kernels.pallas import serve as S
    x=rand(128,128,128)
    mask=jnp.broadcast_to(jnp.arange(128)%7!=0,(128,128)).astype(jnp.bfloat16)
    p={key:(jnp.ones(128) if key.endswith("scale") else jnp.zeros(128))
       for key in ("ln_in_scale","ln_in_offset","ln_c_scale","ln_c_offset")}
    for key in ("left_w","right_w","left_gate_w","right_gate_w","out_w","gate_w"):
        p[key]=rand(128,128,scale=128**-.5)
    fn=lambda x,p:A.pallas_call("triangle_multiplication",x,mask,p,equation="outgoing",word=row,direction="fwdbwd")
    ref=lambda x,p:S.reference_triangle_multiplication(x,mask,p,equation="outgoing")
    close(jax.jit(fn)(x,p),jax.jit(ref)(x,p),.06)
    dy=rand(*x.shape)
    for g,w in zip(jax.tree.leaves(jax.vjp(fn,x,p)[1](dy)),jax.tree.leaves(jax.vjp(ref,x,p)[1](dy)),strict=True):
        close(g,w,.10)


@pytest.mark.parametrize("row", ["cd_opm", "opm_two_launch"])
def test_pallas_opm(row):
    from opt_core.kernels.pallas import serve as S
    x=rand(64,128,64)
    mask=jnp.broadcast_to(jnp.arange(128)%7!=0,(64,128)).astype(jnp.bfloat16)
    p={"ln_scale":jnp.ones(64),"ln_offset":jnp.zeros(64),"left_w":rand(64,32,scale=.125),
       "right_w":rand(64,32,scale=.125),"left_b":rand(32,scale=.1),"right_b":rand(32,scale=.1),
       "out_w":rand(32,32,128,scale=1/32),"out_b":rand(128,scale=.1)}
    fn=lambda x,p:A.pallas_call("outer_product_mean",x,mask,p,word=row)
    ref=lambda x,p:S.reference_outer_product_mean(x,mask,p)
    close(jax.jit(fn)(x,p),jax.jit(ref)(x,p),.06)


@pytest.mark.parametrize("orientation", ["starting", "ending"])
def test_pallas_triangle_attention_block(orientation):
    from opt_core.kernels.pallas import serve as S

    x = rand(128, 128, 128)
    # Keep at least one key in both orientations. The original Pallas VJP's
    # all-empty-row behavior is outside this qualification (recorded separately).
    mask = (jnp.arange(128)[:, None] + jnp.arange(128)[None, :]) % 7 != 0
    p = {"ln_scale": jnp.ones(128), "ln_offset": jnp.zeros(128),
         "bias_w": rand(128, 4, scale=128**-.5),
         "out_w": rand(4, 32, 128, scale=128**-.5)}
    for key in ("q_w", "k_w", "v_w", "gate_w"):
        p[key] = rand(128, 4, 32, scale=128**-.5)
    fn = lambda x, p: A.pallas_call("triangle_attention_block", x, mask, p,
                                    orientation=orientation, word="proj+pallas_attn", direction="fwdbwd")
    ref = lambda x, p: S.reference_triangle_attention(x, mask, p, orientation=orientation)
    close(jax.jit(fn)(x, p), jax.jit(ref)(x, p), .06)
    dy = rand(*x.shape)
    for g, w in zip(jax.tree.leaves(jax.vjp(fn, x, p)[1](dy)),
                    jax.tree.leaves(jax.vjp(ref, x, p)[1](dy)), strict=True):
        close(g, w, .10)


def test_native_trimul_xla_retains_architecture_refusal():
    provider = A.provider("trimul_xla")
    if jax.devices()[0].compute_capability != "8.0":
        pytest.skip("this contract is specific to A100")
    weights = {"ln_in_w": jnp.ones(128), "ln_in_b": jnp.zeros(128),
               "ln_out_w": jnp.ones(128), "ln_out_b": jnp.zeros(128)}
    for key in ("w_ag", "w_ap", "w_bg", "w_bp", "w_o", "w_og"):
        weights[key] = rand(128, 128)
    with pytest.raises(provider.Refused, match="arch"):
        A.triangle_multiplication_xla(rand(128, 128, 128), weights=weights, direction="outgoing", cc="8.0")
