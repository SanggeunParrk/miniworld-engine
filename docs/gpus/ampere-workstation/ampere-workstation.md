# RTX A6000 / A5000 (sm86)


The RTX A5000 / A6000 (GA102 = `sm_86`) are **triton-only** targets, exactly like the
A100 (`sm_80`): every hand-CUDA path is gated behind `sm_90`
(`torch.cuda.get_device_capability()[0] < 9`), so `MINIWORLD` resolves to the portable
Triton family — no dispatch change. They live on the `cssb-master` cluster
(`partition=gpu`, `qos=normal`, A5000=`gpu02`/24 GB, A6000=`gpu01,03-05`/48 GB), which is
separate from the A100/H100/B200 cluster. One parameterized launcher per bench type covers
both cards (pick the card with `--gres`; the script auto-detects it and asserts `sm_86`):

```bash
# one module bench
srun -p gpu --gres=gpu:A6000:1 -c 8 --mem=64G \
  .pixi/envs/default/bin/python benchmarks/runners/bench.py target=transition level=module mode=inference

# the tuned cache for this card: one unit per (op, dtype, shape bucket), across every GPU given
srun -p gpu --gres=gpu:A6000:8 --exclusive \
  .pixi/envs/default/bin/python -m miniworld_engine.cli build all --gpus 8 --resume
```

The cache build replaced `CAPTURE_TARGET=all submits/run_autotune_capture_ampere.sbatch`:
capture used to be driven per bench target, which reached 48 of 91 triton kernels because a
module only fires the kernels its own shapes dispatch to. `build all` drives the DECLARED work
list instead — see the README CLI section and `docs/kernels/autotune-dispatch-cache.md`.

The A5000's 24 GB may OOM at the top of the sweep (L=1024, d=512); `bench.py` records those
points as `status=failed` rows rather than aborting, so the CSV still shows the memory cliff.

