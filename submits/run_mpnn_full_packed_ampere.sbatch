#!/bin/bash
# Full ProteinMPNN model in the packed training layout, at the batch/length grid the
# mpnn_edge_tail kernel bench uses, so kernel and model numbers can be read against each
# other. Timing, parity and peak memory are three phases of one job because they answer
# different questions and have different ceilings.
#
# mpnn_layout=packed with batch_size=B seq_len=T is physical B=1 carrying B logical
# samples concatenated to B*T nodes -- exactly the `nodes = batch_size * seq_len` the
# kernel bench builds. Cost tracks the product, so B=4 T=8192 and B=8 T=4096 are the same
# 32,768 nodes; measure rather than assume.
#
# compile=true is genuinely in effect: bench_mpnn is one of the five module benches that
# call model.compile() ungated. bf16-mixed is autocast BF16 over FP32 masters, so the
# CSV's float32 dtype columns are the masters, not the arithmetic.
#
# Phase ceilings differ and that is a result, not a failure:
#   time    check_correctness=false, so only the model itself has to fit
#   parity  needs the reference model live alongside it, so it stops one step earlier
#   memory  metric=memory short-circuits ahead of the cudagraph branches, so no graph
#           pool inflates the high-water mark
# bench.py records anything that does not fit as status=failed and carries on.
#
#   BATCH=8 sbatch --gres=gpu:A6000:1 submits/run_mpnn_full_packed_ampere.sbatch
#   BATCH=4 PHASES="time memory" sbatch --gres=gpu:A6000:1 submits/run_mpnn_full_packed_ampere.sbatch
#SBATCH --job-name=mwk_mpnn_full_packed
#SBATCH --partition=gpu
#SBATCH --account=cssb
#SBATCH --qos=normal
#SBATCH --gres=gpu:A6000:1
#SBATCH --exclude=gpu01
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=benchmarks/modules/mpnn_full_packed_%j.out

set -uo pipefail
set -f
cd /home/psk6950/practice/miniworld-kernels
export HYDRA_FULL_ERROR=1
export TRITON_CACHE_DIR="/tmp/triton_mpnn_packed_${SLURM_JOB_ID:-$$}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

BATCH="${BATCH:-8}"
LAYERS="${LAYERS:-3}"
PHASES="${PHASES:-time parity memory}"
SWEEP="${SWEEP:-[2048,4096,8192]}"
PARITY_SWEEP="${PARITY_SWEEP:-[512,1024,2048,4096,8192]}"
RUN_ID="${SLURM_JOB_ID:-manual_$$}"
# The generated CSV name leaves only 21 bytes for name_suffix once
# cudagraph-manual is in it, and OSError 36 is what you get for spending 22.
TAG="p${BATCH}L${LAYERS}"

echo "host=$(hostname) date=$(date) job=${SLURM_JOB_ID:-none}"
echo "BATCH=$BATCH LAYERS=$LAYERS PHASES='$PHASES'"
pixi run --frozen python submits/_assert_device.py --capability 8.6 --name A5000 --name A6000 || exit 3

run() {  # $1=phase $2=metric $3=check $4=cudagraph $5=seq_len_values
  echo ""
  echo "############################################################"
  echo "# $1  B=$BATCH L=$LAYERS metric=$2 check=$3 cudagraph=$4  $(date +%T)"
  echo "############################################################"
  pixi run --frozen bash -c \
    'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}; PYTHONPATH=src python benchmarks/runners/bench.py "$@"' _ \
    kernel=mpnn implementations='[pytorch,miniworld]' \
    mode=training precision=bf16-mixed allow_tf32=true \
    n_layers="$LAYERS" n_augment=1 mask_prob=0.0 \
    batch_size="$BATCH" mpnn_layout=packed mpnn_patch_size=8 \
    d_pair=128 k_neighbors=48 \
    sweep_axis=seq_len seq_len_values="$5" sweep_seq_len=8192 \
    metric="$2" check_correctness="$3" \
    compile=true cudagraph="$4" \
    name_suffix="${TAG}${1:0:1}${RUN_ID}"
  echo "[exit=$? phase=$1]"
}

for phase in $PHASES; do
  case "$phase" in
    time)   run time   time   false manual   "$SWEEP" ;;
    parity) run parity time   true  disabled "$PARITY_SWEEP" ;;
    memory) run memory memory false disabled "$SWEEP" ;;
    *) echo "unknown phase '$phase'" >&2; exit 2 ;;
  esac
done

echo ""
echo "======== summary  B=$BATCH L=$LAYERS ========"
DEV="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
pixi run --frozen python - "$DEV" "$RUN_ID" <<'PYEOF'
import csv, pathlib, sys
dev, rid = sys.argv[1], sys.argv[2]
root = pathlib.Path("benchmarks/modules/mpnn/artifacts") / dev
num = lambda r, k: (f"{float(r[k]):.6f}" if r.get(k) else "-")  # noqa: E731
for path in sorted(root.glob(f"*{rid}*.csv")):
    rows = list(csv.DictReader(path.open()))
    # From the rows, not the filename. Parsing the name for the phase letter got every
    # label wrong once already, and the CSV states the answer outright.
    phase = ("memory" if rows and rows[0]["metric"] == "memory"
             else "parity" if any(r["correctness_checked"] == "True" for r in rows)
             else "time")
    print(f"\n### {phase.upper()}")
    if phase == "parity":
        print(f"  {'impl':<11}{'L':>6}{'checked':>9}{'out_cos':>12}{'out_rel':>12}"
              f"{'grad_cos':>12}{'grad_rel':>12}   status")
        for r in rows:
            print(f"  {r['implementation']:<11}{r['seq_len']:>6}{r['correctness_checked']:>9}"
                  f"{num(r,'output_cosine'):>12}{num(r,'output_rel_frob'):>12}"
                  f"{num(r,'grad_cosine'):>12}{num(r,'grad_rel_frob'):>12}   {r['status']}")
    elif phase == "memory":
        print(f"  {'impl':<11}{'L':>6}{'peak MiB':>12}{'vs pytorch':>12}   status")
        base = {r['seq_len']: float(r['value'])
                for r in rows if r['implementation'] == 'pytorch' and r['status'] == 'ok'}
        for r in rows:
            ok = r['status'] == 'ok'
            v = float(r['value']) if ok and r['value'] else float('nan')
            b = base.get(r['seq_len'])
            print(f"  {r['implementation']:<11}{r['seq_len']:>6}{v:>12.0f}"
                  f"{(f'{b/v:.2f}x' if ok and b and v else '-'):>12}   {r['status']}")
    else:
        print(f"  {'impl':<11}{'L':>6}{'ms':>12}{'vs pytorch':>12}   status")
        base = {r['seq_len']: float(r['value'])
                for r in rows if r['implementation'] == 'pytorch' and r['status'] == 'ok'}
        for r in rows:
            ok = r['status'] == 'ok'
            v = float(r['value']) if ok and r['value'] else float('nan')
            b = base.get(r['seq_len'])
            print(f"  {r['implementation']:<11}{r['seq_len']:>6}{v:>12.2f}"
                  f"{(f'{b/v:.2f}x' if ok and b and v else '-'):>12}   {r['status']}")
    for r in rows:
        if r['status'] != 'ok':
            print(f"      {r['implementation']} L={r['seq_len']}: {(r['error'] or '')[:110]}")
PYEOF
echo "MPNN FULL PACKED DONE B=$BATCH L=$LAYERS"
