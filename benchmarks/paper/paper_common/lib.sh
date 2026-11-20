# Sourced by every card script: the measurement stages. Variables (all optional):
#   REP      repeat number 1..3 (a different Slurm job / ssh run per repeat; default 1)
#   EXTRA    extra arguments of paper_common/run_measure.py (smoke runs: EXTRA="--module TriangleMultiplication --max-cases 2")
#   BUDGET   seconds after which a job stops starting new cases (default 480 under Slurm's 10-minute limit; 0 = no limit)
# The card's env file (env_<card>.sh) sets ENGINE (the source tree), CARD, KIT_WRAP and the Python environment.
PAPER=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REP=${REP:-1}
EXTRA=${EXTRA:-}
BUDGET=${BUDGET:-480}

# stage EXP NAME CONFIG [WRAPPER] : measure CONFIG into EXP/records/<card>/NAME.r<REP>.jsonl (WRAPPER: the kit-stack wrapper)
stage() {
  local exp=$1 name=$2 cfg=$3 wrap=${4:-}
  echo "== $exp $name rep $REP $(date +%T)"
  # shellcheck disable=SC2086
  $wrap python "$PAPER/paper_common/run_measure.py" "$PAPER/$exp/$cfg" --kind "${KIND:-speed}" --out "${RECORDS:-$PAPER/$exp/records}/$CARD/$name.r$REP.jsonl" --budget-seconds "$BUDGET" ${EXT:+--ext $EXT} $EXTRA
}

# fp8_stage GATES : E4, the default path against the FP8 path in one process (E4_fp8/fp8_pair.py), records/<card>/fp8_<gates>.r<REP>.jsonl
fp8_stage() {
  local gates=$1
  echo "== E4 fp8 gates=$gates rep $REP $(date +%T)"
  # shellcheck disable=SC2086
  python "$PAPER/E4_fp8/fp8_pair.py" --gates "$gates" --out "${RECORDS:-$PAPER/E4_fp8/records}/$CARD/fp8_$gates.r$REP.jsonl" --budget-seconds "$BUDGET" $EXTRA
}

# E5: the external libraries are installed into PAPER_EXT_PATH (pip install --target, see E5_other_kernels/README.md), not into the engine env
[ -d "${PAPER_EXT_PATH:-$HOME/paper-ext}" ] && export PYTHONPATH=$PYTHONPATH:${PAPER_EXT_PATH:-$HOME/paper-ext}
