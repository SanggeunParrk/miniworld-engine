# Sourced by every card script: the measurement stages. Variables (all optional):
#   REP      repeat number 1..3 (a different Slurm job / ssh run per repeat; default 1)
#   EXTRA    extra arguments of paper_common/run_measure.py (smoke runs: EXTRA="--module TriangleMultiplication --max-cases 2")
#   BUDGET   seconds after which a job stops starting new cases (default 480 under Slurm's 10-minute limit; 0 = no limit)
# The card's env file (env_<card>.sh) sets ENGINE (the source tree), CARD, KIT_WRAP and the Python environment.
PAPER=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REP=${REP:-1}
EXTRA=${EXTRA:-}
BUDGET=${BUDGET:-480}   # seconds per JOB: shared by all stages of the job (remaining budget = BUDGET - SECONDS), so a 2-stage script stays inside Slurm 10 min

# stage EXP NAME CONFIG [WRAPPER] : measure CONFIG into EXP/records/<card>/NAME.r<REP>.jsonl (WRAPPER: the kit-stack wrapper)
stage() {
  local exp=$1 name=$2 cfg=$3 wrap=${4:-}
  if [ -n "${ONLY:-}" ] && [ "$ONLY" != "$name" ]; then return 0; fi   # ONLY=engine|kit|ext|shipped|none: one stage of the script (the lane queue splits tasks by stage)
  if [ "$BUDGET" -gt 0 ] && [ $((BUDGET - SECONDS)) -lt 45 ]; then echo "== $exp $name: job budget used up, next job continues"; return 0; fi
  echo "== $exp $name rep $REP $(date +%T)"
  # shellcheck disable=SC2086
  $wrap python "$PAPER/paper_common/run_measure.py" "$PAPER/$exp/$cfg" --kind "${KIND:-speed}" --out "${RECORDS:-$PAPER/$exp/records}/$CARD/$name.r$REP.jsonl" --budget-seconds "$(job_budget)" ${EXT:+--ext $EXT} $EXTRA
}

# fp8_stage GATES : E4, the default path against the FP8 path in one process (E4_fp8/fp8_pair.py), records/<card>/fp8_<gates>.r<REP>.jsonl
fp8_stage() {
  local gates=$1
  if [ -n "${ONLY:-}" ] && [ "$ONLY" != "$gates" ]; then return 0; fi
  echo "== E4 fp8 gates=$gates rep $REP $(date +%T)"
  # shellcheck disable=SC2086
  python "$PAPER/E4_fp8/fp8_pair.py" --gates "$gates" --out "${RECORDS:-$PAPER/E4_fp8/records}/$CARD/fp8_$gates.r$REP.jsonl" --budget-seconds "$(job_budget)" $EXTRA
}


# job_budget : seconds left of the per-job BUDGET (min 1: run_measure then stops before its first case; 0 = no limit)
job_budget() { local r=$((BUDGET - SECONDS)); [ "$BUDGET" -eq 0 ] && { echo 0; return; }; [ $r -lt 1 ] && r=1; echo $r; }
