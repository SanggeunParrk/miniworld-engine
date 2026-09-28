#!/usr/bin/env bash
# Explicit code push; independent result pull. Never delete either side's files.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
key=${VAST_SSH_KEY:-/home/psk6950/.ssh/vast_engine_20260927}
host=${VAST_SSH_HOST:-root@202.122.49.242}
port=${VAST_SSH_PORT:-27779}
remote=${VAST_REMOTE_ROOT:-/workspace/miniworld-engine}
local_results="$repo/.bench/vast-20260927"
mkdir -p "$local_results"
transport="ssh -F /dev/null -i $key -o BatchMode=yes -o ConnectTimeout=15 -p $port"
case ${1:-status} in
  push)
    # Only tracked source and this helper; credentials and local environments are excluded.
    list=$(mktemp)
    trap 'rm -f "$list"' EXIT
    git -C "$repo" ls-files -z > "$list"
    printf '%s\0' AGENTS.md scripts/vast-sync.sh docs/operations/vast-h100.md >> "$list"
    shift
    for extra in "$@"; do
      [[ "$extra" != /* && "$extra" != *..* && -f "$repo/$extra" ]] || {
        echo "Extra source must be an existing relative file: $extra" >&2; exit 2;
      }
      printf '%s\0' "$extra" >> "$list"
    done
    ssh -F /dev/null -i "$key" -o BatchMode=yes -p "$port" "$host" 'mkdir -p /workspace/locks'
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    rsync -az --checksum --backup --backup-dir="/workspace/sync-backups/$stamp" \
      --rsync-path='flock -n -x /workspace/locks/source.lock rsync' \
      --from0 --files-from="$list" -e "$transport" "$repo/" "$host:$remote/"
    git -C "$repo" rev-parse HEAD > "$local_results/source-commit.txt"
    rsync -az -e "$transport" "$local_results/source-commit.txt" "$host:/workspace/setup/"
    ;;
  pull)
    rsync -az -e "$transport" "$host:/workspace/vast-results/" "$local_results/results/"
    rsync -az -e "$transport" "$host:/workspace/setup/" "$local_results/remote-setup/"
    ;;
  fetch-source)
    # Remote edits land separately for review, never overwrite local source.
    rsync -az --exclude='__pycache__/' --exclude='*.so' --exclude='build*/' \
      -e "$transport" "$host:$remote/src/" "$local_results/remote-source/src/"
    ;;
  watch)
    # Pull only, every two minutes, until today's midnight in Seoul (or override).
    exec 9>"$local_results/sync-watch.lock"
    flock -n 9 || { echo "A result watcher is already running." >&2; exit 0; }
    echo $$ > "$local_results/sync-watch.pid"
    trap 'rm -f "$local_results/sync-watch.pid"' EXIT
    deadline=${VAST_SYNC_UNTIL:-$(TZ=Asia/Seoul date -d 'tomorrow 00:00' +%s)}
    while (( $(date +%s) < deadline )); do
      "$0" pull || printf 'pull failed at %s\n' "$(date -Is)" >&2
      sleep 120
    done
    ;;
  run)
    gpu=${2:?usage: vast-sync.sh run GPU COMMAND [ARG...]}
    [[ "$gpu" == 0 || "$gpu" == 1 ]] || { echo 'GPU must be 0 or 1' >&2; exit 2; }
    shift 2
    (( $# )) || { echo 'A command is required' >&2; exit 2; }
    printf -v command '%q ' "$@"
    printf -v root_q '%q' "$remote"
    body="set -euo pipefail; source /workspace/setup/env.sh; cd $root_q; export CUDA_VISIBLE_DEVICES=$gpu TORCHINDUCTOR_CACHE_DIR=/workspace/cache/inductor-gpu$gpu TRITON_CACHE_DIR=/workspace/cache/triton-gpu$gpu MINIWORLD_ENGINE_JIT_ROOT=/workspace/cache/native-gpu$gpu TORCH_EXTENSIONS_DIR=/workspace/cache/extensions-gpu$gpu; exec $command"
    printf -v body_q '%q' "$body"
    ssh -F /dev/null -i "$key" -o BatchMode=yes -o ConnectTimeout=15 -p "$port" "$host" \
      "mkdir -p /workspace/locks && flock -n -s /workspace/locks/source.lock flock -n -x /workspace/locks/gpu$gpu.lock bash -c $body_q"
    ;;
  status)
    ssh -F /dev/null -i "$key" -o BatchMode=yes -o ConnectTimeout=15 -p "$port" "$host" \
      'hostname; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,power.draw --format=csv; tail -8 /workspace/setup/bootstrap.log'
    ;;
  *) echo 'usage: scripts/vast-sync.sh {push [NEW_FILE...]|pull|fetch-source|watch|status|run GPU COMMAND...}' >&2; exit 2 ;;
esac
