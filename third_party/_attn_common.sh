#!/bin/bash
# Shared helpers of the third_party/<name>/fetch.sh scripts of the pair-bias-attention / Pairformer-operator baselines (E5 ext_attn).
# DEST: where the upstream tree goes (default <name>/upstream, gitignored); on a card the adapters read $PAPER_EXT_ATTN/<name> (default ~/paper-ext-attn/<name>):
#   DEST=$HOME/paper-ext-attn/<name> third_party/<name>/fetch.sh
# No python is needed except for deepspeed (pip --target). Idempotent: an existing tree at the pinned revision / hash is kept.

# git_pin REPO REV DEST : a shallow checkout of REV (a full commit hash)
git_pin() {
  local repo=$1 rev=$2 dest=$3
  if [ -d "$dest/.git" ] && [ "$(git -C "$dest" rev-parse HEAD 2>/dev/null)" = "$rev" ]; then echo "[fetch] $dest already at $rev"; return 0; fi
  mkdir -p "$dest" && git -C "$dest" init -q && git -C "$dest" remote add origin "$repo" 2>/dev/null || true
  git -C "$dest" fetch -q --depth 1 origin "$rev" && git -C "$dest" checkout -q FETCH_HEAD
  [ "$(git -C "$dest" rev-parse HEAD)" = "$rev" ] || { echo "[fetch] FAILED: $dest is not at $rev" >&2; return 1; }
  echo "[fetch] $dest at $rev"
}

# wheel_pin URL SHA256 DEST [EXCLUDE_GLOB] : download a wheel, verify its sha256, unzip it into DEST (a wheel is a zip; no pip, no python)
wheel_pin() {
  local url=$1 sha=$2 dest=$3 exclude=${4:-} f
  if [ -f "$dest/.wheel.sha256" ] && [ "$(cat "$dest/.wheel.sha256")" = "$sha" ]; then echo "[fetch] $dest already has wheel $sha"; return 0; fi
  mkdir -p "$dest" && f=$(mktemp -d)/$(basename "$url")
  curl -fsSL -o "$f" "$url"
  [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$sha" ] || { echo "[fetch] FAILED: sha256 of $url" >&2; return 1; }
  if [ -n "$exclude" ]; then unzip -q -o "$f" -x "$exclude" -d "$dest"; else unzip -q -o "$f" -d "$dest"; fi
  echo "$sha" > "$dest/.wheel.sha256"; rm -f "$f"
  echo "[fetch] $dest: $(basename "$url") sha256 $sha"
}
