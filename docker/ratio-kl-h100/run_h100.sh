#!/usr/bin/env bash
set -euo pipefail

BASE_REF=${1:-063a261516febcde700c1b751b60b73b11abb427}
HEAD_REF=${2:-7cda80c394381805b8967bae596ded4d1343b8d4}
REPO=$(git rev-parse --show-toplevel)
OUT="$REPO/.cache/benchmarks/ratio_kl/h100-$(date -u +%Y%m%dT%H%M%SZ)"
TMP=$(mktemp -d)
BASE_TREE="$TMP/base"
HEAD_TREE="$TMP/head"

cleanup() {
  git -C "$REPO" worktree remove --force "$BASE_TREE" >/dev/null 2>&1 || true
  git -C "$REPO" worktree remove --force "$HEAD_TREE" >/dev/null 2>&1 || true
  rm -rf "$TMP"
}
trap cleanup EXIT

python -c 'import torch; name=torch.cuda.get_device_name(); assert "H100" in name, f"H100 required, found {name}"'
git -C "$REPO" worktree add --detach "$BASE_TREE" "$BASE_REF"
git -C "$REPO" worktree add --detach "$HEAD_TREE" "$HEAD_REF"
cp "$HEAD_TREE/benchmarks/benchmark_ratio_kl.py" "$TMP/benchmark_ratio_kl.py"
mkdir -p "$OUT/base" "$OUT/head"

PYTHONPATH="$HEAD_TREE" python -m pytest "$HEAD_TREE/tests/test_ratio_kl.py" -q | tee "$OUT/tests.txt"

run_case() {
  local side=$1 tree=$2 dtype=$3 label=$4 prompts=$5 groups=$6 tokens=$7 vocab=$8
  (
    cd "$tree"
    PYTHONPATH="$tree" python "$TMP/benchmark_ratio_kl.py" \
      --backward-suite --dtype "$dtype" --num-prompts "$prompts" --g-sizes "$groups" \
      --completion-lens "$tokens" --vocab-sizes "$vocab" --mask-densities 0.1,0.9 \
      --seed 0 --warmup 20 --repeat 100 --output "$OUT/$side/$dtype-$label.json"
  )
}

for dtype in float16 bfloat16; do
  for side in base head; do
    tree=$BASE_TREE
    [[ $side == head ]] && tree=$HEAD_TREE
    run_case "$side" "$tree" "$dtype" n256-v8192 1 1 256 8192
    run_case "$side" "$tree" "$dtype" n512-v8192 1 1 512 8192
    run_case "$side" "$tree" "$dtype" n256-v16384 1 1 256 16384
    run_case "$side" "$tree" "$dtype" n128-v50257 1 1 128 50257
    run_case "$side" "$tree" "$dtype" b32-t256-v32768 4 8 256 32768
  done
done

python "$REPO/docker/ratio-kl-h100/compare.py" \
  "$OUT/base" "$OUT/head" --output "$OUT/CONCLUSION.md"
echo "$OUT"
