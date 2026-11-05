#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: bash scripts/run_sharded.sh <num_gpus> <output.jsonl> python -m voxcanvas.<stage> [args...]" >&2
    exit 1
fi
num_gpus=$1
output_jsonl=$2
shift 2
if ! [[ $num_gpus =~ ^[1-9][0-9]*$ ]]; then
    echo "<num_gpus> must be a positive integer, got '$num_gpus'" >&2
    exit 1
fi

# A shard runs on CUDA_VISIBLE_DEVICES[shard_id], or on GPU shard_id when it is unset (an empty value lists no GPU).
if [ -n "${CUDA_VISIBLE_DEVICES+x}" ]; then
    IFS=, read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
    if [ "${#visible_gpus[@]}" -lt "$num_gpus" ]; then
        echo "CUDA_VISIBLE_DEVICES lists ${#visible_gpus[@]} GPUs, fewer than $num_gpus" >&2
        exit 1
    fi
fi

shard_pids=()
trap 'kill "${shard_pids[@]}" 2>/dev/null || true; exit 130' INT TERM
for ((shard_id = 0; shard_id < num_gpus; shard_id++)); do
    CUDA_VISIBLE_DEVICES=${visible_gpus[shard_id]:-$shard_id} "$@" --output "$output_jsonl.shard$shard_id" \
        --shard-id "$shard_id" --num-shards "$num_gpus" &
    shard_pids+=($!)
done
failed_shards=()
for shard_id in "${!shard_pids[@]}"; do
    wait "${shard_pids[shard_id]}" || failed_shards+=("$shard_id")
done
if [ "${#failed_shards[@]}" -gt 0 ]; then
    echo "Failed shards: ${failed_shards[*]}; nothing merged. Re-run the same command to resume them." >&2
    exit 1
fi
# Merge into a temporary file so a failed merge never truncates a previous $output_jsonl.
for ((shard_id = 0; shard_id < num_gpus; shard_id++)); do
    cat "$output_jsonl.shard$shard_id"
done > "$output_jsonl.tmp"
mv "$output_jsonl.tmp" "$output_jsonl"
echo "Merged $num_gpus shards into $output_jsonl"
