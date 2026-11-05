#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-outputs/mini}
MANIFEST_JSONL=${MANIFEST_JSONL:-examples/mini/manifest.jsonl}
PYTHON=${PYTHON:-python}

# Optional overrides; an unset variable keeps the stage CLI default.
llm_args=(
    ${LLM_BASE_URL:+--base-url "$LLM_BASE_URL"} ${LLM_MODEL:+--model "$LLM_MODEL"}
    ${LLM_API_KEY_ENV:+--api-key-env "$LLM_API_KEY_ENV"} ${LLM_EXTRA_BODY:+--extra-body "$LLM_EXTRA_BODY"}
)

mkdir -p "$OUTPUT_DIR/edits"

if [ "${SKIP_ALIGN:-0}" = 1 ]; then
    aligned_jsonl=${ALIGNED_JSONL:-examples/mini/aligned.jsonl}
else
    aligned_jsonl=$OUTPUT_DIR/aligned.jsonl
    "$PYTHON" -m voxcanvas.align --input "$MANIFEST_JSONL" --output "$aligned_jsonl" \
        ${ASR_MODEL:+--asr-model "$ASR_MODEL"} ${ALIGNER_MODEL:+--aligner-model "$ALIGNER_MODEL"}
fi

task_subsets=(
    single_span:deletion single_span:insertion single_span:substitution
    multi_span:multi
    location_reference:d1 location_reference:d4 location_reference:m1 location_reference:m2
    location_reference:i1 location_reference:i2
    location_reference:l_d location_reference:l_m location_reference:l_i
    location_reference:l_t_d location_reference:l_t_m location_reference:l_t_i
)
subset_edits_files=()
for task_subset in "${task_subsets[@]}"; do
    task=${task_subset%%:*}
    subset=${task_subset#*:}
    edits_jsonl=$OUTPUT_DIR/edits/${task}_${subset}.jsonl
    subset_edits_files+=("$edits_jsonl")
    "$PYTHON" -m voxcanvas.annotate --task "$task" --subset "$subset" --input "$aligned_jsonl" --output "$edits_jsonl" \
        "${llm_args[@]}"
done
cat "${subset_edits_files[@]}" > "$OUTPUT_DIR/edits.jsonl"

"$PYTHON" -m voxcanvas.synthesize --input "$OUTPUT_DIR/edits.jsonl" --output "$OUTPUT_DIR/synth.jsonl" \
    --audio-dir "$OUTPUT_DIR/audio" ${F5_CKPT:+--ckpt "$F5_CKPT"} ${VOCODER:+--vocoder "$VOCODER"}
"$PYTHON" -m voxcanvas.verify.asr --input "$OUTPUT_DIR/synth.jsonl" --output "$OUTPUT_DIR/asr.jsonl" \
    ${ASR_MODEL:+--model "$ASR_MODEL"}
"$PYTHON" -m voxcanvas.verify.sim --input "$OUTPUT_DIR/asr.jsonl" --output "$OUTPUT_DIR/sim.jsonl" \
    ${SIM_CKPT:+--ckpt "$SIM_CKPT"}
"$PYTHON" -m voxcanvas.export --input "$OUTPUT_DIR/sim.jsonl" --output "$OUTPUT_DIR/pairs.jsonl"
echo "Final pairs: $OUTPUT_DIR/pairs.jsonl"
