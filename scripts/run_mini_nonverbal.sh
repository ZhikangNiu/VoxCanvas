#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-outputs/mini_nonverbal}
MANIFEST_JSONL=${MANIFEST_JSONL:-examples/mini_nonverbal/manifest.jsonl}
PYTHON=${PYTHON:-python}

mkdir -p "$OUTPUT_DIR"

if [ "${SKIP_ALIGN:-0}" = 1 ]; then
    aligned_jsonl=${ALIGNED_JSONL:-examples/mini_nonverbal/aligned.jsonl}
else
    aligned_jsonl=$OUTPUT_DIR/aligned.jsonl
    "$PYTHON" -m voxcanvas.align --input "$MANIFEST_JSONL" --output "$aligned_jsonl" --use-transcript \
        ${ALIGNER_MODEL:+--aligner-model "$ALIGNER_MODEL"}
fi

"$PYTHON" -m voxcanvas.annotate --task nonverbal --subset insertion --input "$aligned_jsonl" \
    --output "$OUTPUT_DIR/edits.jsonl"
"$PYTHON" -m voxcanvas.synthesize --input "$OUTPUT_DIR/edits.jsonl" --output "$OUTPUT_DIR/synth.jsonl" \
    --audio-dir "$OUTPUT_DIR/audio" ${F5_CKPT:+--ckpt "$F5_CKPT"} ${VOCODER:+--vocoder "$VOCODER"}
"$PYTHON" -m voxcanvas.verify.asr --input "$OUTPUT_DIR/synth.jsonl" --output "$OUTPUT_DIR/asr.jsonl" \
    ${ASR_MODEL:+--model "$ASR_MODEL"}
"$PYTHON" -m voxcanvas.verify.sim --input "$OUTPUT_DIR/asr.jsonl" --output "$OUTPUT_DIR/sim.jsonl" \
    ${SIM_CKPT:+--ckpt "$SIM_CKPT"}
"$PYTHON" -m voxcanvas.export --input "$OUTPUT_DIR/sim.jsonl" --output "$OUTPUT_DIR/pairs.jsonl" \
    ${MAX_ASR_SCORE:+--max-asr-score "$MAX_ASR_SCORE"}
echo "Final pairs: $OUTPUT_DIR/pairs.jsonl"
