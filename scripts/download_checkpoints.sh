#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

checkpoint_source=${1:-hf}
case $checkpoint_source in
    hf)
        hf download SWivid/F5-TTS F5TTS_v1_Base/model_1250000.safetensors --local-dir checkpoints
        hf download charactr/vocos-mel-24khz config.yaml pytorch_model.bin --local-dir checkpoints/vocos-mel-24khz
        hf download Qwen/Qwen3-ASR-1.7B --local-dir checkpoints/Qwen3-ASR-1.7B
        hf download Qwen/Qwen3-ForcedAligner-0.6B --local-dir checkpoints/Qwen3-ForcedAligner-0.6B
        ;;
    modelscope)
        modelscope_download=(uvx --from modelscope-hub==0.6.1 modelscope download)
        "${modelscope_download[@]}" SWivid/F5-TTS_Emilia-ZH-EN model_1250000.safetensors \
            --local-dir checkpoints/F5TTS_v1_Base
        # charactr/vocos-mel-24khz is not on ModelScope; keep the Hugging Face token away from the mirror.
        HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_IMPLICIT_TOKEN=1 hf download charactr/vocos-mel-24khz \
            config.yaml pytorch_model.bin --local-dir checkpoints/vocos-mel-24khz
        "${modelscope_download[@]}" Qwen/Qwen3-ASR-1.7B --local-dir checkpoints/Qwen3-ASR-1.7B
        "${modelscope_download[@]}" Qwen/Qwen3-ForcedAligner-0.6B --local-dir checkpoints/Qwen3-ForcedAligner-0.6B
        ;;
    *)
        echo "Usage: bash scripts/download_checkpoints.sh [hf|modelscope]" >&2
        exit 1
        ;;
esac
if [ ! -f checkpoints/wavlm_large_finetune.pth ]; then
    echo "Now download wavlm_large_finetune.pth from" \
        "https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view to checkpoints/wavlm_large_finetune.pth"
fi
