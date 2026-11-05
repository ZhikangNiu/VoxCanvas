import argparse
import json
import logging
import os

import torch
import torch.nn.functional as F
import torchaudio
from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL
from tqdm import tqdm

from voxcanvas.utils import LOG_FORMAT, read_shard


logger = logging.getLogger("voxcanvas.verify.sim")


def load_wav_16k(path: str) -> torch.Tensor:
    """Load audio as a mono 16 kHz waveform of shape [1, num_samples]."""
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)
    return torchaudio.functional.resample(wav, sr, 16000)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m voxcanvas.verify.sim",
        description="Speaker similarity between original and synthesized audio.",
    )
    parser.add_argument("--input", required=True, help="jsonl with audio_path and synthetic_audio")
    parser.add_argument("--output", required=True, help="output jsonl (appended to; resumes by id)")
    parser.add_argument(
        "--ckpt",
        default="checkpoints/wavlm_large_finetune.pth",
        help="WavLM-Large ECAPA-TDNN speaker verification checkpoint",
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    rows, num_done = read_shard(args.input, args.output, args.shard_id, args.num_shards)
    logger.info(f"Shard {args.shard_id}/{args.num_shards}: {len(rows)} rows to score, {num_done} already done")
    if not rows:
        return

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(
            f"{args.ckpt} not found; download it (wavlm_large_finetune.pth) from "
            "https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view"
        )
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(args.ckpt, weights_only=True, map_location="cpu")
    # strict=False: the checkpoint also holds the training loss head (loss_calculator.projection.weight).
    missing_keys, _ = model.load_state_dict(state_dict["model"], strict=False)
    if missing_keys:
        raise ValueError(f"{args.ckpt} is not a WavLM ECAPA-TDNN checkpoint; missing {missing_keys[:5]}")
    model = model.to("cuda").eval()

    num_failed = 0
    with open(args.output, "a", encoding="utf-8") as fout:
        for row in tqdm(rows, desc="sim"):
            if row["status"] == "ok":
                try:
                    wav_synthetic, wav_original = load_wav_16k(row["synthetic_audio"]), load_wav_16k(row["audio_path"])
                except RuntimeError as e:  # torchaudio's error for a missing or undecodable file
                    logger.error(f"Cannot load the audio of {row['id']}: {e!r}")
                    num_failed += 1  # no row, so the id stays undone and a re-run retries it
                    continue
                with torch.inference_mode():
                    emb_synthetic = model(wav_synthetic.to("cuda"))
                    emb_original = model(wav_original.to("cuda"))
                row["sim"] = F.cosine_similarity(emb_synthetic, emb_original)[0].item()
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
    if num_failed:
        raise RuntimeError(f"{num_failed} rows have unreadable audio; fix them and re-run to retry only those rows")


if __name__ == "__main__":
    main()
