import argparse
import json
import logging
import re
import unicodedata

import jiwer
import torch
from pypinyin import Style, lazy_pinyin
from qwen_asr import Qwen3ASRModel
from tqdm import tqdm

from voxcanvas.align import transcribe_clips
from voxcanvas.utils import LOG_FORMAT, check_checkpoints, read_shard


logger = logging.getLogger("voxcanvas.verify.asr")


def to_jiwer_input(text: str, lang: str) -> str:
    """Space-separated scoring units: lowercase English words, or tone-numbered pinyin syllables for Chinese.

    English keeps word characters and apostrophes; Chinese drops whitespace, punctuation (P*) and symbols (S*).
    """
    if lang == "zh":
        kept = "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith(("P", "S")))
        return " ".join(lazy_pinyin(kept, style=Style.TONE3, neutral_tone_with_five=True))
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def compute_asr_score(ref: str, hyp: str, lang: str) -> float:
    """WER (en) or tone-sensitive pinyin error rate (zh) of the ASR hypothesis against the expected text."""
    return float(jiwer.wer(to_jiwer_input(ref, lang), to_jiwer_input(hyp, lang)))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m voxcanvas.verify.asr",
        description="Transcribe synthesized audio with Qwen3-ASR and score it.",
    )
    parser.add_argument("--input", required=True, help="synthesized jsonl")
    parser.add_argument("--output", required=True, help="output jsonl (appended to; resumes by id)")
    parser.add_argument("--model", default="checkpoints/Qwen3-ASR-1.7B", help="local directory or Hugging Face id")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    # force: importing qwen_asr already installs a bare root handler.
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)

    rows, num_done = read_shard(args.input, args.output, args.shard_id, args.num_shards)
    logger.info(f"Shard {args.shard_id}/{args.num_shards}: {len(rows)} rows to verify, {num_done} already done")
    if not rows:
        return

    check_checkpoints(args.model)
    model = Qwen3ASRModel.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", max_inference_batch_size=args.batch_size
    )
    num_failed = 0
    with open(args.output, "a", encoding="utf-8") as fout:
        for start in tqdm(range(0, len(rows), args.batch_size), desc="asr"):
            batch = rows[start : start + args.batch_size]
            audio = [row["synthetic_audio"] for row in batch if row["status"] == "ok"]
            # No language hint: the model detects the language itself.
            results = iter(
                transcribe_clips(model, audio, [None] * len(audio), return_time_stamps=False) if audio else []
            )
            for row in batch:
                if row["status"] == "ok":
                    result = next(results)
                    if result is None:  # no row, so the id stays undone and a re-run retries it
                        num_failed += 1
                        continue
                    row["asr_text"] = result.text
                    row["asr_score"] = compute_asr_score(row["edited_text"], result.text, row["language"])
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
    if num_failed:
        raise RuntimeError(
            f"{num_failed} synthetic wavs could not be transcribed; fix them and re-run to retry only those rows"
        )


if __name__ == "__main__":
    main()
