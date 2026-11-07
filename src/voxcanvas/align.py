import argparse
import json
import logging
import time
from collections.abc import Callable

import torch
from audioread.exceptions import DecodeError
from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
from qwen_asr.inference.qwen3_asr import ASRTranscription
from tqdm import tqdm

from voxcanvas.text import strip_nv_tags
from voxcanvas.utils import LOG_FORMAT, check_checkpoints, check_uuids, read_jsonl, read_shard


logger = logging.getLogger("voxcanvas.align")

# Forced language per manifest language; no language identification is run.
LANGUAGES = {"zh": "Chinese", "en": "English"}
# Errors that only spoil one batch or one clip: GPU OOM, missing file, undecodable audio.
TRANSCRIBE_ERRORS = (torch.OutOfMemoryError, OSError, DecodeError)


def retry_clips(run: Callable[..., list], audio: list[str], *columns: list) -> list:
    """`run(audio, *columns)` on a batch; on failure retry clip by clip, None for clips that still fail."""
    try:
        return run(audio, *columns)
    except TRANSCRIBE_ERRORS as e:
        logger.warning(f"Batch of {len(audio)} failed ({e!r}); retrying one by one")
    # Retry outside the except block so the failed batch's tensors are released first.
    results = []
    for clip in zip(audio, *columns, strict=True):
        try:
            results.extend(run(*([value] for value in clip)))
        except TRANSCRIBE_ERRORS as e:
            logger.error(f"Failed on {clip[0]}: {e!r}")
            results.append(None)
    return results


def transcribe_clips(
    model: Qwen3ASRModel, audio: list[str], language: list[str | None], return_time_stamps: bool
) -> list[ASRTranscription | None]:
    """Transcribe a batch of audio paths; on failure retry clip by clip, None for clips that still fail."""
    return retry_clips(
        lambda audio, language: model.transcribe(audio=audio, language=language, return_time_stamps=return_time_stamps),
        audio,
        language,
    )


def align_transcripts(
    aligner: Qwen3ForcedAligner, audio: list[str], texts: list[str], language: list[str]
) -> list[ASRTranscription | None]:
    """Force-align given transcripts, wrapped like ASR results; on failure retry clip by clip, None if still failing."""
    results = retry_clips(
        lambda audio, texts, language: aligner.align(audio=audio, text=texts, language=language),
        audio,
        texts,
        language,
    )
    return [
        None if result is None else ASRTranscription(language=clip_language, text=text, time_stamps=result)
        for result, text, clip_language in zip(results, texts, language, strict=True)
    ]


def to_aligned_row(row: dict, result: ASRTranscription) -> dict | None:
    """Add the ASR text and per-token [text, start_s, end_s] to the manifest row; None if either is empty."""
    text = result.text.strip()
    if not text or not result.time_stamps:
        return None
    align_word = [[item.text, item.start_time, item.end_time] for item in result.time_stamps]
    return {**row, "text": text, "align_word": align_word}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m voxcanvas.align",
        description="Re-transcribe clips with Qwen3-ASR (or take their given transcripts) and add word timestamps.",
    )
    parser.add_argument("--input", required=True, help="manifest jsonl with uuid, audio_path, language")
    parser.add_argument("--output", required=True, help="aligned jsonl (appended to; resumes by uuid)")
    parser.add_argument("--asr-model", default="checkpoints/Qwen3-ASR-1.7B", help="local directory or Hugging Face id")
    parser.add_argument(
        "--aligner-model", default="checkpoints/Qwen3-ForcedAligner-0.6B", help="local directory or Hugging Face id"
    )
    parser.add_argument(
        "--use-transcript",
        action="store_true",
        help="skip ASR and force-align the manifest 'transcript' with its nonverbal [tags] removed",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    # force: importing qwen_asr already installs a bare root handler.
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)

    check_uuids(read_jsonl(args.input), args.input)
    todo, num_done = read_shard(args.input, args.output, args.shard_id, args.num_shards, key="uuid")
    logger.info(f"Shard {args.shard_id}/{args.num_shards}: {len(todo)} rows to align, {num_done} already done")
    if not todo:
        return
    unknown = {row["language"] for row in todo} - LANGUAGES.keys()
    if unknown:
        raise ValueError(f"Unsupported languages {sorted(unknown)}; expected one of {sorted(LANGUAGES)}")

    if args.use_transcript:
        texts = {row["uuid"]: strip_nv_tags(row["transcript"], row["language"])[0] for row in todo}
        for row in todo:
            if not texts[row["uuid"]]:
                logger.warning(f"Skipping {row['uuid']}: no speech left in its transcript")
        todo = [row for row in todo if texts[row["uuid"]]]
        check_checkpoints(args.aligner_model)
        model = Qwen3ForcedAligner.from_pretrained(args.aligner_model, dtype=torch.bfloat16, device_map="cuda")
    else:
        check_checkpoints(args.asr_model, args.aligner_model)
        model = Qwen3ASRModel.from_pretrained(
            args.asr_model,
            dtype=torch.bfloat16,
            device_map="cuda",
            forced_aligner=args.aligner_model,
            forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": "cuda"},
            max_inference_batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )

    num_written = 0
    start_time = time.perf_counter()
    with open(args.output, "a", encoding="utf-8") as fout, tqdm(total=len(todo), desc="align") as progress:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            audio = [row["audio_path"] for row in batch]
            language = [LANGUAGES[row["language"]] for row in batch]
            if args.use_transcript:
                results = align_transcripts(model, audio, [texts[row["uuid"]] for row in batch], language)
            else:
                results = transcribe_clips(model, audio, language, return_time_stamps=True)
            for row, result in zip(batch, results, strict=True):
                aligned = None if result is None else to_aligned_row(row, result)
                if aligned is None:
                    logger.warning(f"Skipping {row['uuid']}: no transcript or alignment")
                    continue
                fout.write(json.dumps(aligned, ensure_ascii=False) + "\n")
                fout.flush()
                num_written += 1
            progress.update(len(batch))
    elapsed = time.perf_counter() - start_time
    logger.info(
        f"Wrote {num_written}/{len(todo)} rows to {args.output} in {elapsed:.1f}s ({len(todo) / elapsed:.2f} clips/s)"
    )


if __name__ == "__main__":
    main()
