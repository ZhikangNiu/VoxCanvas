import argparse
import json
import logging
import os
import time
from collections import Counter
from difflib import SequenceMatcher
from importlib.resources import files

import soundfile as sf
import torch
import torchaudio
from cached_path import cached_path
from f5_tts.infer.utils_infer import hop_length, load_model, load_vocoder, target_rms, target_sample_rate
from f5_tts.model import CFM, DiT
from f5_tts.model.utils import convert_char_to_pinyin
from omegaconf import OmegaConf
from tqdm import tqdm
from vocos import Vocos

from voxcanvas.text import normalize_align_tokens, tokenize_aligned
from voxcanvas.utils import LOG_FORMAT, check_checkpoints, read_shard, rng_for


logger = logging.getLogger("voxcanvas.synthesize")

HF_VOCODER = "charactr/vocos-mel-24khz"
MIN_TARGET_SEC = 0.18  # floor for the duration of replaced or inserted speech
DELETION_SEC = 0.1  # generated filler where speech is deleted
MASK_WIDEN_SEC = 0.5  # regenerated context on each side of an edited part
MAX_FRAMES = 8192  # DiT text position table (~87 s); CFM.sample needs one frame more than the condition
FRAMES_PER_SEC = target_sample_rate / hop_length  # 93.75 mel frames per second
# (lower clamp, upper clamp, fallback) of the seconds per inserted token.
SECONDS_PER_TOKEN = {"zh": (0.08, 0.32, 0.2), "en": (0.1, 0.45, 0.24)}


def build_edit_plan(
    align_word: list[list], edited_text: str, lang: str
) -> tuple[list[list[float]], list[list[float]], list[float]]:
    """Diff the aligned words against `edited_text` into the parts of speech to regenerate.

    Returns (source_parts, parts_to_edit, fix_duration): [start, end] seconds on the original timeline, the same parts
    on the edited timeline, and their generated durations. All empty when no spoken token changed.
    """
    num_words = len(align_word)
    audio_start, audio_end = align_word[0][1], align_word[-1][2]
    lower, upper, fallback = SECONDS_PER_TOKEN[lang]
    num_timed = sum(end - start > 0 for _, start, end in align_word)
    seconds_per_token = min(max((audio_end - audio_start) / num_timed, lower), upper) if num_timed else fallback

    edited_tokens = [token for token, _, _ in tokenize_aligned(edited_text, lang)]
    matcher = SequenceMatcher(a=normalize_align_tokens(align_word), b=edited_tokens, autojunk=False)
    segments = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert":
            # Inserted speech takes the place of the pause between the neighbouring words.
            if i1 == 0:
                start = end = audio_start
            elif i1 == num_words:
                start = end = audio_end
            else:
                start = align_word[i1 - 1][2]
                end = max(align_word[i1][1], start)
            target = max(MIN_TARGET_SEC, (j2 - j1) * seconds_per_token)
        else:
            start, end = align_word[i1][1], align_word[i2 - 1][2]
            replaced = max(end - start, 0.0) * (j2 - j1) / (i2 - i1)
            target = DELETION_SEC if tag == "delete" else max(MIN_TARGET_SEC, replaced)
        segments.append([start, end, target])

    merged = []
    for start, end, target in sorted(segments, key=lambda segment: segment[:2]):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] += target
        else:
            merged.append([start, end, target])

    source_parts, parts_to_edit, fix_duration, shift = [], [], [], 0.0
    for start, end, target in merged:
        source_parts.append([start, end])
        parts_to_edit.append([start + shift, start + shift + target])
        fix_duration.append(target)
        shift += target - max(end - start, 0.0)
    return source_parts, parts_to_edit, fix_duration


def build_adjusted_mel(
    mel: torch.Tensor, source_parts: list[list[float]], fix_duration: list[float], audio_duration: float
) -> torch.Tensor:
    """Replace each source part of `mel` ([1, frames, mels]) by zero frames of its fix duration."""
    num_frames, offset, pieces = mel.shape[1], 0, []
    for (start, end), target in zip(source_parts, fix_duration, strict=True):
        start = min(max(start, 0.0), audio_duration)
        end = min(max(end, start), audio_duration)
        start_frame = max(min(round(start * FRAMES_PER_SEC), num_frames), offset)
        end_frame = max(min(round(end * FRAMES_PER_SEC), num_frames), start_frame)
        num_target_frames = max(round(target * FRAMES_PER_SEC), 1)
        pieces += [mel[:, offset:start_frame], mel.new_zeros(1, num_target_frames, mel.shape[2])]
        offset = end_frame
    return torch.cat([*pieces, mel[:, offset:]], dim=1)


def build_edit_mask(num_frames: int, parts_to_edit: list[list[float]]) -> torch.Tensor:
    """[1, num_frames] bool mask, False on the frames to generate (each part widened by MASK_WIDEN_SEC per side)."""
    mask = torch.ones(1, num_frames, dtype=torch.bool)
    for start, end in parts_to_edit:
        start_frame = round((start - MASK_WIDEN_SEC) * FRAMES_PER_SEC)
        end_frame = round((end + MASK_WIDEN_SEC) * FRAMES_PER_SEC)
        start_frame = min(max(start_frame, 0), num_frames)
        mask[:, start_frame : min(max(end_frame, start_frame + 1), num_frames)] = False
    return mask


def synthesize_record(
    record: dict,
    model: CFM,
    vocoder: Vocos,
    audio_dir: str,
    *,
    seed: int,
    nfe_step: int,
    cfg_strength: float,
    sway_sampling_coef: float,
) -> dict | None:
    """Return `record` plus status, synthetic_audio, durations and edit_regions; writes the wav when status is ok.

    Status: no_change (no spoken token changed), too_long (more than MAX_FRAMES), no_context (the widened edit mask
    covers every frame), ok. None if the source file does not exist. The wav goes to `<audio_dir>/<id>.wav`.
    """
    row = {
        **record,
        "status": "ok",
        "synthetic_audio": "",
        "original_duration": 0.0,
        "synthetic_duration": 0.0,
        "edit_regions": [],
    }
    # align decoded every clip, so a missing file is an environment error: leave the record undone for a re-run.
    try:
        os.stat(record["audio_path"])
    except FileNotFoundError:
        return None
    audio, sample_rate = torchaudio.load(record["audio_path"])
    audio = audio.mean(dim=0, keepdim=True)
    rms = audio.square().mean().sqrt().item()
    if 0 < rms < target_rms:
        audio = audio * (target_rms / rms)
    audio = torchaudio.functional.resample(audio, sample_rate, target_sample_rate)
    audio_duration = audio.shape[1] / target_sample_rate
    row["original_duration"] = round(audio_duration, 4)

    source_parts, parts_to_edit, fix_duration = build_edit_plan(
        record["align_word"], record["edited_text"], record["language"]
    )
    if not source_parts:
        return {**row, "status": "no_change"}
    with torch.inference_mode():
        mel = model.mel_spec(audio.to(model.device)).permute(0, 2, 1)
    cond = build_adjusted_mel(mel, source_parts, fix_duration, audio_duration)
    if cond.shape[1] + 1 > MAX_FRAMES:
        return {**row, "status": "too_long"}
    edit_mask = build_edit_mask(cond.shape[1], parts_to_edit)
    # With no original frame left as the prompt, sampling would be unconditional TTS in another voice.
    if not edit_mask.any():
        return {**row, "status": "no_context"}

    with torch.inference_mode():
        generated, _ = model.sample(
            cond=cond,
            text=convert_char_to_pinyin([record["edited_text"]]),
            duration=cond.shape[1],
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            seed=rng_for(seed, record["id"]).getrandbits(32),
            edit_mask=edit_mask.to(cond.device),
        )
        wav = vocoder.decode(generated.float().permute(0, 2, 1))[0].cpu()
    if 0 < rms < target_rms:
        wav = wav * (rms / target_rms)
    # Vocos can overshoot full scale; scale the peak down to 1 instead of letting PCM_16 hard-clip it.
    wav = wav / max(1.0, wav.abs().max().item())
    path = os.path.join(audio_dir, record["id"] + ".wav")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, wav.numpy(), target_sample_rate, subtype="PCM_16")
    edit_regions = [
        {
            "original_start": round(start, 4),
            "original_end": round(end, 4),
            "synthetic_start": round(new_start, 4),
            "synthetic_end": round(new_end, 4),
        }
        for (start, end), (new_start, new_end) in zip(source_parts, parts_to_edit, strict=True)
    ]
    return {
        **row,
        "synthetic_audio": path,
        "synthetic_duration": round(wav.shape[0] / target_sample_rate, 4),
        "edit_regions": edit_regions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m voxcanvas.synthesize",
        description="Resynthesize edited speech by F5-TTS masked infilling.",
    )
    parser.add_argument("--input", required=True, help="edit records jsonl")
    parser.add_argument("--output", required=True, help="synthesized rows jsonl (appended to; resumes by id)")
    parser.add_argument("--audio-dir", required=True, help="root of the edited 24 kHz PCM_16 wavs (<id>.wav)")
    parser.add_argument(
        "--ckpt",
        default="checkpoints/F5TTS_v1_Base/model_1250000.safetensors",
        help="F5TTS_v1_Base checkpoint (local file or hf:// URL)",
    )
    parser.add_argument(
        "--vocoder", default="checkpoints/vocos-mel-24khz", help=f"local Vocos directory or {HF_VOCODER}"
    )
    parser.add_argument("--vocab", default="", help="vocab.txt (default: the one bundled with f5_tts)")
    parser.add_argument("--nfe-step", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=0, help="base seed; each record's seed derives from it and its id")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    records, num_done = read_shard(args.input, args.output, args.shard_id, args.num_shards)
    logger.info(
        f"Shard {args.shard_id}/{args.num_shards}: {len(records)} records to synthesize, {num_done} already done"
    )
    if not records:
        return

    check_checkpoints(args.ckpt, args.vocoder)
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
    model = load_model(DiT, model_cfg.model.arch, str(cached_path(args.ckpt)), vocab_file=args.vocab, device="cuda")
    vocoder = load_vocoder("vocos", is_local=args.vocoder != HF_VOCODER, local_path=args.vocoder, device="cuda")

    statuses = Counter()
    num_missing = 0
    start_time = time.perf_counter()
    with open(args.output, "a", encoding="utf-8") as fout:
        for record in tqdm(records, desc="synthesize"):
            row = synthesize_record(
                record,
                model,
                vocoder,
                args.audio_dir,
                seed=args.seed,
                nfe_step=args.nfe_step,
                cfg_strength=args.cfg_strength,
                sway_sampling_coef=args.sway_sampling_coef,
            )
            if row is None:
                logger.warning(f"Skipping {record['id']}: {record['audio_path']} does not exist")
                num_missing += 1
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            statuses[row["status"]] += 1
    elapsed = time.perf_counter() - start_time
    logger.info(
        f"Wrote {len(records) - num_missing}/{len(records)} rows in {elapsed:.1f}s "
        f"({len(records) / elapsed:.2f} rows/s): {dict(statuses)}; {num_missing} skipped (missing audio)"
    )
    if num_missing:
        raise FileNotFoundError(f"{num_missing} source audio files are missing; fix the audio paths and re-run")


if __name__ == "__main__":
    main()
