import argparse
import json
import logging
import os
from collections import Counter, defaultdict

from voxcanvas.utils import LOG_FORMAT, read_jsonl


logger = logging.getLogger("voxcanvas.export")


def drop_reason(row: dict, max_duration: float, max_asr_score: float, min_sim: float) -> str | None:
    """First failed filter of a verified row (its status if synthesis failed), or None if the pair is kept."""
    if row["status"] != "ok":
        return row["status"]
    if row["synthetic_duration"] > max_duration:
        return "over_max_duration"
    if row["asr_score"] > max_asr_score:
        return "asr_mismatch"
    if row["sim"] <= min_sim:
        return "low_sim"
    return None


def build_pair(row: dict) -> dict:
    """Release record of a kept row; audio stays as paths."""
    return {
        "id": row["id"],
        "task": row["task"],
        "language": row["language"],
        "edit_type": row["edit_type"],
        "original_audio": row["audio_path"],
        "synthetic_audio": row["synthetic_audio"],
        "original_text": row["original_text"],
        "synthetic_text": row["edited_text"],
        "original_duration": row["original_duration"],
        "synthetic_duration": row["synthetic_duration"],
        "forward_instruction": row["instruction"],
        "reverse_instruction": row["reverse_instruction"],
        "edit_regions": row["edit_regions"],
        "num_edits": row["num_edits"],
        "extra": {
            "subset": row["subset"],
            "edits": row["edits"],
            "meta": row["meta"],
            "align_word": row["align_word"],
            "asr_text": row["asr_text"],
            "asr_score": row["asr_score"],
            "sim": row["sim"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m voxcanvas.export",
        description="Filter verified rows and write the final edit pairs.",
    )
    parser.add_argument("--input", required=True, help="verified jsonl (output of verify.sim)")
    parser.add_argument("--output", required=True, help="pairs jsonl (overwritten)")
    parser.add_argument("--max-duration", type=float, default=30.0, help="keep synthetic_duration <= this (s)")
    parser.add_argument("--max-asr-score", type=float, default=0.0, help="keep asr_score <= this")
    parser.add_argument("--min-sim", type=float, default=0.5, help="keep sim > this")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    stats = defaultdict(Counter)
    seen_ids = set()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for row in read_jsonl(args.input):
            if row["id"] in seen_ids:
                reason = "duplicate"
            else:
                reason = drop_reason(row, args.max_duration, args.max_asr_score, args.min_sim)
            seen_ids.add(row["id"])
            stats[row["task"], row["subset"], row["language"]][reason or "kept"] += 1
            if reason is None:
                fout.write(json.dumps(build_pair(row), ensure_ascii=False) + "\n")

    columns = ["kept", *sorted({reason for counter in stats.values() for reason in counter} - {"kept"})]
    total = sum(stats.values(), Counter())
    lines = [["task", "subset", "language", "total", *columns]]
    for key, counter in [*sorted(stats.items()), (("all", "", ""), total)]:
        lines.append([*key, str(sum(counter.values())), *(str(counter[name]) for name in columns)])
    widths = [max(len(line[i]) for line in lines) for i in range(len(lines[0]))]
    table = "\n".join("  ".join(cell.ljust(width) for cell, width in zip(line, widths, strict=True)) for line in lines)
    logger.info(f"Kept / dropped pairs:\n{table}")


if __name__ == "__main__":
    main()
