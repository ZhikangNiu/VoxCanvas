import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

from voxcanvas.annotate import boundary, content, nonverbal, occurrence, timestamp
from voxcanvas.annotate.llm import LLMClient
from voxcanvas.annotate.records import SUBSETS_OF_TASK, record_id, stratified_sample
from voxcanvas.text import gate
from voxcanvas.utils import LOG_FORMAT, check_uuids, load_done_ids, read_jsonl, rng_for


logger = logging.getLogger("voxcanvas.annotate")

GENERATORS = (content, occurrence, boundary, timestamp)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m voxcanvas.annotate",
        description="Rule-based edit selection, LLM new-word generation and instruction templates.",
    )
    parser.add_argument("--task", required=True, choices=list(SUBSETS_OF_TASK))
    parser.add_argument("--subset", required=True, help="subset of the task, e.g. deletion, multi, d4, l_t_m")
    parser.add_argument(
        "--input",
        required=True,
        help="aligned rows (jsonl, output of voxcanvas.align; with --use-transcript for the nonverbal task)",
    )
    parser.add_argument("--output", required=True, help="edit records (jsonl); appended, finished ids are skipped")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="candidates to annotate, length-stratified; 0 = all. Resume into --output only with the same input rows",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed of the selection, sampling and completion draws; finished ids are skipped whatever the seed, "
        "so a new seed takes effect only in a fresh --output",
    )
    parser.add_argument("--concurrency", type=int, default=16, help="parallel LLM requests")
    parser.add_argument("--base-url", default="https://api.deepseek.com", help="OpenAI-compatible endpoint")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY", help="environment variable holding the API key")
    parser.add_argument(
        "--extra-body",
        type=json.loads,
        default=None,
        help=(
            "JSON extra_body for the request, e.g. "
            '\'{"top_k": 20, "chat_template_kwargs": {"enable_thinking": false}}\' for vLLM'
        ),
    )
    args = parser.parse_args()
    if args.subset not in SUBSETS_OF_TASK[args.task]:
        parser.error(f"--subset {args.subset} does not belong to --task {args.task}: {SUBSETS_OF_TASK[args.task]}")
    # WARNING root level keeps the per-request httpx INFO logs quiet.
    logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)
    logger.setLevel(logging.INFO)
    load_dotenv(find_dotenv(usecwd=True))

    # nonverbal shares the subset name "insertion" with single_span, so it is picked by task.
    if args.task == "nonverbal":
        generator = nonverbal
    else:
        generator = next(module for module in GENERATORS if args.subset in module.SUBSETS)
    llm = None
    if args.subset in generator.NEEDS_LLM:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(f"Set {args.api_key_env} in the environment or in .env to annotate {args.subset}")
        llm = LLMClient(args.base_url, args.model, api_key, extra_body=args.extra_body)

    rows = list(read_jsonl(args.input))
    if args.task == "nonverbal":
        rows = [split for row in rows for split in nonverbal.split_by_tag(row)]
    check_uuids(rows, args.input)
    num_gated = 0
    candidates = []
    for row in rows:
        if not gate(row):
            continue
        num_gated += 1
        candidate = generator.make_candidate(
            row, args.subset, rng_for(args.seed, row["uuid"], args.subset, "candidate")
        )
        if candidate is not None:
            candidates.append(candidate)
    selected = stratified_sample(candidates, args.num_samples, args.seed, args.subset)
    done = load_done_ids(args.output)
    todo = [
        candidate for candidate in selected if record_id(args.task, args.subset, candidate["row"]["uuid"]) not in done
    ]
    logger.info(
        f"{args.task}/{args.subset}: {len(rows)} rows, {num_gated} pass the gate, {len(candidates)} candidates, "
        f"{len(selected)} selected, {len(selected) - len(todo)} already done"
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    num_kept = 0
    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    try:
        with open(args.output, "a", encoding="utf-8") as f:
            futures = [
                executor.submit(
                    generator.complete_candidate,
                    candidate,
                    llm,
                    rng_for(args.seed, candidate["row"]["uuid"], args.subset, "complete"),
                )
                for candidate in todo
            ]
            for future in tqdm(futures, desc=args.subset):
                record = future.result()
                if record is None:
                    continue
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                num_kept += 1
    finally:
        # Drop queued candidates on Ctrl-C or a worker error instead of sending all their LLM requests.
        executor.shutdown(cancel_futures=True)
    logger.info(f"{args.task}/{args.subset}: kept {num_kept}, dropped {len(todo) - num_kept} of {len(todo)} annotated")
    if llm is not None and llm.num_requests and not llm.num_replies:
        raise RuntimeError(
            f"{args.task}/{args.subset}: every LLM request failed; check --base-url, --model, --extra-body "
            "and the warnings above"
        )


if __name__ == "__main__":
    main()
