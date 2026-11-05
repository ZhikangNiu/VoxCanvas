# VoxCanvas data pipeline

Content-editing data pipeline of VoxCanvas-41M: align -> annotate -> synthesize -> verify -> export
(see README.md; algorithms, formats and differences from the released data in docs/pipeline.md).

## Layout
- `src/voxcanvas/`: one CLI per stage (`align.py`, `annotate/`, `synthesize.py`, `verify/asr.py`, `verify/sim.py`,
  `export.py`); shared text logic in `text.py`; annotate generators in `annotate/{content,occurrence,boundary,timestamp}.py`.
- `scripts/`: `download_checkpoints.sh [hf|modelscope]` (checkpoints into `checkpoints/`, the stage defaults; WavLM is a manual
  download), `run_mini_pipeline.sh` (all stages on `examples/mini`), `run_sharded.sh` (multi-GPU launcher).
- `checkpoints/`: downloaded model checkpoints, git-ignored except `.gitkeep`; tests must not depend on it.
- `examples/mini/`: 20 Emilia clips, manifest, precomputed `aligned.jsonl`.
- `tests/`: pytest; `third_party/F5-TTS`: upstream submodule (do not modify).
- `agent_space/`: scratch space for experiments and one-off scripts (git-ignored); never import it from `src/` or `tests/`.

## Environment
- uv project env in `.venv` (Python 3.10, torch 2.7). Run from the repo root with `.venv/bin/python`.
- Add dependencies in `pyproject.toml`, then `uv lock` and `uv sync`. Keep `uv.lock` on the default PyPI index.
- `annotate` reads the LLM key from `.env` (`DEEPSEEK_API_KEY` by default); never print it.

## Rules
- Code style is mandatory: `.claude/rules/code_style.md` and `.claude/rules/examples.md`.
- Licensing: code is MIT (`LICENSE`); data keeps the license of its source corpus (Emilia: CC BY-NC 4.0), never MIT.
  Any data file added to the repo must ship with its source license and attribution (see `examples/mini/LICENSE`).
- Record format changes and any behaviour that differs from the released data in `docs/pipeline.md`.

## Checks
```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/python -m pytest -q      # the SIM CLI test is skipped without a CUDA GPU
```
