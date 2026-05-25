# uv cheat sheet

This project is managed entirely with [`uv`](https://docs.astral.sh/uv/). Do not invoke `pip` or run `python` directly — the registered console scripts (`dictextractor-*`) only resolve when launched through `uv run`.

## First time on a new machine

```bash
uv sync                    # creates .venv and installs all dependencies pinned in uv.lock
uv sync --extra paddle     # additionally install PaddleOCR / paddlepaddle (optional)
```

## Specialised VLM venvs

The specialised document VLMs (MinerU 2.5 Pro, PaddleOCR-VL 1.5, GLM-OCR) ship heavy dependencies that conflict with each other, so each runs in its own isolated venv:

```bash
bash examples/helper/install_models_venv.sh
# creates: .venv-mineru-vllm, .venv-paddleocr, .venv-glmocr
```

The Stage 1 driver (`examples/stage-1/run_stage1_extraction.sh`) picks the right venv per `--vlm-model` automatically.

## Running things

```bash
uv run dictextractor-extract --help                # any registered console script
uv run dictextractor-eval-flat --help
uv run python scripts/flatten_stage1_gold.py       # ad-hoc Python scripts go through uv run

# Reproducing paper sweeps
bash examples/stage-1/run_stage1_extraction.sh
bash examples/stage-2/run_stage2_extraction.sh
bash examples/evaluation/run_stage1_eval_flat.sh
bash examples/evaluation/run_stage2_eval_mdf.sh
```

## Managing dependencies

```bash
uv add some-package         # updates pyproject.toml + uv.lock
uv remove some-package
uv lock --upgrade-package some-package
```

Always commit `uv.lock` alongside `pyproject.toml` so collaborators get an identical environment.
