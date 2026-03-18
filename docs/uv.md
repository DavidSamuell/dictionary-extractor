# First time on a new machine
uv sync                          # creates .venv + installs everything
# Add a new dependency
uv add some-package              # updates pyproject.toml + uv.lock
# Run anything
uv run python -m dictextractor.cli.extract ...
uv run dictextractor-extract ...   # short form via registered entry point
bash scripts/run_2_stage_extraction.sh  # script now uses uv run internally
# With PaddleOCR
uv sync --extra paddle