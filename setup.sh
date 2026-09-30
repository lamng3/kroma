#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

uv venv --python 3.12
source .venv/bin/activate
uv pip install ".[dev]"

echo "KROMA is ready."
echo "Activate it with: source .venv/bin/activate"
echo "Run it with: kroma --help"
echo "Model backends: uv pip install '.[models]'"
