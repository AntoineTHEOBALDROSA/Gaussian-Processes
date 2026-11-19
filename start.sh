#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PY_VERSION="3.11"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo ">> uv not found, installing it with pip --user"
    python3 -m pip install --user uv
    hash -r
fi
uv python install "$PY_VERSION"
if [[ ! -x .venv/bin/python ]] || \
   ! .venv/bin/python -c "import sys; sys.exit(0 if sys.version_info[:2] == tuple(map(int, '$PY_VERSION'.split('.'))) else 1)" 2>/dev/null; then
    echo ">> creating .venv with Python $PY_VERSION"
    rm -rf .venv
    uv venv --python "$PY_VERSION" .venv
fi
set +u
source .venv/bin/activate
set -u
uv pip install -r requirements.txt
exec python app.py "$@"
