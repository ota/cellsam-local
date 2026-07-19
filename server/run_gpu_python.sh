#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON="$ROOT/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
  echo "Missing .venv. Run: uv venv .venv && uv pip install --python .venv/bin/python -r server/requirements.txt" >&2
  exit 1
fi

SITE_PACKAGES=$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')
NVIDIA_LIBS=""

for dir in "$SITE_PACKAGES"/nvidia/*/lib; do
  [ -d "$dir" ] || continue
  if [ -z "$NVIDIA_LIBS" ]; then
    NVIDIA_LIBS="$dir"
  else
    NVIDIA_LIBS="$NVIDIA_LIBS:$dir"
  fi
done

if [ -n "$NVIDIA_LIBS" ]; then
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"
  else
    export LD_LIBRARY_PATH="$NVIDIA_LIBS"
  fi
fi

exec "$PYTHON" "$@"
