#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON="${CELLSAM_PYTHON:-$ROOT/.venv/bin/python}"

case "$PYTHON" in
  /*) ;;
  *) PYTHON="$ROOT/$PYTHON" ;;
esac

if [ ! -x "$PYTHON" ]; then
  echo "Missing Python interpreter: $PYTHON" >&2
  echo "Run: uv venv .venv && uv pip install --python .venv/bin/python -r server/requirements.txt" >&2
  exit 1
fi

GPU_LIB_MODE="${CELLSAM_GPU_LIB_MODE:-onnxruntime}"

case "$GPU_LIB_MODE" in
  onnxruntime)
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
    ;;
  torch|none)
    ;;
  *)
    echo "Unknown CELLSAM_GPU_LIB_MODE: $GPU_LIB_MODE" >&2
    echo "Use one of: onnxruntime, torch, none" >&2
    exit 1
    ;;
esac

export CELLPOSE_LOCAL_MODELS_PATH="${CELLPOSE_LOCAL_MODELS_PATH:-$ROOT/.cache/cellpose/models}"
export MICROSAM_MODEL_CACHE="${MICROSAM_MODEL_CACHE:-$ROOT/.cache/microsam/models}"

exec "$PYTHON" "$@"
