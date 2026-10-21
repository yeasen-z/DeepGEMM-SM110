#!/usr/bin/env bash
#
# Build a wheel for the `sgl-deep-gemm` distribution.
#
# Distribution name: sgl-deep-gemm. Top-level import name: `deep_gemm`
# (so existing call sites like `import deep_gemm` in sglang keep working).
#
# Build flow:
#   1. Initialises submodules (cutlass, deep_jit) — same prerequisite as `bash build.sh`.
#   2. Stages the package layout under build/deep_gemm/ with the Python
#      sub-modules pulled from the source deep_gemm/ tree (utils, testing,
#      legacy, mega).
#   3. Reads the version string from sgl_deep_gemm/VERSION.
#   4. Pre-compiles the TORCH_LIBRARY `_C.so` extension and bundles it into the wheel.
#   5. Invokes `python -m build` to produce dist/*.whl.

set -euo pipefail

# Which interpreter builds _C.so decides which torch ABI _C.so is linked
# against: torch declares `c10_cuda_check_implementation` with a different
# signature in 2.13 (no CUDAErrorLogCapture* argument) than in 2.14, so a _C.so
# built against one is not importable under the other. Never let this be an
# accident of $PATH ordering -- override PYTHON_EXE, or make sure the first
# python3 on $PATH is the environment you intend to install into.
PYTHON_EXE="${PYTHON_EXE:-$(which python3 || which python)}"
ROOT_DIR=$(realpath "$(dirname "$0")")
BUILD_DIR="${ROOT_DIR}/build"
PKG_DIR="${BUILD_DIR}/deep_gemm"
DIST_DIR="${ROOT_DIR}/dist"

cd "$ROOT_DIR"

echo "--- Build interpreter ---"
echo "PYTHON_EXE: $PYTHON_EXE"
"$PYTHON_EXE" -c '
import sys, torch
print(f"python    : {sys.version.split()[0]}")
print(f"torch     : {torch.__version__}")
print(f"torch cuda: {torch.version.cuda}")
' || { echo "Error: $PYTHON_EXE cannot import torch — it must be the environment you will install into." >&2; exit 1; }

if [[ -z "${DG_SKIP_TORCH_CHECK:-}" ]]; then
    if ! "$PYTHON_EXE" -c '
import torch, re, sys, subprocess, glob, os
# Find the signature _C.so will be linked against, so a torch/pip mismatch is
# reported here rather than as "undefined symbol" at import time.
lib = os.path.join(os.path.dirname(torch.__file__), "lib", "libc10_cuda.so")
out = subprocess.run(["nm", "-D", "--defined-only", lib],
                     capture_output=True, text=True).stdout
want_new = "CUDAErrorLogCaptureE" in out
print("torch ABI: "
      + (">=2.14 (c10_cuda_check_implementation takes CUDAErrorLogCapture*)" if want_new
         else "<2.14 (c10_cuda_check_implementation has no CUDAErrorLogCapture*)"))
'; then
        echo "Note: could not probe the torch ABI via libc10_cuda.so (nm missing?); continuing." >&2
    fi
fi

if [[ ! -f "setup.py" || ! -d "sgl_deep_gemm" || ! -d "deep_gemm" || ! -d "csrc" ]]; then
    echo "Error: Run from the DeepGEMM project root." >&2
    exit 1
fi

echo "--- Initialising submodules ---"
git submodule update --init --recursive

echo "--- Linking CUTLASS headers into deep_gemm/include ---"
ln -sfn "${ROOT_DIR}/third-party/cutlass/include/cutlass" "${ROOT_DIR}/deep_gemm/include/cutlass"
ln -sfn "${ROOT_DIR}/third-party/cutlass/include/cute" "${ROOT_DIR}/deep_gemm/include/cute"

echo "--- Preparing build directory ---"
rm -rf "$BUILD_DIR"
mkdir -p "$PKG_DIR"

cp sgl_deep_gemm/LICENSE sgl_deep_gemm/README.md sgl_deep_gemm/pyproject.toml "$BUILD_DIR/"
cp sgl_deep_gemm/__init__.py sgl_deep_gemm/cuda_helpers.py sgl_deep_gemm/build_ext.py "$PKG_DIR/"

# `__init__.py` imports `.utils`, `.testing`, `.legacy`, `.mega` — pulled from
# the existing deep_gemm/ tree.
for sub in utils testing legacy mega; do
    cp -r "deep_gemm/${sub}" "$PKG_DIR/"
done

# Headers required by the runtime JIT (same set the deep_gemm wheel ships).
mkdir -p "$PKG_DIR/include"
cp -r "${ROOT_DIR}/deep_gemm/include/deep_gemm" "$PKG_DIR/include/deep_gemm"
cp -r "${ROOT_DIR}/third-party/cutlass/include/cute" "$PKG_DIR/include/cute"
cp -r "${ROOT_DIR}/third-party/cutlass/include/cutlass" "$PKG_DIR/include/cutlass"
cp -r "${ROOT_DIR}/third-party/deep_jit/include/deep_jit" "$PKG_DIR/include/deep_jit"

echo "--- Reading version from sgl_deep_gemm/VERSION ---"
if [[ ! -f "sgl_deep_gemm/VERSION" ]]; then
    echo "Error: sgl_deep_gemm/VERSION is missing — create it with the desired version (e.g. 0.0.1)." >&2
    exit 1
fi
# Strip surrounding whitespace; the file is the single source of truth.
tr -d '[:space:]' < sgl_deep_gemm/VERSION > "$PKG_DIR/VERSION"
echo "Version: $(cat "$PKG_DIR/VERSION")"

echo "--- Compiling _C.so ---"
"$PYTHON_EXE" sgl_deep_gemm/build_ext.py "$ROOT_DIR" "$PKG_DIR/_C.so"

echo "--- Installing build frontend ---"
"$PYTHON_EXE" -m pip install --quiet --upgrade build

echo "--- Building wheel ---"
mkdir -p "$DIST_DIR"
"$PYTHON_EXE" -m build --wheel "$BUILD_DIR" --outdir "$DIST_DIR"

echo "--- Done ---"
ls -lh "$DIST_DIR"/sgl_deep_gemm-*.whl 2>/dev/null || ls -lh "$DIST_DIR"/sgl-deep-gemm-*.whl 2>/dev/null || ls -lh "$DIST_DIR"/sgl_deep_gemm*.whl
