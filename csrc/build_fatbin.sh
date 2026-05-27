#!/bin/bash
# Build a multi-arch pearl_gemm.fatbin from the .cu kernel source.
# Requires a CUDA toolkit (nvcc) that knows sm_120 — i.e. CUDA 12.4+.
#
# Default output: /tmp/pearl_gemm.fatbin (override with OUT=…).
# Emits SASS for sm_86 / sm_89 / sm_120 plus sm_86 PTX as the JIT fallback
# for future arches.

set -euo pipefail

OUT=${OUT:-/tmp/pearl_gemm.fatbin}
HERE="$(cd "$(dirname "$0")" && pwd)"

nvcc \
    -O3 -std=c++17 \
    --expt-relaxed-constexpr \
    --expt-extended-lambda \
    -I "$HERE/pearl_gemm" \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_120,code=sm_120 \
    -gencode arch=compute_86,code=compute_86 \
    -fatbin \
    -o "$OUT" \
    "$HERE/pearl_gemm_kernels_only.cu"

echo "built: $OUT ($(du -h "$OUT" | cut -f1))"
