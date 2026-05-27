// noising_sm80.cuh
//
// Standalone Ampere/Ada ports of the noisingA / noisingB kernels.
//
// What the upstream does
// ----------------------
// noisingA (per `compute_ref_noise_A` in tests/test_pearl_gemm.py):
//   EA       = EAL @ EAR_R_major.T                      (m, k) int32
//   ApEA     = (A + EA) cast to int8 (wrap-around)      (m, k) int8
//   AxEBL    = A @ EBL_R_major                          (m, R) int32
//   AxEBL_f16 = (AxEBL * 2**-14) cast to fp16           (m, R) float16
//
// noisingB (per `compute_ref_noise_B`):
//   EB       = EBR @ EBL_R_major.T                      (n, k) int32
//   BpEB     = (B + EB) cast to int8 (wrap-around)      (n, k) int8
//   EARxBpEB    = BpEB @ EAR_R_major                    (n, R) int32
//   EARxBpEB_f16 = (EARxBpEB * 2**-12) cast to fp16     (n, R) float16
//
// All matmuls are int8 × int8 → int32 with full int32 accumulation, so
// any int8-tensor-core implementation that computes the same MAC sum
// produces bit-identical output. We use mma.sync m16n8k32 (validated
// bit-exactly in hello_mma_sm80) for the multiply.
//
// This is a correctness-first implementation: each warp owns a 16×8
// output tile and loads its A/B fragments straight from gmem per K-atom.
// No shared memory, no async-copy pipelining, no swizzling. Perf tuning
// is a separate later pass; the goal here is "bit-exact, ~1 day to land,
// validated against captured Python reference fixtures".

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace pearl::sm80::noising {

// =============================================================================
//   mma.sync atom and per-lane fragment packing
// =============================================================================

__device__ __forceinline__ uint32_t pack4(int8_t a, int8_t b, int8_t c,
                                          int8_t d) {
  uint32_t out;
  uint8_t* bytes = reinterpret_cast<uint8_t*>(&out);
  bytes[0] = static_cast<uint8_t>(a);
  bytes[1] = static_cast<uint8_t>(b);
  bytes[2] = static_cast<uint8_t>(c);
  bytes[3] = static_cast<uint8_t>(d);
  return out;
}

// mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32, exactly as in
// hello_mma_sm80.cu (validated bit-exactly there). `d` is accumulator
// in/out; `a` is 4 packed-int32 (16 int8 each lane); `b` is 2 packed-int32
// (8 int8 each lane).
__device__ __forceinline__ void
mma_m16n8k32_s8s8s32(uint32_t d[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]));
}

namespace detail {

// Load 4 int8 from a row of A at row `r`, K-start `k0`. Returns a packed u32
// (bytes 0..3 = A[r, k0..k0+3]). Assumes 0 <= r < M and 0 <= k0+3 < K.
__device__ __forceinline__ uint32_t
load_a_row4(const int8_t* __restrict__ A, int M, int K, int r, int k0) {
  (void)M;  // unused; left for symmetry
  // Row-major: A[r, k0..k0+3] are 4 contiguous bytes. The matrix is
  // 1-byte-aligned, so we read individually.
  const int8_t* p = A + r * K + k0;
  return pack4(p[0], p[1], p[2], p[3]);
}

// Load 4 int8 from B at column `c`, K-start `k0`. B is described by its two
// strides — B[k, n] lives at B[k * b_stride_k + n * b_stride_n].
//   * For Y stored (K, N) row-major (variant "X @ Y"):    b_stride_k = N, b_stride_n = 1.
//   * For Y stored (N, K) row-major, used as Y^T (variant "X @ Y^T"):
//                                                          b_stride_k = 1, b_stride_n = K.
__device__ __forceinline__ uint32_t
load_b_col4(const int8_t* __restrict__ B, int K, int N, int c, int k0,
            int b_stride_k, int b_stride_n) {
  (void)K; (void)N;
  const int8_t* base = B + c * b_stride_n;
  int8_t v0 = base[(k0 + 0) * b_stride_k];
  int8_t v1 = base[(k0 + 1) * b_stride_k];
  int8_t v2 = base[(k0 + 2) * b_stride_k];
  int8_t v3 = base[(k0 + 3) * b_stride_k];
  return pack4(v0, v1, v2, v3);
}

// Per-lane MMA fragment loads matching the m16n8k32 .s8 layout documented
// in hello_mma_sm80.cu:
//   A: a[0..3] at (rowBase, k0+4*tig+{0, 16}) for {rowBase, rowBase+8}
//   B: b[0..1] at column colBase, K bytes 4*tig+{0, 16}
__device__ __forceinline__ void load_a_fragment(
    const int8_t* A, int M, int K, int tile_m, int laneId,
    int k_base, uint32_t a[4]) {
  const int gid = laneId >> 2;
  const int tig = laneId & 3;
  const int rowA = tile_m * 16 + gid;
  const int rowB = tile_m * 16 + gid + 8;
  const int koA = 4 * tig;
  const int koB = 4 * tig + 16;
  a[0] = load_a_row4(A, M, K, rowA, k_base + koA);
  a[1] = load_a_row4(A, M, K, rowB, k_base + koA);
  a[2] = load_a_row4(A, M, K, rowA, k_base + koB);
  a[3] = load_a_row4(A, M, K, rowB, k_base + koB);
}

__device__ __forceinline__ void load_b_fragment(
    const int8_t* B, int K, int N, int tile_n, int laneId,
    int k_base, int b_stride_k, int b_stride_n, uint32_t b[2]) {
  const int gid = laneId >> 2;
  const int tig = laneId & 3;
  const int col = tile_n * 8 + gid;
  const int koA = 4 * tig;
  const int koB = 4 * tig + 16;
  b[0] = load_b_col4(B, K, N, col, k_base + koA, b_stride_k, b_stride_n);
  b[1] = load_b_col4(B, K, N, col, k_base + koB, b_stride_k, b_stride_n);
}

}  // namespace detail

// =============================================================================
//   GEMM kernels (int8 × int8 → int32)
// =============================================================================

// Plain MxK × KxN -> MxN matmul, accumulated in int32.
// Grid: (N/8, M/16). One warp per output tile. Caller must guarantee that
// M is a multiple of 16, N a multiple of 8, K a multiple of 32.
__global__ void gemm_int8_int32_kernel(
    int M, int N, int K, const int8_t* __restrict__ A,
    const int8_t* __restrict__ B, int32_t* __restrict__ C,
    int b_stride_k, int b_stride_n) {
  const int laneId = threadIdx.x;
  const int tile_n = blockIdx.x;
  const int tile_m = blockIdx.y;

  uint32_t accum[4] = {0u, 0u, 0u, 0u};
  for (int k0 = 0; k0 < K; k0 += 32) {
    uint32_t a[4];
    uint32_t b[2];
    detail::load_a_fragment(A, M, K, tile_m, laneId, k0, a);
    detail::load_b_fragment(B, K, N, tile_n, laneId, k0,
                            b_stride_k, b_stride_n, b);
    mma_m16n8k32_s8s8s32(accum, a, b);
  }

  // Store the 4 int32 outputs per lane.
  const int gid = laneId >> 2;
  const int tig = laneId & 3;
  const int rowA = tile_m * 16 + gid;
  const int rowB = tile_m * 16 + gid + 8;
  const int colA = tile_n * 8 + 2 * tig;
  const int colB = tile_n * 8 + 2 * tig + 1;
  C[rowA * N + colA] = static_cast<int32_t>(accum[0]);
  C[rowA * N + colB] = static_cast<int32_t>(accum[1]);
  C[rowB * N + colA] = static_cast<int32_t>(accum[2]);
  C[rowB * N + colB] = static_cast<int32_t>(accum[3]);
}

// Fused: ApEA = (int8)(A + EAL @ EAR^T). Same grid/warp shape as above; the
// matmul produces 4 int32 lanes worth of EA, then we add the matching A
// values and narrow to int8.
//   X: the "A" operand (M, K) row-major   — A in noisingA, B in noisingB.
//   Y: the "EAL/EBR" operand (M, K_inner) — EAL in noisingA, EBR in noisingB.
//   Z: the "EAR_R/EBL_R" operand (N, K_inner) row-major — used as Y^T in MMA.
//   So the matmul is Y @ Z^T -> (M, N) int32 (the "EA" or "EB" term).
//   The fused output is `Out[i,j] = int8(X[i,j] + Y@Z^T[i,j])`.
// K_inner is R (the noise rank).
__global__ void gemm_int8_add_x_to_int8_kernel(
    int M, int N, int K_inner,
    const int8_t* __restrict__ X,         // (M, N) — yes, indexed (M, N)
    const int8_t* __restrict__ Y,         // (M, K_inner)
    const int8_t* __restrict__ Z,         // (N, K_inner), used as Z^T
    int8_t* __restrict__ Out) {           // (M, N) int8
  const int laneId = threadIdx.x;
  const int tile_n = blockIdx.x;
  const int tile_m = blockIdx.y;

  uint32_t accum[4] = {0u, 0u, 0u, 0u};
  // Y is (M, K_inner) row-major, K_inner = R. Z is (N, K_inner) row-major,
  // used as Z^T (K_inner, N). For the B operand (K_inner, N) col-major =>
  // stride 1 along K_inner, stride K_inner along N => b_stride_k=1,
  // b_stride_n=K_inner.
  const int b_stride_k = 1;
  const int b_stride_n = K_inner;
  for (int k0 = 0; k0 < K_inner; k0 += 32) {
    uint32_t a[4];
    uint32_t b[2];
    detail::load_a_fragment(Y, M, K_inner, tile_m, laneId, k0, a);
    detail::load_b_fragment(Z, K_inner, N, tile_n, laneId, k0,
                            b_stride_k, b_stride_n, b);
    mma_m16n8k32_s8s8s32(accum, a, b);
  }

  // Now accum[0..3] = EA fragment at the lane's four output positions.
  // Load the matching 4 int8 values from X, add, narrow to int8, store.
  const int gid = laneId >> 2;
  const int tig = laneId & 3;
  const int rowA = tile_m * 16 + gid;
  const int rowB = tile_m * 16 + gid + 8;
  const int colA = tile_n * 8 + 2 * tig;
  const int colB = tile_n * 8 + 2 * tig + 1;

  auto add_narrow = [](int8_t x, uint32_t ea) -> int8_t {
    int32_t v = static_cast<int32_t>(x) + static_cast<int32_t>(ea);
    return static_cast<int8_t>(v & 0xff);  // wrap to int8 (two's-complement)
  };

  Out[rowA * N + colA] = add_narrow(X[rowA * N + colA], accum[0]);
  Out[rowA * N + colB] = add_narrow(X[rowA * N + colB], accum[1]);
  Out[rowB * N + colA] = add_narrow(X[rowB * N + colA], accum[2]);
  Out[rowB * N + colB] = add_narrow(X[rowB * N + colB], accum[3]);
}

// int32 -> fp16 with multiplicative scale. Each thread does one element.
// scale_power: the exponent of 2 used to scale (e.g. -14 for AxEBL_fp16,
// -12 for EARxBpEB_fp16). Equivalent host op:
//   fp16_out = float16(int32_in.astype(float32) * 2**scale_power).
__global__ void int32_to_fp16_scaled_kernel(
    int N, const int32_t* __restrict__ src, __half* __restrict__ dst,
    int scale_power) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  // Build 2^scale_power as float via ldexpf for exact representability.
  const float scale = ldexpf(1.0f, scale_power);
  const float scaled = static_cast<float>(src[i]) * scale;
  dst[i] = __float2half(scaled);
}

// =============================================================================
//   Host launchers
// =============================================================================

// Generic int8 matmul.
//   - For "row-major A @ row-major (K,N) B"   : b_stride_k=N, b_stride_n=1.
//   - For "row-major A @ (N,K)-rmaj as B^T"    : b_stride_k=1, b_stride_n=K.
inline void launch_gemm_int8_int32(int M, int N, int K,
                                   const int8_t* d_A, const int8_t* d_B,
                                   int32_t* d_C, int b_stride_k,
                                   int b_stride_n,
                                   cudaStream_t stream = nullptr) {
  dim3 grid(N / 8, M / 16);
  dim3 block(32);  // one warp per tile
  gemm_int8_int32_kernel<<<grid, block, 0, stream>>>(M, N, K, d_A, d_B, d_C,
                                                     b_stride_k, b_stride_n);
}

// Fused ApEA / BpEB launcher. Computes Out = (int8)(X + Y @ Z^T) for
// Y (M, K_inner) row-major and Z (N, K_inner) row-major. Output (M, N) int8.
inline void launch_add_gemm_int8(int M, int N, int K_inner, const int8_t* d_X,
                                 const int8_t* d_Y, const int8_t* d_Z,
                                 int8_t* d_Out,
                                 cudaStream_t stream = nullptr) {
  dim3 grid(N / 8, M / 16);
  dim3 block(32);
  gemm_int8_add_x_to_int8_kernel<<<grid, block, 0, stream>>>(
      M, N, K_inner, d_X, d_Y, d_Z, d_Out);
}

inline void launch_int32_to_fp16_scaled(int N, const int32_t* d_src,
                                        __half* d_dst, int scale_power,
                                        cudaStream_t stream = nullptr) {
  const int block = 256;
  const int grid = (N + block - 1) / block;
  int32_to_fp16_scaled_kernel<<<grid, block, 0, stream>>>(N, d_src, d_dst,
                                                          scale_power);
}

}  // namespace pearl::sm80::noising
