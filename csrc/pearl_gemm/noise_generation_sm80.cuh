// noise_generation_sm80.cuh
//
// Correctness-first Ampere/Ada port of `noise_generation_kernel.h`.
//
// The upstream is ~760 LOC of CUTLASS Tensor/Copy abstractions, swizzled
// shared-memory layouts, and vectorised 16-byte stores. We don't reproduce
// any of that here — the algorithm itself is just "do one keyed Blake3 hash
// per 32-byte output chunk, decode bytes into an int8 noise matrix". The
// goal is bit-for-bit agreement with the Python reference (and therefore
// with the upstream CUDA kernel — upstream's own tests assert equality
// against the reference). Performance tuning is a follow-up step.
//
// Each call to compress_msg_block_u32 uses the single-block-keyed flags
// (KEYED_HASH | CHUNK_START | CHUNK_END | ROOT), counter=0, block_len=64.
// The message is 8 int32 zeros followed by the 32-byte seed, with one
// position patched per the upstream's rule:
//   * dense matrices: message_u32[0]   = chunk_idx + 1
//   * sparse matrices: message_u32[1]  = chunk_idx + 1
// Everything else (key, seed, decoding rule) is identical between the two.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "blake3_sm80.cuh"

namespace pearl::sm80::noise_gen {

namespace detail {

// Compute one keyed Blake3 compress (single-block-keyed) into an 8-u32
// chaining value. `key` is 32 bytes copied into the chaining value;
// `message` is the 64-byte block. Both are register-resident on entry.
__device__ __forceinline__ void keyed_compress(const uint32_t key[8],
                                               uint32_t message[16],
                                               uint32_t cv[8]) {
  #pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = key[i];
  pearl::sm80::blake3::compress_msg_block_u32(
      message, cv, pearl::sm80::blake3::make_single_block_keyed_params());
}

// 8 int32 zeros followed by seed (8 u32). Then patch in position [slot].
__device__ __forceinline__ void make_message(const uint32_t seed_u32[8],
                                             uint32_t out[16], int slot,
                                             uint32_t chunk_idx_plus_one) {
  #pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = 0u;
  #pragma unroll
  for (int i = 0; i < 8; ++i) out[8 + i] = seed_u32[i];
  out[slot] = chunk_idx_plus_one;
}

// Upper 32 bits of (a * b), with a and b uint32.
__device__ __forceinline__ uint32_t mul_hi_u32(uint32_t a, uint32_t b) {
  return __umulhi(a, b);
}

}  // namespace detail

// =============================================================================
//   Dense noise (EAL / EBR / EAL_fp16 / EBR_fp16)
// =============================================================================
//
// One thread per 32-byte chunk. R must be a multiple of 32 (true for the
// only sizes the upstream supports, 64 and 128), so each chunk lives
// entirely within one row of the output.

template <int R>
__global__ void noise_gen_dense_int8_kernel(int rows,
                                            const uint32_t* __restrict__ key,
                                            const uint32_t* __restrict__ seed,
                                            int8_t* __restrict__ out) {
  static_assert(R % 32 == 0, "R must be multiple of 32");

  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_chunks = (rows * R) / 32;
  if (chunk >= num_chunks) return;

  uint32_t key_r[8];
  uint32_t seed_r[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) { key_r[i] = key[i]; seed_r[i] = seed[i]; }

  uint32_t msg[16];
  detail::make_message(seed_r, msg, /*slot=*/0,
                       /*chunk_idx_plus_one=*/static_cast<uint32_t>(chunk + 1));

  uint32_t cv[8];
  detail::keyed_compress(key_r, msg, cv);

  // Decode 32 bytes (= 8 u32) → 32 int8 in [-32, 32) and store contiguously.
  const int row = (chunk * 32) / R;
  const int col0 = (chunk * 32) % R;
  int8_t* out_row = out + row * R + col0;

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t w = cv[i];
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
      int8_t hb = static_cast<int8_t>(w & 0xff);
      w >>= 8;
      // Reference: ((int32(hb) + 128) % 64) - 32. With NOISE_ABS_MAX=128
      // and PERM_IDXS_PER_COL=2, NOISE_RANGE = 64.
      int32_t v = (static_cast<int32_t>(hb) + 128) % 64 - 32;
      out_row[i * 4 + b] = static_cast<int8_t>(v);
    }
  }
}

template <int R>
__global__ void noise_gen_dense_fp16_kernel(int rows,
                                            const uint32_t* __restrict__ key,
                                            const uint32_t* __restrict__ seed,
                                            int32_t scale_factor,
                                            __half* __restrict__ out) {
  static_assert(R % 32 == 0, "R must be multiple of 32");

  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_chunks = (rows * R) / 32;
  if (chunk >= num_chunks) return;

  uint32_t key_r[8];
  uint32_t seed_r[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) { key_r[i] = key[i]; seed_r[i] = seed[i]; }

  uint32_t msg[16];
  detail::make_message(seed_r, msg, /*slot=*/0,
                       static_cast<uint32_t>(chunk + 1));

  uint32_t cv[8];
  detail::keyed_compress(key_r, msg, cv);

  const int row = (chunk * 32) / R;
  const int col0 = (chunk * 32) % R;
  __half* out_row = out + row * R + col0;

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t w = cv[i];
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
      int8_t hb = static_cast<int8_t>(w & 0xff);
      w >>= 8;
      int32_t v = (static_cast<int32_t>(hb) + 128) % 64 - 32;
      float scaled = static_cast<float>(v * scale_factor);
      out_row[i * 4 + b] = __float2half(scaled);
    }
  }
}

// =============================================================================
//   Sparse noise (EAR_R_major / EBL_R_major; K-major is the transpose)
// =============================================================================
//
// Each chunk produces indices for 8 rows of a (k, R) int8 matrix; each
// row has exactly two non-zeros: +1 at k0, -1 at k1. R is a power of 2
// (64 or 128) so `& (R-1)` is a valid modulus.

template <int R>
__global__ void noise_gen_sparse_kernel(int k,
                                        const uint32_t* __restrict__ key,
                                        const uint32_t* __restrict__ seed,
                                        int8_t* __restrict__ out_r_major) {
  static_assert((R & (R - 1)) == 0, "R must be a power of 2");

  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_chunks = (k + 7) / 8;
  if (chunk >= num_chunks) return;

  uint32_t key_r[8];
  uint32_t seed_r[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) { key_r[i] = key[i]; seed_r[i] = seed[i]; }

  uint32_t msg[16];
  detail::make_message(seed_r, msg, /*slot=*/1,
                       static_cast<uint32_t>(chunk + 1));

  uint32_t cv[8];
  detail::keyed_compress(key_r, msg, cv);

  const int k_base = chunk * 8;

  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int row = k_base + j;
    if (row >= k) return;
    uint32_t u = cv[j];
    uint32_t k0 = u & static_cast<uint32_t>(R - 1);
    uint32_t k1 = k0 ^ (1u + detail::mul_hi_u32(static_cast<uint32_t>(R - 1), u));
    // The reference does `% R` even though k0 < R; k1 may be exactly R
    // when `mul_hi(R-1, u) = R-1` and `1 + (R-1) = R`. Modulo handles it.
    k1 = k1 % R;
    int8_t* out_row = out_r_major + row * R;
    // out is zero-initialised by the host before launch.
    out_row[k0] = 1;
    out_row[k1] = -1;
  }
}

// =============================================================================
//   Host launchers
// =============================================================================

template <int R>
inline void launch_dense_int8(int rows, const uint32_t* d_key,
                              const uint32_t* d_seed, int8_t* d_out,
                              cudaStream_t stream = nullptr) {
  const int num_chunks = (rows * R) / 32;
  const int block = 128;
  const int grid = (num_chunks + block - 1) / block;
  noise_gen_dense_int8_kernel<R><<<grid, block, 0, stream>>>(rows, d_key,
                                                             d_seed, d_out);
}

template <int R>
inline void launch_dense_fp16(int rows, const uint32_t* d_key,
                              const uint32_t* d_seed, int32_t scale_factor,
                              __half* d_out, cudaStream_t stream = nullptr) {
  const int num_chunks = (rows * R) / 32;
  const int block = 128;
  const int grid = (num_chunks + block - 1) / block;
  noise_gen_dense_fp16_kernel<R><<<grid, block, 0, stream>>>(rows, d_key,
                                                             d_seed,
                                                             scale_factor,
                                                             d_out);
}

template <int R>
inline void launch_sparse(int k, const uint32_t* d_key, const uint32_t* d_seed,
                          int8_t* d_out_r_major, cudaStream_t stream = nullptr) {
  // Caller must cudaMemset the output to 0 before launch — the kernel only
  // writes the two non-zeros per row.
  const int num_chunks = (k + 7) / 8;
  const int block = 128;
  const int grid = (num_chunks + block - 1) / block;
  noise_gen_sparse_kernel<R><<<grid, block, 0, stream>>>(k, d_key, d_seed,
                                                         d_out_r_major);
}

// Tiny transpose kernel: (k, R) int8 -> (R, k) int8. We use it to derive the
// K-major variants from the R-major sparse outputs. Not perf-tuned (one
// thread per element).
__global__ inline void transpose_kr_kernel(int k, int R,
                                           const int8_t* __restrict__ src_kr,
                                           int8_t* __restrict__ dst_rk) {
  const int kk = blockIdx.x * blockDim.x + threadIdx.x;
  const int rr = blockIdx.y * blockDim.y + threadIdx.y;
  if (kk >= k || rr >= R) return;
  dst_rk[rr * k + kk] = src_kr[kk * R + rr];
}

inline void launch_transpose_kr(int k, int R, const int8_t* d_kr,
                                int8_t* d_rk, cudaStream_t stream = nullptr) {
  dim3 block(16, 16);
  dim3 grid((k + 15) / 16, (R + 15) / 16);
  transpose_kr_kernel<<<grid, block, 0, stream>>>(k, R, d_kr, d_rk);
}

}  // namespace pearl::sm80::noise_gen
