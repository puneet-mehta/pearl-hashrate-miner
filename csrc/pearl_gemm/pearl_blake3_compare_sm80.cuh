// pearl_blake3_compare_sm80.cuh
//
// Post-pass kernel for the Triton search drop-in (paired Triton variant). Takes
// per-(tile, thread) 16-u32 transcripts produced by the Triton search
// kernel and runs:
//   1. Per-thread Blake3 compress(transcript || pow_key)
//   2. uint256 compare hash vs pow_target
// Writes per-(tile, thread) hashes + hit bytes that pow_scan_emit
// then scans to find the first hit.
//
// Mirrors the Blake3+compare logic in pearl_gemm_search_perthread_sm80.cuh
// (lines 217-242), extracted as a standalone kernel so we can run it on
// Triton's output without coupling to the C++ search kernel.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "blake3_sm80.cuh"

namespace pearl::sm80::triton_postpass {

__global__ void blake3_compare_kernel(
    const uint32_t* __restrict__ d_transcripts,   // (num_tiles * THREADS_PER_TILE, 16)
    const uint32_t* __restrict__ pow_key,         // (8,)
    const uint32_t* __restrict__ pow_target,      // (8,)
    uint32_t* __restrict__ d_hash,                // (num_tiles * THREADS_PER_TILE * 8)
    uint8_t* __restrict__ d_hit,                  // (num_tiles * THREADS_PER_TILE)
    int total_threads) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_threads) return;

  uint32_t msg[16];
  uint32_t cv[8];

  // Load transcript + pow_key
  const uint32_t* transcript = d_transcripts + idx * 16;
  #pragma unroll
  for (int i = 0; i < 16; ++i) msg[i] = transcript[i];
  #pragma unroll
  for (int i = 0; i < 8; ++i) cv[i] = pow_key[i];

  // Compress (single-block keyed Blake3, mirrors search_perthread)
  pearl::sm80::blake3::compress_msg_block_u32(
      msg, cv, pearl::sm80::blake3::make_single_block_keyed_params());

  // uint256 LE compare: hit if cv (as uint256 little-endian) <= pow_target.
  bool hit = true;
  #pragma unroll
  for (int i = 7; i >= 0; --i) {
    const uint32_t hi = cv[i];
    const uint32_t ti = pow_target[i];
    if (hi > ti) { hit = false; break; }
    if (hi < ti) {               break; }
  }

  // Writeback
  uint32_t* out_hash = d_hash + idx * 8;
  #pragma unroll
  for (int i = 0; i < 8; ++i) out_hash[i] = cv[i];
  d_hit[idx] = hit ? 1u : 0u;
}

inline void launch_blake3_compare(
    const uint32_t* d_transcripts,
    const uint32_t* pow_key,
    const uint32_t* pow_target,
    uint32_t* d_hash,
    uint8_t*  d_hit,
    int total_threads,
    cudaStream_t stream) {
  const int block = 256;
  const int grid  = (total_threads + block - 1) / block;
  blake3_compare_kernel<<<grid, block, 0, stream>>>(
      d_transcripts, pow_key, pow_target, d_hash, d_hit, total_threads);
}

}  // namespace pearl::sm80::triton_postpass
