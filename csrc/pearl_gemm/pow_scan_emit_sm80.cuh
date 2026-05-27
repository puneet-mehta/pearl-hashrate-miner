// pow_scan_emit_sm80.cuh
//
// GPU-side replacement for the host-side hit scan that
// `pearl_gemm_noisy_gemm` used to do after `pearl_gemm_search_perthread`.
//
// The original pattern (cudaMemcpyAsync(d_hit -> host) + cudaStreamSynchronize
// + host scan + host writeback to pinned_header) blocked the stream every
// iteration. These two kernels stay fully async:
//
//   pow_scan_hits        — parallel scan over d_hit, atomicMin into a 4-byte
//                          device-side slot (first_hit_idx). Reset to
//                          UINT32_MAX before each call via cudaMemsetAsync.
//   pow_emit_header      — single-thread kernel that, if first_hit_idx !=
//                          UINT32_MAX, writes the 640-byte HostSignalHeader
//                          directly into the caller's pinned UVA buffer and
//                          publishes status=1 last with __threadfence_system.
//
// Byte layout of the emitted header mirrors host_signal_header.hpp exactly
// (see pearl_gemm_sm80_torch.cu's old host-side code for the offset table).
// Submitted-proof bytes are unchanged; only the path that produces them moved
// from host to device.

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace pearl {
namespace sm80 {
namespace pow_scan_emit {

// One thread per byte of d_hit; rare hits → atomicMin contention is
// negligible. Block size 256, grid sized to cover `total`.
__global__ inline void pow_scan_hits_kernel(
    const uint8_t* __restrict__ d_hit,
    int total,
    uint32_t* __restrict__ g_first_hit_idx) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total) return;
  if (d_hit[tid]) {
    atomicMin(g_first_hit_idx, static_cast<uint32_t>(tid));
  }
}

// Single-thread emit. Skipped (no writes) when first_hit_idx == UINT32_MAX.
// Layout offsets mirror the CPU writeback in pearl_gemm_noisy_gemm
// (host_signal_header.hpp under __align__(128), sizeof=640).
__global__ inline void pow_emit_header_kernel(
    const uint32_t* __restrict__ g_first_hit_idx,
    const uint32_t* __restrict__ pow_target,   // (8,) u32
    uint8_t* __restrict__ pinned_header,        // 640 B, UVA host pointer
    uint8_t* __restrict__ pinned_sync,          // optional, may be nullptr
    int num_tile_m,
    int num_tile_n,
    int threads_per_tile,
    int m,
    int n,
    int k) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  const uint32_t first_hit = *g_first_hit_idx;
  if (first_hit == 0xFFFFFFFFu) return;

  const int tile_linear = static_cast<int>(first_hit / static_cast<uint32_t>(threads_per_tile));
  const int thread_idx  = static_cast<int>(first_hit % static_cast<uint32_t>(threads_per_tile));
  const int tile_m_idx  = tile_linear / num_tile_n;
  const int tile_n_idx  = tile_linear % num_tile_n;

  auto store_u32 = [&](int off, uint32_t v) {
    *reinterpret_cast<uint32_t*>(pinned_header + off) = v;
  };
  auto store_u16 = [&](int off, uint16_t v) {
    *reinterpret_cast<uint16_t*>(pinned_header + off) = v;
  };
  auto store_i32 = [&](int off, int32_t v) {
    *reinterpret_cast<int32_t*>(pinned_header + off) = v;
  };

  // gridDim / blockDim / blockIdx / tileCoord / threadIdx (status written last).
  store_u32(4,  static_cast<uint32_t>(num_tile_n));
  store_u32(8,  static_cast<uint32_t>(num_tile_m));
  store_u32(12, 1u);
  store_u32(16, 256u); store_u32(20, 1u); store_u32(24, 1u);
  store_u32(28, static_cast<uint32_t>(tile_n_idx));
  store_u32(32, static_cast<uint32_t>(tile_m_idx));
  store_u32(36, 0u);
  store_u32(40, static_cast<uint32_t>(tile_m_idx));
  store_u32(44, static_cast<uint32_t>(tile_n_idx));
  store_u32(48, 0u);
  store_u32(52, static_cast<uint32_t>(thread_idx));
  store_u32(56, 0u); store_u32(60, 0u);

  // num_registers_per_thread @ 64 (u16) — 64 register cells per thread.
  store_u16(64, static_cast<uint16_t>(64));

  // thread_rows[64] + thread_cols[64] at offsets 66 and 322.
  // Layout per pearl_gemm_search_perthread:
  //   warp_id=thread_idx>>5, lane_id=thread_idx&31,
  //   gid=lane_id>>2,         tig=lane_id&3
  //   for na in [0,16):
  //     (row_lo, col0), (row_lo, col1), (row_hi, col0), (row_hi, col1)
  //   row_lo=warp_id*16+gid, row_hi=row_lo+8
  //   col0=na*8+2*tig, col1=col0+1
  const int warp_id = thread_idx >> 5;
  const int lane_id = thread_idx & 31;
  const int gid     = lane_id >> 2;
  const int tig     = lane_id & 3;
  const int row_lo  = warp_id * 16 + gid;
  const int row_hi  = row_lo + 8;
  #pragma unroll
  for (int na = 0; na < 16; ++na) {
    const int col0  = na * 8 + 2 * tig;
    const int col1  = col0 + 1;
    const int base  = 66 + na * 4;
    pinned_header[base + 0] = static_cast<uint8_t>(row_lo);
    pinned_header[base + 1] = static_cast<uint8_t>(row_lo);
    pinned_header[base + 2] = static_cast<uint8_t>(row_hi);
    pinned_header[base + 3] = static_cast<uint8_t>(row_hi);
    const int cbase = 322 + na * 4;
    pinned_header[cbase + 0] = static_cast<uint8_t>(col0);
    pinned_header[cbase + 1] = static_cast<uint8_t>(col1);
    pinned_header[cbase + 2] = static_cast<uint8_t>(col0);
    pinned_header[cbase + 3] = static_cast<uint8_t>(col1);
  }

  // mma_size {m, n, k} @ 580; mma_tile_size {128, 128, 128} @ 592.
  store_i32(580, m); store_i32(584, n); store_i32(588, k);
  store_i32(592, 128); store_i32(596, 128); store_i32(600, 128);

  // target uint256 (8 u32, LE) @ 604.
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    store_u32(604 + i * 4, pow_target[i]);
  }

  // Optional sync.status (mirrors the old host-side "*(sync+4) = 1" write).
  if (pinned_sync != nullptr) {
    *reinterpret_cast<uint32_t*>(pinned_sync + 4) = 1u;
  }

  // Make all field writes visible to the host before publishing status.
  __threadfence_system();
  store_u32(0, 1u);  // status = kSignalTriggered
  __threadfence_system();
}

inline void launch_pow_scan_and_emit(
    const uint8_t* d_hit,
    int num_tile_m,
    int num_tile_n,
    int threads_per_tile,
    int m, int n, int k,
    const uint32_t* pow_target,
    uint8_t* pinned_header,
    uint8_t* pinned_sync,
    uint32_t* g_first_hit_idx,
    cudaStream_t stream) {
  // Caller must cudaMemsetAsync(g_first_hit_idx, 0xFF, 4, stream) before.
  const int total = num_tile_m * num_tile_n * threads_per_tile;
  const int block = 256;
  const int grid  = (total + block - 1) / block;
  pow_scan_hits_kernel<<<grid, block, 0, stream>>>(
      d_hit, total, g_first_hit_idx);
  pow_emit_header_kernel<<<1, 1, 0, stream>>>(
      g_first_hit_idx, pow_target, pinned_header, pinned_sync,
      num_tile_m, num_tile_n, threads_per_tile, m, n, k);
}

// =============================================================================
//   Triton paired-pattern (h=2, w=128) variant — Triton drop-in
// =============================================================================
//
// The Triton search kernel uses 256 "compute threads" (8 warps × 32 lanes)
// for the matmul, but per-K-checkpoint pairs adjacent thread hashes to
// produce 128 (h=2, w=128) candidate hashes (TILE_H=2 protocol constraint).
//
// Candidate `cand_idx` (in [0, 128)) for tile (tm, tn) covers:
//   rows {tm*BM + 2*cand_idx, tm*BM + 2*cand_idx + 1}
//   cols {tn*BN .. tn*BN + 127}
//
// The header writes thread_rows = {0, 1, 0, 1, ...} (interleaved 256 entries)
// indexed by candidate-relative cell idx — actually simpler: 128 entries of
// row=2*cand_idx, then 128 of row=2*cand_idx+1, or just 256 entries with the
// (row, col) for each of the 256 cells of the candidate. The gateway dedups,
// so we just need EVERY unique row and EVERY unique col to appear at least
// once in the buffer.
//
// num_registers_per_thread = 256 (h*w = 2*128 = 256 cells per candidate).

__global__ inline void pow_emit_header_triton_paired_kernel(
    const uint32_t* __restrict__ g_first_hit_idx,
    const uint32_t* __restrict__ pow_target,
    uint8_t* __restrict__ pinned_header,
    uint8_t* __restrict__ pinned_sync,
    int num_tile_m,
    int num_tile_n,
    int hash_candidates,    // = 128 for Triton paired (= compute_threads / 2)
    int m, int n, int k,
    int block_m,            // = 256 (BM)
    int block_n,            // = 128 (BN)
    int block_k) {          // = 64  (BK)
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  const uint32_t first_hit = *g_first_hit_idx;
  if (first_hit == 0xFFFFFFFFu) return;

  const int tile_linear = static_cast<int>(first_hit / static_cast<uint32_t>(hash_candidates));
  const int cand_idx    = static_cast<int>(first_hit % static_cast<uint32_t>(hash_candidates));
  const int tile_m_idx  = tile_linear / num_tile_n;
  const int tile_n_idx  = tile_linear % num_tile_n;

  auto store_u32 = [&](int off, uint32_t v) {
    *reinterpret_cast<uint32_t*>(pinned_header + off) = v;
  };
  auto store_u16 = [&](int off, uint16_t v) {
    *reinterpret_cast<uint16_t*>(pinned_header + off) = v;
  };
  auto store_i32 = [&](int off, int32_t v) {
    *reinterpret_cast<int32_t*>(pinned_header + off) = v;
  };

  store_u32(4,  static_cast<uint32_t>(num_tile_n));
  store_u32(8,  static_cast<uint32_t>(num_tile_m));
  store_u32(12, 1u);
  store_u32(16, static_cast<uint32_t>(hash_candidates));
  store_u32(20, 1u); store_u32(24, 1u);
  store_u32(28, static_cast<uint32_t>(tile_n_idx));
  store_u32(32, static_cast<uint32_t>(tile_m_idx));
  store_u32(36, 0u);
  store_u32(40, static_cast<uint32_t>(tile_m_idx));
  store_u32(44, static_cast<uint32_t>(tile_n_idx));
  store_u32(48, 0u);
  store_u32(52, static_cast<uint32_t>(cand_idx));
  store_u32(56, 0u); store_u32(60, 0u);

  // num_registers_per_thread = 256 (h*w = 2*128). thread_rows / thread_cols
  // buffers are 256 bytes each, so this exactly fills them.
  store_u16(64, static_cast<uint16_t>(256));

  // Write 256 (row, col) cell entries for the (h=2, w=128) candidate.
  // Cell layout: i in [0, 128) → row 2*cand_idx, col i
  //              i in [128, 256) → row 2*cand_idx + 1, col i - 128
  // The gateway dedups via sorted(set(...)) on each axis, so duplicates
  // are fine — we just need the 2 unique rows and 128 unique cols to all
  // appear. Cell-pair layout matches what verifier_compute_jackpot expects.
  const uint8_t r0 = static_cast<uint8_t>(2 * cand_idx);
  const uint8_t r1 = static_cast<uint8_t>(2 * cand_idx + 1);
  #pragma unroll
  for (int i = 0; i < 128; ++i) {
    pinned_header[66 + i]         = r0;
    pinned_header[66 + 128 + i]   = r1;
    pinned_header[322 + i]        = static_cast<uint8_t>(i);
    pinned_header[322 + 128 + i]  = static_cast<uint8_t>(i);
  }

  store_i32(580, m); store_i32(584, n); store_i32(588, k);
  store_i32(592, block_m); store_i32(596, block_n); store_i32(600, block_k);

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    store_u32(604 + i * 4, pow_target[i]);
  }

  if (pinned_sync != nullptr) {
    *reinterpret_cast<uint32_t*>(pinned_sync + 4) = 1u;
  }

  __threadfence_system();
  store_u32(0, 1u);
  __threadfence_system();
}

inline void launch_pow_scan_and_emit_triton_paired(
    const uint8_t* d_hit,
    int num_tile_m,
    int num_tile_n,
    int hash_candidates,     // = 128
    int m, int n, int k,
    int block_m, int block_n, int block_k,
    const uint32_t* pow_target,
    uint8_t* pinned_header,
    uint8_t* pinned_sync,
    uint32_t* g_first_hit_idx,
    cudaStream_t stream) {
  const int total = num_tile_m * num_tile_n * hash_candidates;
  const int block = 256;
  const int grid  = (total + block - 1) / block;
  pow_scan_hits_kernel<<<grid, block, 0, stream>>>(
      d_hit, total, g_first_hit_idx);
  pow_emit_header_triton_paired_kernel<<<1, 1, 0, stream>>>(
      g_first_hit_idx, pow_target, pinned_header, pinned_sync,
      num_tile_m, num_tile_n, hash_candidates, m, n, k,
      block_m, block_n, block_k);
}

}  // namespace pow_scan_emit
}  // namespace sm80
}  // namespace pearl
