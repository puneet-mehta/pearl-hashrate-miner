// noising_smem_sm80.cuh
//
// Shared-memory-tiled variants of the noising kernels in noising_sm80.cuh.
// Same pattern as pearl_gemm_search_perthread_smem_sm80.cuh: cooperative
// global → shared loads of (TILE_M, CTA_BK)/(TILE_N, CTA_BK) slabs of A
// and B per K-tile, then per-thread MMAs from shared.
//
// Two kernels, sharing the same 8-warp×1-warp CTA layout:
//
//   gemm_int8_add_x_to_int8_smem_kernel
//     Wraps the Y @ Z^T int32 result with X[i,j] and narrows to int8.
//     Used by launch_add_gemm_int8 for ApEA / BpEB. B-side operand
//     (Z) is (N, K_inner) row-major — same layout sB pattern as
//     search_perthread_smem.
//
//   gemm_int8_int32_smem_kernel
//     Pure int8 matmul into int32 output. Used for AxEBL and EARxBpEB.
//     B operand is (K, N) row-major: cooperative load writes sB in
//     (K, N) shared layout, MMA fragment loads use 4 single-byte
//     shared reads per fragment (vs 1 u32 read for the (N, K) sB
//     layout). The cost is small relative to global-load savings.
//
// Bit-exactness invariant: each output element is the same scalar sum
// over K (or K_inner) input pairs. Integer addition is associative, so
// any reordering within the K accumulation produces byte-identical
// results. Per-output store positions and the X-add-then-wrap_int8
// behavior are preserved exactly.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "noising_sm80.cuh"   // for mma_m16n8k32_s8s8s32 and pack4

namespace pearl::sm80::noising_smem {

constexpr int TILE_M = 128;
constexpr int TILE_N = 128;
constexpr int ATOM_M = 16;
constexpr int ATOM_N = 8;
constexpr int ATOM_K = 32;
constexpr int CTA_BK = 128;
constexpr int K_BLOCKS_PER_TILE = CTA_BK / ATOM_K;   // 4
constexpr int NUM_WARPS = TILE_M / ATOM_M;           // 8
constexpr int N_ATOMS_PER_WARP = TILE_N / ATOM_N;    // 16
constexpr int CTA_THREADS = NUM_WARPS * 32;          // 256

using pearl::sm80::noising::mma_m16n8k32_s8s8s32;
using pearl::sm80::noising::pack4;

// =============================================================================
//   gemm_int8_add_x_to_int8 — B is (N, K_inner) row-major
// =============================================================================
//
// Out[i,j] = wrap_int8(X[i,j] + sum_r(Y[i,r] * Z[j,r]))
//   X: (M, N) int8 row-major
//   Y: (M, K_inner) int8 row-major
//   Z: (N, K_inner) int8 row-major  — accessed as Z^T (K_inner, N) for MMA
//   Out: (M, N) int8 row-major
//
// For mining shapes, K_inner=R=128. CTA covers (TILE_M=128, TILE_N=128) of
// output. K_inner is full per K-tile (no loop). 8 warps in M, each running
// 16 N-atoms of the m16n8k32.s8 MMA → covers (128, 128) per CTA.

__global__ void gemm_int8_add_x_to_int8_smem_kernel(
    int M, int N, int K_inner,
    const int8_t* __restrict__ X,       // (M, N) — added to result
    const int8_t* __restrict__ Y,       // (M, K_inner) row-major
    const int8_t* __restrict__ Z,       // (N, K_inner) row-major
    int8_t* __restrict__ Out) {         // (M, N) int8 row-major

  __shared__ alignas(16) int8_t sA[TILE_M * CTA_BK];   // 16 KB
  __shared__ alignas(16) int8_t sB[TILE_N * CTA_BK];   // 16 KB

  const int tile_m_idx = blockIdx.y;
  const int tile_n_idx = blockIdx.x;
  const int tile_m_base = tile_m_idx * TILE_M;
  const int tile_n_base = tile_n_idx * TILE_N;

  const int tid     = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane_id = tid & 31;
  const int gid     = lane_id >> 2;
  const int tig     = lane_id & 3;
  const int warp_row_base = warp_id * ATOM_M;

  // Per-thread accumulator: 16 N-atoms × 4 int32 per atom.
  uint32_t accum[N_ATOMS_PER_WARP][4];
  #pragma unroll
  for (int na = 0; na < N_ATOMS_PER_WARP; ++na) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) accum[na][j] = 0u;
  }

  // K_inner is the entire matmul depth (no outer K-tile loop). For our
  // mining shape K_inner=128 = CTA_BK exactly. Generalize for safety:
  // any K_inner ≤ CTA_BK fits in one tile; sA/sB beyond K_inner stay 0
  // and contribute 0 to the dot product.
  const int k_tile_count = (K_inner + CTA_BK - 1) / CTA_BK;

  for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
    const int k_base = k_tile * CTA_BK;
    const int k_remain = (K_inner - k_base) < CTA_BK ? (K_inner - k_base) : CTA_BK;

    // Cooperative loads. 256 threads × 64 bytes each = 16 KB per tile.
    // Layout matches search_perthread_smem: 2 threads per row, each owning
    // 64 contiguous bytes (4 × int4 = 16 bytes/load).
    {
      const int row_offset = tid >> 1;
      const int col_offset = (tid & 1) * 64;
      // sA[row_offset][col_offset..col_offset+63] ← Y[tile_m_base+row_offset][k_base+col_offset..]
      int8_t* a_dst = sA + row_offset * CTA_BK + col_offset;
      int8_t* b_dst = sB + row_offset * CTA_BK + col_offset;
      if (k_remain == CTA_BK) {
        // Fast path: full tile. No bounds checking.
        const int8_t* a_src = Y + (tile_m_base + row_offset) * K_inner + k_base + col_offset;
        const int8_t* b_src = Z + (tile_n_base + row_offset) * K_inner + k_base + col_offset;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          *reinterpret_cast<int4*>(a_dst + i * 16) =
              *reinterpret_cast<const int4*>(a_src + i * 16);
          *reinterpret_cast<int4*>(b_dst + i * 16) =
              *reinterpret_cast<const int4*>(b_src + i * 16);
        }
      } else {
        // Tail tile: zero-fill the remainder so unused k accumulates 0.
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
          int k_local = col_offset + i;
          a_dst[i] = (k_local < k_remain)
              ? Y[(tile_m_base + row_offset) * K_inner + k_base + k_local]
              : (int8_t)0;
          b_dst[i] = (k_local < k_remain)
              ? Z[(tile_n_base + row_offset) * K_inner + k_base + k_local]
              : (int8_t)0;
        }
      }
    }
    __syncthreads();

    // Run K_BLOCKS_PER_TILE k_blocks of MMAs from shared memory.
    #pragma unroll
    for (int k_block = 0; k_block < K_BLOCKS_PER_TILE; ++k_block) {
      const int k_offset = k_block * ATOM_K;

      // A fragment: 4 u32 from sA.
      uint32_t a_frag[4];
      {
        const int rA = warp_row_base + gid;
        const int rB = warp_row_base + gid + 8;
        const int kA = k_offset + 4 * tig;
        const int kB = k_offset + 4 * tig + 16;
        a_frag[0] = *reinterpret_cast<const uint32_t*>(sA + rA * CTA_BK + kA);
        a_frag[1] = *reinterpret_cast<const uint32_t*>(sA + rB * CTA_BK + kA);
        a_frag[2] = *reinterpret_cast<const uint32_t*>(sA + rA * CTA_BK + kB);
        a_frag[3] = *reinterpret_cast<const uint32_t*>(sA + rB * CTA_BK + kB);
      }

      #pragma unroll
      for (int na = 0; na < N_ATOMS_PER_WARP; ++na) {
        const int col = na * ATOM_N + gid;
        uint32_t b_frag[2];
        b_frag[0] = *reinterpret_cast<const uint32_t*>(
            sB + col * CTA_BK + k_offset + 4 * tig);
        b_frag[1] = *reinterpret_cast<const uint32_t*>(
            sB + col * CTA_BK + k_offset + 4 * tig + 16);
        mma_m16n8k32_s8s8s32(accum[na], a_frag, b_frag);
      }
    }
    __syncthreads();
  }

  // Add X and wrap to int8. Write all 16 N-atoms × 4 elements per thread.
  const int rowA = tile_m_base + warp_row_base + gid;
  const int rowB = tile_m_base + warp_row_base + gid + 8;
  auto add_narrow = [](int8_t x, uint32_t ea) -> int8_t {
    int32_t v = static_cast<int32_t>(x) + static_cast<int32_t>(ea);
    return static_cast<int8_t>(v & 0xff);  // two's-complement wrap
  };
  #pragma unroll
  for (int na = 0; na < N_ATOMS_PER_WARP; ++na) {
    const int colA = tile_n_base + na * ATOM_N + 2 * tig;
    const int colB = colA + 1;
    Out[rowA * N + colA] = add_narrow(X[rowA * N + colA], accum[na][0]);
    Out[rowA * N + colB] = add_narrow(X[rowA * N + colB], accum[na][1]);
    Out[rowB * N + colA] = add_narrow(X[rowB * N + colA], accum[na][2]);
    Out[rowB * N + colB] = add_narrow(X[rowB * N + colB], accum[na][3]);
  }
}

// =============================================================================
//   gemm_int8_int32 — B is (K, N) row-major
// =============================================================================
//
// C[i,j] = sum_k(A[i,k] * B[k,j])
//   A: (M, K) int8 row-major
//   B: (K, N) int8 row-major  — accessed as B[k][n], k major, n minor.
//   C: (M, N) int32 row-major
//
// For mining usage M = m or n, N = R = 128, K = k = 4096. So 32 K-tiles.
// sB stays in (K, N) layout — that matches the global load pattern and is
// coalesced. MMA fragment loads do 4 single-byte shared reads per u32 frag
// (vs 1 contiguous u32 for the (N, K) sB layout); shared bandwidth makes
// that affordable compared to the global-load savings.

__global__ void gemm_int8_int32_smem_kernel(
    int M, int N, int K,
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t* __restrict__ C) {

  __shared__ alignas(16) int8_t sA[TILE_M * CTA_BK];           // 16 KB
  __shared__ alignas(16) int8_t sB[CTA_BK * TILE_N];           // 16 KB ((K, N) layout)

  const int tile_m_idx = blockIdx.y;
  const int tile_n_idx = blockIdx.x;
  const int tile_m_base = tile_m_idx * TILE_M;
  const int tile_n_base = tile_n_idx * TILE_N;

  const int tid     = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane_id = tid & 31;
  const int gid     = lane_id >> 2;
  const int tig     = lane_id & 3;
  const int warp_row_base = warp_id * ATOM_M;

  uint32_t accum[N_ATOMS_PER_WARP][4];
  #pragma unroll
  for (int na = 0; na < N_ATOMS_PER_WARP; ++na) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) accum[na][j] = 0u;
  }

  const int num_k_tiles = (K + CTA_BK - 1) / CTA_BK;

  for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
    const int k_base = k_tile * CTA_BK;

    // Cooperative loads.
    //   sA[row][col] ← A[tile_m_base+row][k_base+col]  (M-major in shared)
    //   sB[row][col] ← B[k_base+row][tile_n_base+col]  (K-major in shared)
    // Both are coalesced reads from row-major globals; both are coalesced
    // writes (no transpose) to row-major shared.
    {
      const int row_offset = tid >> 1;
      const int col_offset = (tid & 1) * 64;
      const int8_t* a_src = A + (tile_m_base + row_offset) * K + k_base + col_offset;
      const int8_t* b_src = B + (k_base + row_offset) * N + tile_n_base + col_offset;
      int8_t* a_dst = sA + row_offset * CTA_BK + col_offset;
      int8_t* b_dst = sB + row_offset * TILE_N + col_offset;
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        *reinterpret_cast<int4*>(a_dst + i * 16) =
            *reinterpret_cast<const int4*>(a_src + i * 16);
        *reinterpret_cast<int4*>(b_dst + i * 16) =
            *reinterpret_cast<const int4*>(b_src + i * 16);
      }
    }
    __syncthreads();

    #pragma unroll
    for (int k_block = 0; k_block < K_BLOCKS_PER_TILE; ++k_block) {
      const int k_offset = k_block * ATOM_K;

      uint32_t a_frag[4];
      {
        const int rA = warp_row_base + gid;
        const int rB = warp_row_base + gid + 8;
        const int kA = k_offset + 4 * tig;
        const int kB = k_offset + 4 * tig + 16;
        a_frag[0] = *reinterpret_cast<const uint32_t*>(sA + rA * CTA_BK + kA);
        a_frag[1] = *reinterpret_cast<const uint32_t*>(sA + rB * CTA_BK + kA);
        a_frag[2] = *reinterpret_cast<const uint32_t*>(sA + rA * CTA_BK + kB);
        a_frag[3] = *reinterpret_cast<const uint32_t*>(sA + rB * CTA_BK + kB);
      }

      // B fragment from (K, N) layout: b_frag[i] needs 4 consecutive K
      // bytes from a single N column. Those bytes live at sB[k][col]
      // for k in {k_offset+4*tig..+3} (and +16..+19) — *not* contiguous
      // in (K, N) layout (each is 128 bytes apart). Use byte-wise reads.
      #pragma unroll
      for (int na = 0; na < N_ATOMS_PER_WARP; ++na) {
        const int col = na * ATOM_N + gid;
        const int8_t b0a = sB[(k_offset + 4 * tig + 0) * TILE_N + col];
        const int8_t b0b = sB[(k_offset + 4 * tig + 1) * TILE_N + col];
        const int8_t b0c = sB[(k_offset + 4 * tig + 2) * TILE_N + col];
        const int8_t b0d = sB[(k_offset + 4 * tig + 3) * TILE_N + col];
        const int8_t b1a = sB[(k_offset + 4 * tig + 16 + 0) * TILE_N + col];
        const int8_t b1b = sB[(k_offset + 4 * tig + 16 + 1) * TILE_N + col];
        const int8_t b1c = sB[(k_offset + 4 * tig + 16 + 2) * TILE_N + col];
        const int8_t b1d = sB[(k_offset + 4 * tig + 16 + 3) * TILE_N + col];
        uint32_t b_frag[2];
        b_frag[0] = pack4(b0a, b0b, b0c, b0d);
        b_frag[1] = pack4(b1a, b1b, b1c, b1d);
        mma_m16n8k32_s8s8s32(accum[na], a_frag, b_frag);
      }
    }
    __syncthreads();
  }

  // Store int32 outputs. Same per-thread (row, col) layout as the legacy
  // 1-warp-per-tile kernel; this writes 64 int32 per thread.
  const int rowA = tile_m_base + warp_row_base + gid;
  const int rowB = tile_m_base + warp_row_base + gid + 8;
  #pragma unroll
  for (int na = 0; na < N_ATOMS_PER_WARP; ++na) {
    const int colA = tile_n_base + na * ATOM_N + 2 * tig;
    const int colB = colA + 1;
    C[rowA * N + colA] = static_cast<int32_t>(accum[na][0]);
    C[rowA * N + colB] = static_cast<int32_t>(accum[na][1]);
    C[rowB * N + colA] = static_cast<int32_t>(accum[na][2]);
    C[rowB * N + colB] = static_cast<int32_t>(accum[na][3]);
  }
}

// =============================================================================
//   Launchers
// =============================================================================

inline void launch_add_gemm_int8_smem(
    int M, int N, int K_inner,
    const int8_t* d_X, const int8_t* d_Y, const int8_t* d_Z,
    int8_t* d_Out, cudaStream_t stream = nullptr) {
  // Mining-path constraints — keep these as runtime checks since callers
  // (vllm-miner included) may legitimately pass shapes the legacy kernel
  // handles but the smem variant doesn't.
  if ((M % TILE_M) != 0 || (N % TILE_N) != 0) {
    // Fall back to legacy kernel for non-aligned shapes.
    pearl::sm80::noising::launch_add_gemm_int8(
        M, N, K_inner, d_X, d_Y, d_Z, d_Out, stream);
    return;
  }
  dim3 grid(N / TILE_N, M / TILE_M);
  dim3 block(CTA_THREADS);
  gemm_int8_add_x_to_int8_smem_kernel<<<grid, block, 0, stream>>>(
      M, N, K_inner, d_X, d_Y, d_Z, d_Out);
}

inline void launch_gemm_int8_int32_smem(
    int M, int N, int K,
    const int8_t* d_A, const int8_t* d_B,
    int32_t* d_C,
    int b_stride_k, int b_stride_n,
    cudaStream_t stream = nullptr) {
  // The smem kernel assumes B is (K, N) row-major (b_stride_k = N,
  // b_stride_n = 1). Fall back to the legacy kernel for other layouts.
  if (b_stride_k != N || b_stride_n != 1
      || (M % TILE_M) != 0 || (N % TILE_N) != 0) {
    pearl::sm80::noising::launch_gemm_int8_int32(
        M, N, K, d_A, d_B, d_C, b_stride_k, b_stride_n, stream);
    return;
  }
  dim3 grid(N / TILE_N, M / TILE_M);
  dim3 block(CTA_THREADS);
  gemm_int8_int32_smem_kernel<<<grid, block, 0, stream>>>(
      M, N, K, d_A, d_B, d_C);
}

}  // namespace pearl::sm80::noising_smem
