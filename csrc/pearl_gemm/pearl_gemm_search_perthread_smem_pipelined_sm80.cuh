// pearl_gemm_search_perthread_smem_pipelined_sm80.cuh
//
// cp.async double-buffered variant of the shared-mem-tiled per-thread
// PoW search kernel. Builds on pearl_gemm_search_perthread_smem_sm80.cuh
// by overlapping the next K-tile's global → shared loads with the
// current tile's MMAs.
//
// Pipelining model (2 stages):
//   Pre-loop:  cp.async load tile 0 → sA[0], sB[0]; commit.
//   Iter S:    if S+1 < N: cp.async load tile S+1 → sA[(S+1)%2], sB[(S+1)%2]; commit.
//              __pipeline_wait_prior(1 if has_next else 0).
//              __syncthreads.
//              MMAs on sA[S%2], sB[S%2].
//              __syncthreads — gate before next iter's prefetch
//                              overwrites the (S%2)^1 buffer.
//
// Bit-exactness: identical per-thread accumulator layout, identical
// reduce-firing schedule, identical Blake3 finalize. Only the path
// from global → shared changes (cp.async instead of synchronous int4
// copies); the path from shared → register and the MMA atom sequence
// are unchanged.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#include "blake3_sm80.cuh"

namespace pearl::sm80::search_perthread_smem_pipelined {

// Layout constants — keep in sync with the smem reference variant.
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
constexpr int THREADS_PER_TILE = CTA_THREADS;
constexpr int MSG_BLOCK_SIZE_U32 = 16;

__device__ __forceinline__ void
mma_m16n8k32_s8s8s32(uint32_t d[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]));
}

template <int R>
__global__ void pearl_gemm_search_perthread_smem_pipelined_kernel(
    int M, int N, int K,
    const int8_t* __restrict__ ApEA,
    const int8_t* __restrict__ BpEB,
    const uint32_t* __restrict__ pow_key,
    const uint32_t* __restrict__ pow_target,
    uint32_t* __restrict__ hash_per_tile_thread,
    uint8_t*  __restrict__ hit_per_tile_thread,
    uint32_t* __restrict__ transcript_per_tile_thread) {
  (void)M; (void)N;
  static_assert(R == 64 || R == 128, "R must be 64 or 128");
  constexpr int REDUCE_EVERY_K  = R / ATOM_K;
  constexpr int ACCUMS_PER_TILE =
      (K_BLOCKS_PER_TILE / REDUCE_EVERY_K > 0)
          ? (K_BLOCKS_PER_TILE / REDUCE_EVERY_K) : 1;

  // Triple-buffered shared. Three stages × (TILE_M=128) × (CTA_BK=128)
  // = 48 KB for sA + 48 KB for sB = 96 KB / CTA. Just under sm_80's
  // 99 KB dynamic shared cap; the launcher opts in via
  // cudaFuncAttributeMaxDynamicSharedMemorySize. The extra stage lets
  // TWO cp.async groups stay in flight while MMAs of the current tile
  // run — more load-latency hiding than the 2-stage variant. Same
  // 1 CTA/SM occupancy ceiling either way (96 KB > 100/2 = 50 KB), so
  // no occupancy regression.
  constexpr int kNumStages = 3;
  extern __shared__ __align__(16) int8_t s_buf[];
  int8_t* const sA = s_buf;
  int8_t* const sB = s_buf + kNumStages * TILE_M * CTA_BK;

  const int tile_m_idx = blockIdx.y;
  const int tile_n_idx = blockIdx.x;
  const int num_tile_n = gridDim.x;

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

  uint32_t transcript[MSG_BLOCK_SIZE_U32];
  uint32_t m_tile_transcript[ACCUMS_PER_TILE];
  uint32_t m_reduction_count = 0u;
  uint32_t m_k_block_count   = 0u;
  #pragma unroll
  for (int i = 0; i < MSG_BLOCK_SIZE_U32; ++i) transcript[i] = 0u;
  #pragma unroll
  for (int i = 0; i < ACCUMS_PER_TILE; ++i) m_tile_transcript[i] = 0u;

  const int num_k_blocks      = K / ATOM_K;
  const int last_full_k_block = num_k_blocks;
  const int num_k_tiles       = K / CTA_BK;

  // Cooperative cp.async load helper: each thread issues 4×16-byte
  // cp.async loads for its half-row of A and B into stage[0|1].
  // Layout: thread tid owns row (tid >> 1), cols [(tid & 1)*64 .. +63].
  auto issue_load_tile = [&](int k_base, int stage) {
    const int row_offset = tid >> 1;
    const int col_offset = (tid & 1) * 64;
    int8_t* a_dst = sA + stage * (TILE_M * CTA_BK)
                       + row_offset * CTA_BK + col_offset;
    int8_t* b_dst = sB + stage * (TILE_N * CTA_BK)
                       + row_offset * CTA_BK + col_offset;
    const int8_t* a_src =
        ApEA + (tile_m_base + row_offset) * K + k_base + col_offset;
    const int8_t* b_src =
        BpEB + (tile_n_base + row_offset) * K + k_base + col_offset;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      __pipeline_memcpy_async(a_dst + i * 16, a_src + i * 16, 16);
      __pipeline_memcpy_async(b_dst + i * 16, b_src + i * 16, 16);
    }
  };

  // Pre-loop: prefetch the first up-to-(kNumStages - 1) tiles so the
  // pipeline starts already populated. With 3 stages we issue tiles 0
  // and 1 here; tile 2's prefetch fires inside iter 0 of the loop.
  // For K-tile counts smaller than kNumStages - 1, we just issue what's
  // available (degenerate but correct).
  #pragma unroll
  for (int s = 0; s < kNumStages - 1; ++s) {
    if (s < num_k_tiles) {
      issue_load_tile(s * CTA_BK, s % kNumStages);
      __pipeline_commit();
    }
  }

  for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
    const int stage = k_tile % kNumStages;
    const int next_prefetch = k_tile + (kNumStages - 1);  // tile we issue THIS iter
    const bool prefetch_in_flight = next_prefetch < num_k_tiles;

    // Preload transcript values that this K-tile's reductions will
    // rotl-then-XOR into. Matches the non-pipelined smem kernel.
    #pragma unroll
    for (int i = 0; i < ACCUMS_PER_TILE; ++i) {
      m_tile_transcript[i] = transcript[(m_reduction_count + i)
                                        & (MSG_BLOCK_SIZE_U32 - 1)];
    }

    // Issue the next pre-fetch (kNumStages - 1 ahead of the current
    // K-tile), keeping the pipeline depth saturated.
    if (prefetch_in_flight) {
      issue_load_tile(next_prefetch * CTA_BK,
                      next_prefetch % kNumStages);
      __pipeline_commit();
    }

    // Wait until the CURRENT tile's load completes. In-flight count at
    // this point = min(kNumStages - 1, num_k_tiles - 1 - k_tile) future
    // tiles still queued; wait_prior leaves that many in flight.
    const int wait_for = (num_k_tiles - 1 - k_tile < kNumStages - 1)
                             ? (num_k_tiles - 1 - k_tile)
                             : (kNumStages - 1);
    // __pipeline_wait_prior takes a compile-time integer; dispatch by
    // value with explicit if/else (kNumStages - 1 = 2, so 3 possible
    // values: 0, 1, 2).
    if (wait_for <= 0) {
      __pipeline_wait_prior(0);
    } else if (wait_for == 1) {
      __pipeline_wait_prior(1);
    } else {
      __pipeline_wait_prior(2);
    }
    __syncthreads();

    // ---- Run K_BLOCKS_PER_TILE k_blocks of MMAs from sA[stage]/sB[stage] ----
    const int8_t* sA_stage = sA + stage * (TILE_M * CTA_BK);
    const int8_t* sB_stage = sB + stage * (TILE_N * CTA_BK);

    #pragma unroll
    for (int k_block = 0; k_block < K_BLOCKS_PER_TILE; ++k_block) {
      const int k_offset = k_block * ATOM_K;

      uint32_t a_frag[4];
      {
        const int rA = warp_row_base + gid;
        const int rB = warp_row_base + gid + 8;
        const int kA = k_offset + 4 * tig;
        const int kB = k_offset + 4 * tig + 16;
        a_frag[0] = *reinterpret_cast<const uint32_t*>(sA_stage + rA * CTA_BK + kA);
        a_frag[1] = *reinterpret_cast<const uint32_t*>(sA_stage + rB * CTA_BK + kA);
        a_frag[2] = *reinterpret_cast<const uint32_t*>(sA_stage + rA * CTA_BK + kB);
        a_frag[3] = *reinterpret_cast<const uint32_t*>(sA_stage + rB * CTA_BK + kB);
      }

      #pragma unroll
      for (int na = 0; na < N_ATOMS_PER_WARP; ++na) {
        const int col = na * ATOM_N + gid;
        uint32_t b_frag[2];
        b_frag[0] = *reinterpret_cast<const uint32_t*>(
            sB_stage + col * CTA_BK + k_offset + 4 * tig);
        b_frag[1] = *reinterpret_cast<const uint32_t*>(
            sB_stage + col * CTA_BK + k_offset + 4 * tig + 16);
        mma_m16n8k32_s8s8s32(accum[na], a_frag, b_frag);
      }

      ++m_k_block_count;
      const bool do_reduce =
          (m_k_block_count % REDUCE_EVERY_K == 0u) &&
          (m_k_block_count <= (uint32_t)last_full_k_block);

      if (do_reduce) {
        uint32_t pt_hash = 0u;
        #pragma unroll
        for (int na = 0; na < N_ATOMS_PER_WARP; ++na) {
          #pragma unroll
          for (int j = 0; j < 4; ++j) pt_hash ^= accum[na][j];
        }
        const int idx = k_block / REDUCE_EVERY_K;
        const uint32_t prev = m_tile_transcript[idx];
        m_tile_transcript[idx] = ((prev << 13) | (prev >> 19)) ^ pt_hash;
      }
    }

    // Writeback this K-tile's transcript additions; rotate the cursor.
    #pragma unroll
    for (int i = 0; i < ACCUMS_PER_TILE; ++i) {
      transcript[(m_reduction_count + i) & (MSG_BLOCK_SIZE_U32 - 1)] =
          m_tile_transcript[i];
    }
    m_reduction_count = (m_reduction_count + ACCUMS_PER_TILE)
                        & (MSG_BLOCK_SIZE_U32 - 1);

    // Gate before the NEXT iter's prefetch (issued at the top of iter
    // S+1) starts overwriting the buffer this iter just read from. The
    // prefetch targets sA[(S+1)%2 ^ 1] = sA[S%2] = the buffer we just
    // consumed; without this barrier a slow warp could still be reading
    // sA[stage] when a fast warp's cp.async write to the same buffer
    // for iter S+2's prefetch lands.
    __syncthreads();
  }

  // ---- Per-thread Blake3 finalize + uint256 compare (unchanged) -------
  uint32_t hash_cv[8];
  uint32_t msg[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) msg[i] = transcript[i];
  #pragma unroll
  for (int i = 0; i < 8; ++i) hash_cv[i] = pow_key[i];
  pearl::sm80::blake3::compress_msg_block_u32(
      msg, hash_cv,
      pearl::sm80::blake3::make_single_block_keyed_params());

  bool hit = true;
  #pragma unroll
  for (int i = 7; i >= 0; --i) {
    const uint32_t hi = hash_cv[i];
    const uint32_t ti = pow_target[i];
    if (hi > ti) { hit = false; break; }
    if (hi < ti) {              break; }
  }

  const int tile_linear = tile_m_idx * num_tile_n + tile_n_idx;
  const int out_base    = (tile_linear * THREADS_PER_TILE + tid) * 8;
  uint32_t* out_hash = hash_per_tile_thread + out_base;
  #pragma unroll
  for (int i = 0; i < 8; ++i) out_hash[i] = hash_cv[i];
  hit_per_tile_thread[tile_linear * THREADS_PER_TILE + tid] = hit ? 1u : 0u;

  if (transcript_per_tile_thread != nullptr) {
    const int t_base = (tile_linear * THREADS_PER_TILE + tid) * 16;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      transcript_per_tile_thread[t_base + i] = transcript[i];
    }
  }
}

template <int R>
inline void launch_pearl_gemm_search_perthread_smem_pipelined_R(
    int M, int N, int K,
    const int8_t* d_ApEA, const int8_t* d_BpEB,
    const uint32_t* d_pow_key, const uint32_t* d_pow_target,
    uint32_t* d_hash_per_tile_thread, uint8_t* d_hit_per_tile_thread,
    cudaStream_t stream = nullptr,
    uint32_t* d_transcript_per_tile_thread = nullptr) {
  dim3 grid(N / TILE_N, M / TILE_M);
  dim3 block(CTA_THREADS);
  // 96 KB dynamic shared (3 × (TILE_M × CTA_BK) for sA, same for sB).
  // sm_80 + needs opt-in for > 48 KB; idempotent across launches.
  // Must match kNumStages in the kernel above (currently 3).
  constexpr int kNumStages = 3;
  constexpr int smem_bytes =
      kNumStages * TILE_M * CTA_BK + kNumStages * TILE_N * CTA_BK;
  static bool attr_set = false;
  if (!attr_set) {
    cudaFuncSetAttribute(
        pearl_gemm_search_perthread_smem_pipelined_kernel<R>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    attr_set = true;
  }
  pearl_gemm_search_perthread_smem_pipelined_kernel<R>
      <<<grid, block, smem_bytes, stream>>>(
          M, N, K, d_ApEA, d_BpEB, d_pow_key, d_pow_target,
          d_hash_per_tile_thread, d_hit_per_tile_thread,
          d_transcript_per_tile_thread);
}

inline void launch_pearl_gemm_search_perthread_smem_pipelined(
    int R, int M, int N, int K,
    const int8_t* d_ApEA, const int8_t* d_BpEB,
    const uint32_t* d_pow_key, const uint32_t* d_pow_target,
    uint32_t* d_hash_per_tile_thread, uint8_t* d_hit_per_tile_thread,
    cudaStream_t stream = nullptr,
    uint32_t* d_transcript_per_tile_thread = nullptr) {
  if (R == 64) {
    launch_pearl_gemm_search_perthread_smem_pipelined_R<64>(
        M, N, K, d_ApEA, d_BpEB, d_pow_key, d_pow_target,
        d_hash_per_tile_thread, d_hit_per_tile_thread, stream,
        d_transcript_per_tile_thread);
  } else {
    launch_pearl_gemm_search_perthread_smem_pipelined_R<128>(
        M, N, K, d_ApEA, d_BpEB, d_pow_key, d_pow_target,
        d_hash_per_tile_thread, d_hit_per_tile_thread, stream,
        d_transcript_per_tile_thread);
  }
}

}  // namespace pearl::sm80::search_perthread_smem_pipelined
