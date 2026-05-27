// Pearl search no-rotl, hand-rolled CUDA for sm_120.
//
// V5 — bit-exact, ldmatrix.x4 for A and B, 3-stage cp.async pipelining,
// B-loads hoisted out of sm_M loop, smem padded to 80B rows.
//
// Status: torch.equal(transcripts_triton, transcripts_ours) == True on a
// CA/WA 5090 at default shape (m=8192 n=32768 k=2048 r=128).
//
// Performance: 4.77 ms / iter  →  115 TMAC/s
//   vs Triton: 1.65 ms / iter  →  333 TMAC/s
//   ratio: 0.34× (unchanged across V3, V4, V5 micro-opts)
//
// Optimizations applied and measured ineffective on this kernel:
//   • ldmatrix.x4 for A operand (replaces 4 direct u32 loads/lane/mma).
//   • ldmatrix.x4 for B (no .trans) — empirically derived via probe
//     kernel [local probe scripts].
//     Cuts B-load count in half by loading 2 mma B operands per call.
//   • 3-stage cp.async pipelining (prefetch STAGES-1 stages pre-loop,
//     prefetch (k+STAGES-1) per-iter, wait_group<STAGES-1> at top).
//   • Hoist B loads out of sm_M loop — B doesn't depend on sm_M, so
//     8 B pairs are cached in registers per kk, then both sm_M values
//     mma against them.
//   • smem row padding (BK=64 → 80 bytes) to break 8-way bank conflicts.
//   • Without checkpoint: 4.72 ms (only 0.12 ms savings = 2.5%) →
//     the XOR-reduce epilogue is NOT the bottleneck.
//
// Resource usage:
//   • PTX: ~1k lines, 20 ldmatrix, 64 mma.sync, 24 cp.async.
//   • ptxas: 206 registers, 0 spills, ~48 KB smem (limits 2 CTAs/SM).
//   • Triton PTX: 1332 lines, 16 ldmatrix, 64 mma.sync, 33 KB smem.
//
// Where the 3× gap likely lives (un-instrumented):
//   ptxas / nvcc do not interleave mma issue across the unrolled inner
//   loops. Triton's MLIR explicitly software-pipelines mma issue with
//   ldmatrix prefetch. To replicate this in CUDA needs either
//   handwritten inline-asm with explicit ldmatrix-then-mma-then-ldmatrix
//   ordering, or a higher-level pipelining IR (e.g., CUTLASS CuTe).
//
// Closing the gap requires nsys/ncu profiling on a 5090 to identify the
// actual stall: SMEM bank conflicts (probably reduced but not zero with
// 80B padding), register file pressure, mma issue rate per warp, or
// warp-scheduler underutilization. Each diagnosis suggests different
// fixes; without the profile data each ineffective micro-opt above
// rules one out. The work is solidly in "perf engineering with hardware
// profiler in hand" territory — multi-day investigation, not a session-
// sized task.

#include <cstdint>
#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 64
#define REDUCE_EVERY 2
#define JS 16
#define HC 64
#define GROUP_M 8
#define NUM_WARPS 4
#define THREADS 128
#define STAGES 3

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async_16(uint32_t dst_smem, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(dst_smem), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group_t() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__
void mma_s8s8s32(int32_t (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm(
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32"
        " {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__
void ldmatrix_x4(uint32_t (&r)[4], uint32_t smem) {
    asm("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem));
}

__device__ __forceinline__
void ldmatrix_x2_trans(uint32_t (&r)[2], uint32_t smem) {
    asm("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1]) : "r"(smem));
}

__device__ __forceinline__
void ldmatrix_x2(uint32_t (&r)[2], uint32_t smem) {
    asm("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1]) : "r"(smem));
}

template <int SS>
__device__ __forceinline__
void load_tile_to_smem(
    const int8_t* __restrict__ A_global, int row_origin, int stride_a,
    const int8_t* __restrict__ B_global, int col_origin, int stride_b,
    int k0,
    int8_t (*A_smem)[SS], int8_t (*B_smem)[SS],
    int tid)
{
    {
        int r = tid;
        const int8_t* src = A_global + (row_origin + r) * stride_a + k0;
        uint32_t dst = smem_addr(&A_smem[r][0]);
        cp_async_16(dst,       src);
        cp_async_16(dst + 16,  src + 16);
        cp_async_16(dst + 32,  src + 32);
        cp_async_16(dst + 48,  src + 48);
    }
    {
        int n = tid;
        const int8_t* src = B_global + (col_origin + n) * stride_b + k0;
        uint32_t dst = smem_addr(&B_smem[n][0]);
        cp_async_16(dst,       src);
        cp_async_16(dst + 16,  src + 16);
        cp_async_16(dst + 32,  src + 32);
        cp_async_16(dst + 48,  src + 48);
    }
}

extern "C" __global__ __launch_bounds__(THREADS, 1)
void _pearl_search_norotl_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    uint32_t* __restrict__ transcripts,
    int M, int N, int K,
    int stride_am, int stride_bn
) {
    const int pid = blockIdx.x;
    const int num_pid_m = M / BM;
    const int num_pid_n = N / BN;
    const int num_pid_in_group = GROUP_M * num_pid_n;
    const int group_id = pid / num_pid_in_group;
    const int first_pid_m = group_id * GROUP_M;
    const int group_size_m = min(num_pid_m - first_pid_m, GROUP_M);
    const int pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m);
    const int pid_n = (pid % num_pid_in_group) / group_size_m;

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;

    // Row stride padded from BK=64 to 80 bytes to break smem bank
    // conflicts on ldmatrix reads. With 64-byte rows, the per-lane row
    // addresses (T*64 for lanes 0..15) hit only 2 banks → 8-way
    // conflict. 80-byte rows give 8 unique banks → 2-way conflict.
    constexpr int SMEM_STRIDE = 80;
    __shared__ __align__(128) int8_t A_smem[STAGES][BM][SMEM_STRIDE];
    __shared__ __align__(128) int8_t B_smem[STAGES][BN][SMEM_STRIDE];

    int32_t acc[2][16][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 16; ++j)
            #pragma unroll
            for (int k = 0; k < 4; ++k) acc[i][j][k] = 0;

    const int row0 = (pid_m * BM) % M;
    const int col0 = (pid_n * BN) % N;

    const int num_k_iters = K / BK;
    const int tile_linear = pid_m * num_pid_n + pid_n;
    const int tile_base = tile_linear * HC * JS;
    int chk = 0;

    // Prefetch first STAGES-1 stages (so we always have STAGES-1 in flight
    // entering the loop body).
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < num_k_iters) {
            load_tile_to_smem(A, row0, stride_am, B, col0, stride_bn,
                              s * BK, A_smem[s], B_smem[s], tid);
            cp_async_commit();
        }
    }

    for (int k_iter = 0; k_iter < num_k_iters; ++k_iter) {
        int cur = k_iter % STAGES;

        if (k_iter > 0) {
            __syncthreads();
        }

        // Issue prefetch for k_iter + (STAGES-1) (the slot we just
        // released) and let it overlap with the mma below.
        const int pf_k = k_iter + (STAGES - 1);
        const int pf_stage = pf_k % STAGES;
        if (pf_k < num_k_iters) {
            load_tile_to_smem(A, row0, stride_am, B, col0, stride_bn,
                              pf_k * BK,
                              A_smem[pf_stage], B_smem[pf_stage], tid);
            cp_async_commit();
            // Keep at most STAGES-1 prefetches in flight (drain older).
            cp_async_wait_group_t<STAGES - 1>();
        } else {
            // Towards the tail: drain everything older than 'cur'.
            // STAGES-1-i in flight before, after wait_group<remaining-1>
            // we keep the remaining future ones in flight.
            cp_async_wait_group_t<0>();
        }
        __syncthreads();

        // Restructured for software pipelining:
        //   B operands don't depend on sm_M, so hoist B loads out of the
        //   sm_M loop. Prefetch ALL 8 B pairs (= 16 mma B operands) into
        //   registers BEFORE the inner mma loop. Inner mma loop has NO
        //   loads — compiler can issue mma back-to-back at peak rate.
        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int k_off = kk * 32;

            // Cache all 8 B pairs for this kk. Each pair = 4 u32 = 2 mma
            // B operands. Total: 32 u32 in registers per thread per kk.
            uint32_t b_pairs[8][4];
            #pragma unroll
            for (int sm_N_pair = 0; sm_N_pair < 8; ++sm_N_pair) {
                const int sm_N0 = sm_N_pair * 2;
                const int matrix_idx       = (lane >> 3) & 3;
                const int b_row_in_8       = lane & 7;
                const int n_offset_in_pair = (matrix_idx >> 1);
                const int b_col_block      = matrix_idx & 1;
                const int abs_n = (sm_N0 + n_offset_in_pair) * 8 + b_row_in_8;
                const int abs_k = k_off + b_col_block * 16;
                ldmatrix_x4(b_pairs[sm_N_pair],
                            smem_addr(&B_smem[cur][abs_n][abs_k]));
            }

            // For each sm_M, load A and run the inner 16-mma loop using
            // the cached B operands. NO loads inside the mma loop.
            #pragma unroll
            for (int sm_M = 0; sm_M < 2; ++sm_M) {
                uint32_t a_frag[4];
                {
                    const int a_row_in_16 = lane & 15;
                    const int a_col_off   = (lane >> 4) * 16;
                    const int abs_row = warp * 32 + sm_M * 16 + a_row_in_16;
                    const int abs_col = k_off + a_col_off;
                    ldmatrix_x4(a_frag, smem_addr(&A_smem[cur][abs_row][abs_col]));
                }

                #pragma unroll
                for (int sm_N_pair = 0; sm_N_pair < 8; ++sm_N_pair) {
                    const int sm_N0 = sm_N_pair * 2;
                    uint32_t b0[2] = { b_pairs[sm_N_pair][0],
                                       b_pairs[sm_N_pair][1] };
                    uint32_t b1[2] = { b_pairs[sm_N_pair][2],
                                       b_pairs[sm_N_pair][3] };
                    mma_s8s8s32(acc[sm_M][sm_N0    ], a_frag, b0);
                    mma_s8s8s32(acc[sm_M][sm_N0 + 1], a_frag, b1);
                }
            }
        }

        // ---- Checkpoint ----
        if ((k_iter + 1) % REDUCE_EVERY == 0) {
            uint32_t part[2][2] = {{0,0},{0,0}};
            #pragma unroll
            for (int sm_M = 0; sm_M < 2; ++sm_M) {
                #pragma unroll
                for (int sm_N = 0; sm_N < 16; ++sm_N) {
                    part[sm_M][0] ^= (uint32_t)acc[sm_M][sm_N][0]
                                  ^  (uint32_t)acc[sm_M][sm_N][1];
                    part[sm_M][1] ^= (uint32_t)acc[sm_M][sm_N][2]
                                  ^  (uint32_t)acc[sm_M][sm_N][3];
                }
            }
            #pragma unroll
            for (int sm_M = 0; sm_M < 2; ++sm_M) {
                #pragma unroll
                for (int rc = 0; rc < 2; ++rc) {
                    uint32_t v = part[sm_M][rc];
                    v ^= __shfl_xor_sync(0xFFFFFFFF, v, 1);
                    v ^= __shfl_xor_sync(0xFFFFFFFF, v, 2);
                    v ^= __shfl_xor_sync(0xFFFFFFFF, v, 4);
                    part[sm_M][rc] = v;
                }
            }
            if ((lane % 8) == 0) {
                int row_class_offset = lane / 8;
                #pragma unroll
                for (int sm_M = 0; sm_M < 2; ++sm_M) {
                    #pragma unroll
                    for (int rc = 0; rc < 2; ++rc) {
                        int cand_i = warp * 16 + sm_M * 8 + rc * 4 + row_class_offset;
                        transcripts[tile_base + cand_i * JS + chk] = part[sm_M][rc];
                    }
                }
            }
            chk += 1;
        }
    }
}
