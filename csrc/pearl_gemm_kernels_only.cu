// Kernel translation unit compiled into pearl_gemm.fatbin.
//
// Contents:
//   1. extern-C kernel shims that the Rust crate looks up by name in
//      src/fatbin.rs::symbols.
//   2. Explicit template instantiations forcing code-gen for the kernel
//      ranks the miner uses (noising_smem, search_perthread_smem_pipelined,
//      blake3_compare, paired emit_header).
//
// Build via csrc/build_fatbin.sh — emits SASS for sm_86/89/120 plus
// sm_86 PTX as the JIT fallback.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "pearl_gemm/blake3_sm80.cuh"
#include "pearl_gemm/noise_generation_sm80.cuh"
#include "pearl_gemm/noising_sm80.cuh"
#include "pearl_gemm/noising_smem_sm80.cuh"
#include "pearl_gemm/merkle_sm80.cuh"
#include "pearl_gemm/merkle_combine_sm80.cuh"
#include "pearl_gemm/pearl_gemm_search_perthread_smem_pipelined_sm80.cuh"
#include "pearl_gemm/pow_scan_emit_sm80.cuh"
#include "pearl_gemm/pearl_blake3_compare_sm80.cuh"

#include "extern_c_shims.inc"

// =============================================================================
//   Template instantiations — force code-gen for the search kernel ranks
//   we need. Headers declare the template; without explicit instantiation
//   nvcc won't emit code for any rank.
// =============================================================================

template __global__ void
    pearl::sm80::search_perthread_smem_pipelined
        ::pearl_gemm_search_perthread_smem_pipelined_kernel<128>(
            int, int, int,
            const int8_t*, const int8_t*,
            const uint32_t*, const uint32_t*,
            uint32_t*, uint8_t*, uint32_t*);

template __global__ void
    pearl::sm80::search_perthread_smem_pipelined
        ::pearl_gemm_search_perthread_smem_pipelined_kernel<64>(
            int, int, int,
            const int8_t*, const int8_t*,
            const uint32_t*, const uint32_t*,
            uint32_t*, uint8_t*, uint32_t*);
