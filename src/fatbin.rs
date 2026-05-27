//! Locate + load the pearl-gemm fatbin.
//!
//! The fatbin is the `.nv_fatbin` section embedded inside
//! `pearl_gemm_sm80/_C.cpython-*.so`. Two ways callers can supply it:
//!
//! 1. Read straight from the trim image's `.so` via `objcopy` at build / install
//!    time and ship a standalone `fatbin.bin` next to the Rust binary.
//! 2. (Future) Compile-time embed via `include_bytes!` so the binary is fully
//!    self-contained — what we may do for the slim image.

use std::path::Path;

use crate::driver::Module;
use crate::error::MinerError;

/// Default in-image install path of pearl-gemm-sm80 (standard wheel layout).
pub const DEFAULT_SO_PATH: &str =
    "/opt/venv/lib/python3.12/site-packages/pearl_gemm_sm80/_C.cpython-312-x86_64-linux-gnu.so";

/// Load a pre-extracted fatbin blob from disk.
pub fn load_fatbin_file(path: impl AsRef<Path>) -> Result<Module, MinerError> {
    let blob = std::fs::read(path)?;
    Module::load_fatbin(&blob)
}

/// Names of the extern-C kernel shims added in pearl_gemm_sm80_torch.cu
/// (the "pearl_*" prefix). New shims should be added here as each kernel
/// is ported in subsequent sessions.
pub mod symbols {
    pub const COMMITMENT_HASH: &str = "pearl_commitment_hash_kernel";

    // Merkle / tensor_hash trio:
    pub const CHUNK_CV_ROOT: &str = "pearl_chunk_cv_kernel_root";
    pub const CHUNK_CV_NOROOT: &str = "pearl_chunk_cv_kernel_noroot";
    pub const MERKLE_LAYER: &str = "pearl_merkle_layer_kernel";

    // Noise generation (R=128 production rank):
    pub const NOISE_GEN_DENSE_INT8_R128: &str = "pearl_noise_gen_dense_int8_R128";
    pub const NOISE_GEN_DENSE_FP16_R128: &str = "pearl_noise_gen_dense_fp16_R128";
    pub const NOISE_GEN_SPARSE_R128: &str = "pearl_noise_gen_sparse_R128";
    pub const TRANSPOSE_KR: &str = "pearl_transpose_kr_kernel";

    // PoW scan + emit:
    pub const POW_SCAN_HITS: &str = "pearl_pow_scan_hits_kernel";
    pub const POW_EMIT_HEADER: &str = "pearl_pow_emit_header_kernel";

    // Per-iter random A fill (replaces torch's curand-backed random_):
    pub const RANDOM_INT8_SEEDED: &str = "pearl_random_int8_seeded_kernel";

    // Noisy-GEMM noising path (Itanium-mangled C++ names — kernels are at
    // file scope in their .cuh headers, so symbols are stable across rebuilds).
    // Extracted via `cuobjdump --dump-elf-symbols` against the reference image.
    pub const NOISING_ADD_GEMM_INT8_SMEM: &str =
        "_ZN5pearl4sm8012noising_smem35gemm_int8_add_x_to_int8_smem_kernelEiiiPKaS3_S3_Pa";
    pub const NOISING_GEMM_INT8_INT32_SMEM: &str =
        "_ZN5pearl4sm8012noising_smem27gemm_int8_int32_smem_kernelEiiiPKaS3_Pi";
    pub const NOISING_INT32_TO_FP16_SCALED: &str =
        "_ZN5pearl4sm807noising27int32_to_fp16_scaled_kernelEiPKiP6__halfi";

    // Search kernel — template instantiations for R=64 and R=128. Production
    // miner uses R=128 (`MinerSettings.noise_rank`).
    pub const SEARCH_PIPELINED_R128: &str =
        "_ZN5pearl4sm8031search_perthread_smem_pipelined49pearl_gemm_search_perthread_smem_pipelined_kernelILi128EEEviiiPKaS4_PKjS6_PjPhS7_";
    pub const SEARCH_PIPELINED_R64: &str =
        "_ZN5pearl4sm8031search_perthread_smem_pipelined49pearl_gemm_search_perthread_smem_pipelined_kernelILi64EEEviiiPKaS4_PKjS6_PjPhS7_";

    // Triton-postpass: Blake3+compare on per-thread transcripts, then the
    // triton-paired emit kernel that writes the host signal header with
    // (h=2, w=128) pattern.
    pub const TRITON_BLAKE3_COMPARE: &str =
        "_ZN5pearl4sm8015triton_postpass21blake3_compare_kernelEPKjS3_S3_PjPhi";
    pub const TRITON_PAIRED_EMIT_HEADER: &str =
        "_ZN5pearl4sm8013pow_scan_emit36pow_emit_header_triton_paired_kernelEPKjS3_PhS4_iiiiiiiii";
}
