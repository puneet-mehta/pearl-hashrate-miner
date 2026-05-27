//! `pearl_gemm_search_perthread_smem_pipelined_kernel<R>` — the hot inner loop.
//!
//! Templated on R (production: 128). Reads ApEA / BpEB int8 tensors, performs
//! the matmul + Blake3 hashing + hit-bit scan in a single fused kernel.
//! Writes `hash_per_tile_thread` (u32, num_tiles × 256 × 8), `hit_per_tile_thread`
//! (u8, num_tiles × 256), and optionally `transcript_per_tile_thread` (u32).
//!
//! ## Important: 96 KB dynamic shared memory
//!
//! The pipelined search kernel uses **3 stages × (TILE_M=128 × CTA_BK=128) × 2
//! = 96 KB** of dynamic shared memory per CTA. This exceeds sm_80's default
//! 48 KB limit, so we MUST opt in via `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE
//! _BYTES, 96 KB)` once per kernel handle before the first launch.
//!
//! ## Bit-exact testing
//!
//! Not done here. The kernel's output depends on correctly-noised `ApEA` /
//! `BpEB`, which requires running the full per-iter pipeline (random_int8 →
//! tensor_hash → commitment_hash → noise_gen → noising_add_gemm → search).
//! Validation deferred until MinerBufs and the per-iter loop are in place.
//! This module just exposes the dispatch.

use std::ffi::c_void;

use cudarc::driver::sys as cu;
use cudarc::driver::sys::{CUdeviceptr, CUfunction, CUstream};

use crate::driver::{launch_kernel, Module};
use crate::error::{cu_check, MinerError};
use crate::fatbin::symbols;

pub const TILE_M: i32 = 128;
pub const TILE_N: i32 = 128;
pub const CTA_THREADS: u32 = 256;
pub const NUM_STAGES: i32 = 3;
pub const CTA_BK: i32 = 128;

/// Dynamic shared memory per CTA: 3 stages × TILE_M × CTA_BK (sA) +
/// 3 stages × TILE_N × CTA_BK (sB) = 96 KB.
pub const SMEM_BYTES: u32 = (NUM_STAGES * TILE_M * CTA_BK + NUM_STAGES * TILE_N * CTA_BK) as u32;

pub struct Search {
    kernel_r128: CUfunction,
}

impl Search {
    /// Construct + set the 96 KB shared-memory attribute on the kernel handle.
    pub fn new(module: &Module) -> Result<Self, MinerError> {
        let kernel_r128 = module.get_function(symbols::SEARCH_PIPELINED_R128)?;
        unsafe {
            cu_check(
                cu::cuFuncSetAttribute(
                    kernel_r128,
                    cu::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    SMEM_BYTES as i32,
                ),
                "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
            )?;
        }
        Ok(Self { kernel_r128 })
    }

    /// Launch the R=128 search kernel.
    ///
    /// Workspace shapes:
    /// - `ApEA`: (M, K) int8 row-major
    /// - `BpEB`: (N, K) int8 row-major (accessed as B^T inside)
    /// - `pow_key`: 8 u32 (32 bytes) — derived from commit_A_pool[slot]
    /// - `pow_target`: 8 u32 (32 bytes) — adjusted target
    /// - `hash_per_tile_thread`: num_tiles × 256 × 8 u32
    /// - `hit_per_tile_thread`: num_tiles × 256 u8
    /// - `transcript_per_tile_thread`: optional, may be 0 (null)
    ///
    /// `num_tiles = (M / TILE_M) * (N / TILE_N)`.
    ///
    /// # Safety
    /// All device pointers must be valid on the same context as the loaded
    /// module. M % 128 == 0 and N % 128 == 0 required.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_r128(
        &self,
        m: i32,
        n: i32,
        k: i32,
        ap_ea: CUdeviceptr,
        bp_eb: CUdeviceptr,
        pow_key: CUdeviceptr,
        pow_target: CUdeviceptr,
        hash_per_tile_thread: CUdeviceptr,
        hit_per_tile_thread: CUdeviceptr,
        transcript_per_tile_thread: CUdeviceptr, // may be 0
        stream: CUstream,
    ) -> Result<(), MinerError> {
        assert!(
            m % TILE_M == 0 && n % TILE_N == 0,
            "search requires M%128==0 && N%128==0; got M={} N={}",
            m,
            n
        );
        let grid_x: u32 = (n / TILE_N) as u32;
        let grid_y: u32 = (m / TILE_M) as u32;

        let mut p_m = m;
        let mut p_n = n;
        let mut p_k = k;
        let mut p_ap = ap_ea;
        let mut p_bp = bp_eb;
        let mut p_pk = pow_key;
        let mut p_pt = pow_target;
        let mut p_hash = hash_per_tile_thread;
        let mut p_hit = hit_per_tile_thread;
        let mut p_trans = transcript_per_tile_thread;
        let mut params: [*mut c_void; 10] = [
            &mut p_m as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut p_ap as *mut _ as *mut c_void,
            &mut p_bp as *mut _ as *mut c_void,
            &mut p_pk as *mut _ as *mut c_void,
            &mut p_pt as *mut _ as *mut c_void,
            &mut p_hash as *mut _ as *mut c_void,
            &mut p_hit as *mut _ as *mut c_void,
            &mut p_trans as *mut _ as *mut c_void,
        ];
        launch_kernel(
            self.kernel_r128,
            (grid_x, grid_y, 1),
            (CTA_THREADS, 1, 1),
            SMEM_BYTES,
            stream,
            &mut params,
        )
    }
}
