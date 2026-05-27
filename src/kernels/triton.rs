//! Triton-compiled kernel wrappers.
//!
//! Triton kernels are Python-first JIT-compiled artifacts. We extract
//! their CUDA cubins from `~/.triton/cache/` after a one-time JIT run
//! (see `the build script`) and embed them into the
//! Rust binary via `include_bytes!`. At runtime, `cuModuleLoadData`
//! turns the cubin into a CUmodule we can grab functions from.
//!
//! Two kernels:
//!
//! - `_noising_kernel` — `Out[i,j] = wrap_int8(X[i,j] + sum_r Y[i,r] * Z[j,r])`.
//!   Used to noise A → ApEA and B → BpEB. Replaces both the C++ noising
//!   kernel (which only hits ~5% int8 peak at K_inner=128) and is ~3×
//!   faster.
//!
//! - `_pearl_search_norotl_kernel` — norotl search kernel. Writes a
//!   `(num_tiles, HASH_CANDIDATES=64, JACKPOT_SIZE=16)` uint32
//!   transcript buffer. Each tile holds the (h=2, w=128) candidate
//!   hashes that `triton_postpass` then Blake3's + scan/emits.
//!
//! Both cubins are specialized for:
//!   - sm_89 (RTX 4090)
//!   - default shape: M=8192, N=32768, K=2048, R=128
//!   - norotl config: BM=BN=128, BK=64, W=4, S=3, HASH_CANDIDATES=64
//!
//! Argument layout (post-Triton-specialization — args with statically
//! known values like inner strides = 1 are dropped from the runtime
//! signature):
//!
//! `noising_kernel`:
//!   X_ptr, Y_ptr, Z_ptr, Out_ptr, M, N, K_inner,
//!   stride_xm, stride_ym, stride_zn, stride_om
//!
//! `pearl_search_norotl_kernel`:
//!   A_ptr, B_ptr, transcript_ptr, M, N, K, stride_am, stride_bn

use std::ffi::c_void;

use cudarc::driver::sys::{CUdeviceptr, CUfunction, CUstream};

use crate::driver::{launch_kernel, Module};
use crate::error::{cu_check, MinerError};
use crate::fatbin::symbols;

// =============================================================================
//   Embedded PTX (per-arch, runtime-selected)
// =============================================================================
//
// We ship one PTX blob per supported arch (3090=sm_86, 4090=sm_89,
// 5090=sm_120). At load time we query the device's compute capability
// and pick the matching blob; if no exact match, we fall back to the
// nearest lower arch (PTX is forward-compatible — sm_86 PTX runs on
// sm_89 / sm_120 via the driver JIT, just with slightly worse perf).
//
// PTX must be null-terminated when handed to `cuModuleLoadData`; helpers
// append the NUL.

const NOISING_PTX_SM86: &[u8] = include_bytes!("../../triton_kernels/sm86/noising_kernel.ptx");
const NOISING_PTX_SM89: &[u8] = include_bytes!("../../triton_kernels/sm89/noising_kernel.ptx");
const NOISING_PTX_SM120: &[u8] = include_bytes!("../../triton_kernels/sm120/noising_kernel.ptx");

const SEARCH_NOROTL_PTX_SM86: &[u8] =
    include_bytes!("../../triton_kernels/sm86/pearl_search_norotl_kernel.ptx");
const SEARCH_NOROTL_PTX_SM89: &[u8] =
    include_bytes!("../../triton_kernels/sm89/pearl_search_norotl_kernel.ptx");
const SEARCH_NOROTL_PTX_SM120: &[u8] =
    include_bytes!("../../triton_kernels/sm120/pearl_search_norotl_kernel.ptx");

/// Pick the noising PTX blob for `(cc_major, cc_minor)`. Exact match
/// preferred; otherwise the highest available arch <= the device.
pub fn noising_ptx(cc_major: i32, cc_minor: i32) -> Vec<u8> {
    let blob = pick_ptx(
        cc_major,
        cc_minor,
        NOISING_PTX_SM86,
        NOISING_PTX_SM89,
        NOISING_PTX_SM120,
    );
    null_terminate(blob)
}

pub fn search_norotl_ptx(cc_major: i32, cc_minor: i32) -> Vec<u8> {
    let blob = pick_ptx(
        cc_major,
        cc_minor,
        SEARCH_NOROTL_PTX_SM86,
        SEARCH_NOROTL_PTX_SM89,
        SEARCH_NOROTL_PTX_SM120,
    );
    null_terminate(blob)
}

fn pick_ptx(
    cc_major: i32,
    cc_minor: i32,
    sm86: &'static [u8],
    sm89: &'static [u8],
    sm120: &'static [u8],
) -> &'static [u8] {
    let cc = cc_major * 10 + cc_minor; // 86, 89, 120 etc.
    if cc >= 120 {
        if !sm120.is_empty() {
            return sm120;
        }
        if !sm89.is_empty() {
            return sm89;
        }
        sm86
    } else if cc >= 89 {
        if !sm89.is_empty() {
            return sm89;
        }
        sm86
    } else {
        sm86
    }
}

fn null_terminate(blob: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(blob.len() + 1);
    v.extend_from_slice(blob);
    v.push(0);
    v
}

// Legacy cubin re-exports for callers still on the cubin path
// (`triton-smoke` bin). (transitional helper);
// PTX on every arch.
pub const NOISING_CUBIN_SM89: &[u8] =
    include_bytes!("../../triton_kernels/sm89/noising_kernel.cubin");
pub const SEARCH_NOROTL_CUBIN_SM89: &[u8] =
    include_bytes!("../../triton_kernels/sm89/pearl_search_norotl_kernel.cubin");

// =============================================================================
//   Kernel constants (must match the JIT-time constexpr values)
// =============================================================================

pub const NOISING_KERNEL_NAME: &str = "_noising_kernel";
pub const SEARCH_NOROTL_KERNEL_NAME: &str = "_pearl_search_norotl_kernel";

// Block sizes (= num_warps * 32). num_warps=4 for both.
pub const NOISING_BLOCK_X: u32 = 128;
pub const SEARCH_NOROTL_BLOCK_X: u32 = 128;

// Dynamic shared-memory bytes (from kernel.json `shared` field).
pub const NOISING_SHARED_BYTES: u32 = 49_152;
pub const SEARCH_NOROTL_SHARED_BYTES: u32 = 33_792;

// norotl tile / hash constants. Must match Python triton_search_no_rotl.py
// `DEFAULT_BLOCK_M/N/K`, `HASH_CANDIDATES`, `JACKPOT_SIZE`.
pub const BLOCK_M: usize = 128;
pub const BLOCK_N: usize = 128;
pub const HASH_CANDIDATES: usize = 64;
pub const JACKPOT_SIZE: usize = 16;
pub const NOISING_GROUP_M: usize = 8;
pub const SEARCH_GROUP_M: usize = 8;
// Noising tile shape (different from search!): from triton_noising.py defaults.
pub const NOISING_BLOCK_M: usize = 128;
pub const NOISING_BLOCK_N: usize = 64;

// =============================================================================
//   Wrappers
// =============================================================================

pub struct TritonNoising {
    func: CUfunction,
}

impl TritonNoising {
    /// Load the noising kernel from an already-loaded cubin module.
    pub fn new(module: &Module) -> Result<Self, MinerError> {
        let func = module.get_function(NOISING_KERNEL_NAME)?;
        // Opt into 48 KB dynamic shared mem (the default cap on sm_89 is
        // 48 KB; we need 48 KB exactly, so this should be no-op, but be
        // explicit).
        unsafe {
            cu_check(
                cudarc::driver::sys::cuFuncSetAttribute(
                    func,
                    cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    NOISING_SHARED_BYTES as i32,
                ),
                "cuFuncSetAttribute(noising max smem)",
            )?;
        }
        Ok(Self { func })
    }

    /// `Out = wrap_int8(X + Y @ Z^T)`. Row-major contiguous tensors.
    ///
    /// # Safety
    /// All pointers must be valid device allocations on the same context
    /// as `module`.
    pub unsafe fn launch(
        &self,
        m: i32,
        n: i32,
        k_inner: i32,
        x: CUdeviceptr,
        y: CUdeviceptr,
        z: CUdeviceptr,
        out: CUdeviceptr,
        stream: CUstream,
    ) -> Result<(), MinerError> {
        // Strides for row-major contiguous: X is (M, N) → stride_xm=N.
        //                                    Y is (M, K) → stride_ym=K.
        //                                    Z is (N, K) → stride_zn=K.
        //                                    Out is (M, N) → stride_om=N.
        // Triton specialized away stride_xn/yk/zk/on (=1).
        let mut p_x = x;
        let mut p_y = y;
        let mut p_z = z;
        let mut p_out = out;
        let mut p_m = m;
        let mut p_n = n;
        let mut p_k = k_inner;
        let mut p_strxm = n;
        let mut p_strym = k_inner;
        let mut p_strzn = k_inner;
        let mut p_strom = n;
        let mut params: [*mut c_void; 11] = [
            &mut p_x as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut p_z as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut p_m as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut p_strxm as *mut _ as *mut c_void,
            &mut p_strym as *mut _ as *mut c_void,
            &mut p_strzn as *mut _ as *mut c_void,
            &mut p_strom as *mut _ as *mut c_void,
        ];

        // Grid: ceil(M/BM) * ceil(N/BN) tiles in 1D.
        let num_pid_m = (m as u32 + NOISING_BLOCK_M as u32 - 1) / NOISING_BLOCK_M as u32;
        let num_pid_n = (n as u32 + NOISING_BLOCK_N as u32 - 1) / NOISING_BLOCK_N as u32;
        let grid_x = num_pid_m * num_pid_n;

        launch_kernel(
            self.func,
            (grid_x, 1, 1),
            (NOISING_BLOCK_X, 1, 1),
            NOISING_SHARED_BYTES,
            stream,
            &mut params,
        )
    }
}

pub struct TritonSearchNorotl {
    func: CUfunction,
}

impl TritonSearchNorotl {
    pub fn new(module: &Module) -> Result<Self, MinerError> {
        let func = module.get_function(SEARCH_NOROTL_KERNEL_NAME)?;
        unsafe {
            cu_check(
                cudarc::driver::sys::cuFuncSetAttribute(
                    func,
                    cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    SEARCH_NOROTL_SHARED_BYTES as i32,
                ),
                "cuFuncSetAttribute(search_norotl max smem)",
            )?;
        }
        Ok(Self { func })
    }

    /// `transcripts_out: (num_tiles, HASH_CANDIDATES=64, JACKPOT_SIZE=16)` u32.
    /// Caller must pre-zero the transcript buffer.
    ///
    /// `ApEA: (M, K) int8 row-major`.
    /// `BpEB: (N, K) int8 row-major` — we pass `B` directly; the kernel
    /// expects the equivalent of `B.t()`, i.e. (K, N) col-major view,
    /// which has the same byte layout when B is (N, K) row-major
    /// — Triton uses stride_bk=1 (specialized) and stride_bn=K.
    ///
    /// # Safety
    /// All pointers must be valid device allocations on the same context.
    pub unsafe fn launch(
        &self,
        m: i32,
        n: i32,
        k: i32,
        ap_ea: CUdeviceptr,
        bp_eb: CUdeviceptr,
        transcripts: CUdeviceptr,
        stream: CUstream,
    ) -> Result<(), MinerError> {
        // A is (M, K) → stride_am = K, stride_ak = 1 (specialized).
        // B^T view is (K, N) col-major → equivalent to B (N, K) row-major,
        //   so stride_bn = K, stride_bk = 1 (specialized).
        let mut p_a = ap_ea;
        let mut p_b = bp_eb;
        let mut p_t = transcripts;
        let mut p_m = m;
        let mut p_n = n;
        let mut p_k = k;
        let mut p_stram = k;
        let mut p_strbn = k;
        let mut params: [*mut c_void; 8] = [
            &mut p_a as *mut _ as *mut c_void,
            &mut p_b as *mut _ as *mut c_void,
            &mut p_t as *mut _ as *mut c_void,
            &mut p_m as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut p_stram as *mut _ as *mut c_void,
            &mut p_strbn as *mut _ as *mut c_void,
        ];

        let num_pid_m = (m as u32 + BLOCK_M as u32 - 1) / BLOCK_M as u32;
        let num_pid_n = (n as u32 + BLOCK_N as u32 - 1) / BLOCK_N as u32;
        let grid_x = num_pid_m * num_pid_n;

        launch_kernel(
            self.func,
            (grid_x, 1, 1),
            (SEARCH_NOROTL_BLOCK_X, 1, 1),
            SEARCH_NOROTL_SHARED_BYTES,
            stream,
            &mut params,
        )
    }
}

// =============================================================================
//   triton_postpass — Blake3 compare + scan + emit (h=2, w=128) emit header
// =============================================================================
//
// Lives in the production pearl_gemm fatbin (NOT in the Triton cubin).
// Two kernels:
//   1. blake3_compare_kernel — per-candidate Blake3(transcript || pow_key)
//      then compare hash u256 LE vs pow_target. Writes per-candidate hash
//      and hit byte.
//   2. pow_emit_header_triton_paired_kernel — single-thread emit that finds
//      the first hit (via the existing pow_scan_hits + atomicMin) and
//      writes the host signal header with (h=2, w=128) interleaved layout.

pub struct TritonPostpass {
    blake3_compare: CUfunction,
    paired_emit: CUfunction,
    /// Reused for the scan step (= same kernel the C++ search path uses).
    scan_hits: CUfunction,
}

impl TritonPostpass {
    pub fn new(module: &Module) -> Result<Self, MinerError> {
        Ok(Self {
            blake3_compare: module.get_function(symbols::TRITON_BLAKE3_COMPARE)?,
            paired_emit: module.get_function(symbols::TRITON_PAIRED_EMIT_HEADER)?,
            scan_hits: module.get_function(symbols::POW_SCAN_HITS)?,
        })
    }

    /// Stage 1: Blake3 + compare over `total_candidates` transcripts.
    ///
    /// # Safety
    /// All pointers must be valid device allocations on the same context.
    pub unsafe fn launch_blake3_compare(
        &self,
        transcripts: CUdeviceptr,
        pow_key: CUdeviceptr,
        pow_target: CUdeviceptr,
        d_hash: CUdeviceptr,
        d_hit: CUdeviceptr,
        total_candidates: i32,
        stream: CUstream,
    ) -> Result<(), MinerError> {
        let mut p_tr = transcripts;
        let mut p_key = pow_key;
        let mut p_tgt = pow_target;
        let mut p_hash = d_hash;
        let mut p_hit = d_hit;
        let mut p_n = total_candidates;
        let mut params: [*mut c_void; 6] = [
            &mut p_tr as *mut _ as *mut c_void,
            &mut p_key as *mut _ as *mut c_void,
            &mut p_tgt as *mut _ as *mut c_void,
            &mut p_hash as *mut _ as *mut c_void,
            &mut p_hit as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
        ];
        let block: u32 = 256;
        let grid: u32 = ((total_candidates as u32) + block - 1) / block;
        launch_kernel(
            self.blake3_compare,
            (grid, 1, 1),
            (block, 1, 1),
            0,
            stream,
            &mut params,
        )
    }

    /// Stage 2: scan d_hit for the first set byte, atomicMin into
    /// pow_workspace_scan (caller pre-resets it to 0xFFFFFFFF).
    ///
    /// # Safety: same as above.
    pub unsafe fn launch_scan(
        &self,
        d_hit: CUdeviceptr,
        total: i32,
        g_first_hit_idx: CUdeviceptr,
        stream: CUstream,
    ) -> Result<(), MinerError> {
        let mut p_hit = d_hit;
        let mut p_n = total;
        let mut p_first = g_first_hit_idx;
        let mut params: [*mut c_void; 3] = [
            &mut p_hit as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
            &mut p_first as *mut _ as *mut c_void,
        ];
        let block: u32 = 256;
        let grid: u32 = ((total as u32) + block - 1) / block;
        launch_kernel(
            self.scan_hits,
            (grid, 1, 1),
            (block, 1, 1),
            0,
            stream,
            &mut params,
        )
    }

    /// Stage 3: emit the host signal header (h=2, w=128 pattern).
    /// Single-thread launch.
    ///
    /// # Safety: same as above.
    pub unsafe fn launch_emit(
        &self,
        g_first_hit_idx: CUdeviceptr,
        pow_target: CUdeviceptr,
        pinned_header: CUdeviceptr, // UVA-mapped pinned host
        num_tile_m: i32,
        num_tile_n: i32,
        hash_candidates: i32,
        m: i32,
        n: i32,
        k: i32,
        block_m: i32,
        block_n: i32,
        block_k: i32,
        stream: CUstream,
    ) -> Result<(), MinerError> {
        let mut p_first = g_first_hit_idx;
        let mut p_tgt = pow_target;
        let mut p_header = pinned_header;
        let mut p_sync: CUdeviceptr = 0; // nullptr
        let mut p_ntm = num_tile_m;
        let mut p_ntn = num_tile_n;
        let mut p_hc = hash_candidates;
        let mut p_m = m;
        let mut p_n = n;
        let mut p_k = k;
        let mut p_bm = block_m;
        let mut p_bn = block_n;
        let mut p_bk = block_k;
        let mut params: [*mut c_void; 13] = [
            &mut p_first as *mut _ as *mut c_void,
            &mut p_tgt as *mut _ as *mut c_void,
            &mut p_header as *mut _ as *mut c_void,
            &mut p_sync as *mut _ as *mut c_void,
            &mut p_ntm as *mut _ as *mut c_void,
            &mut p_ntn as *mut _ as *mut c_void,
            &mut p_hc as *mut _ as *mut c_void,
            &mut p_m as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut p_bm as *mut _ as *mut c_void,
            &mut p_bn as *mut _ as *mut c_void,
            &mut p_bk as *mut _ as *mut c_void,
        ];
        launch_kernel(
            self.paired_emit,
            (1, 1, 1),
            (1, 1, 1),
            0,
            stream,
            &mut params,
        )
    }
}
