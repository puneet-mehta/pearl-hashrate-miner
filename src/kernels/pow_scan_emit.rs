//! `pow_scan_and_emit` — find first set hit + emit verifier status header.
//!
//! Two-kernel sequence run at the end of the search:
//!
//! 1. [`PowScanEmit::launch_scan`] — `pow_scan_hits_kernel` scans `d_hit` and
//!    atomic-mins the first set index into `g_first_hit_idx` (sentinel
//!    `UINT32_MAX` written by the caller before launch).
//! 2. [`PowScanEmit::launch_emit`] — `pow_emit_header_kernel` writes the
//!    640-byte status header into the pinned host buffer iff a hit was
//!    found. The status byte at offset 0 is published LAST behind a
//!    `__threadfence_system` so a polling host sees a fully-consistent
//!    header.
//!
//! `HostSignalStatus`:
//! - `0` = `kSignalEmpty` (initial; no hit this iter)
//! - `1` = `kSignalTriggered` (hit found, header populated)
//!
//! Caller responsibility:
//! - `cuMemsetD8Async(g_first_hit_idx, 0xFF, 4, stream)` before the scan.
//! - `cuMemsetD8Async(pinned_header + 0, 0, 1, stream)` (or zero the whole
//!   buffer) before each iter so a stale `1` from a prior hit doesn't fool
//!   the polling callback.

use std::ffi::c_void;

use cudarc::driver::sys::{CUdeviceptr, CUfunction, CUstream};

use crate::driver::{launch_kernel, Module};
use crate::error::MinerError;
use crate::fatbin::symbols;

/// Cached kernel handles for the two-step sequence.
pub struct PowScanEmit {
    scan: CUfunction,
    emit: CUfunction,
}

impl PowScanEmit {
    pub fn new(module: &Module) -> Result<Self, MinerError> {
        Ok(Self {
            scan: module.get_function(symbols::POW_SCAN_HITS)?,
            emit: module.get_function(symbols::POW_EMIT_HEADER)?,
        })
    }

    /// Run the hit-scan kernel.
    ///
    /// `total` = `num_tile_m * num_tile_n * threads_per_tile`. Caller must have
    /// reset `g_first_hit_idx` to `0xFFFFFFFF` before this call.
    ///
    /// # Safety
    /// All pointers must be valid device allocations on the same context.
    pub unsafe fn launch_scan(
        &self,
        d_hit: CUdeviceptr,
        total: i32,
        g_first_hit_idx: CUdeviceptr,
        stream: CUstream,
    ) -> Result<(), MinerError> {
        let block_x: u32 = 256;
        let grid_x: u32 = ((total as u32) + block_x - 1) / block_x;
        let mut p_hit = d_hit;
        let mut p_total = total;
        let mut p_idx = g_first_hit_idx;
        let mut params: [*mut c_void; 3] = [
            &mut p_hit as *mut _ as *mut c_void,
            &mut p_total as *mut _ as *mut c_void,
            &mut p_idx as *mut _ as *mut c_void,
        ];
        launch_kernel(
            self.scan,
            (grid_x, 1, 1),
            (block_x, 1, 1),
            0,
            stream,
            &mut params,
        )
    }

    /// Run the emit kernel.
    ///
    /// `pinned_header` is a 640-byte CPU-pinned (UVA) host buffer that the
    /// host polls for `header[0] == 1`. `pinned_sync` may be `0` (null) to
    /// skip the optional sync write.
    ///
    /// # Safety
    /// `g_first_hit_idx` must point to a u32 device allocation populated by
    /// [`Self::launch_scan`]. `pow_target` is 8 u32 on device. `pinned_header`
    /// must be at least 640 bytes of UVA-mapped pinned host memory.
    pub unsafe fn launch_emit(
        &self,
        g_first_hit_idx: CUdeviceptr,
        pow_target: CUdeviceptr,
        pinned_header: CUdeviceptr, // UVA host pointer
        pinned_sync: CUdeviceptr,   // may be 0
        num_tile_m: i32,
        num_tile_n: i32,
        threads_per_tile: i32,
        m: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    ) -> Result<(), MinerError> {
        let mut p_idx = g_first_hit_idx;
        let mut p_target = pow_target;
        let mut p_hdr = pinned_header;
        let mut p_sync = pinned_sync;
        let mut p_ntm = num_tile_m;
        let mut p_ntn = num_tile_n;
        let mut p_tpt = threads_per_tile;
        let mut p_m = m;
        let mut p_n = n;
        let mut p_k = k;
        let mut params: [*mut c_void; 10] = [
            &mut p_idx as *mut _ as *mut c_void,
            &mut p_target as *mut _ as *mut c_void,
            &mut p_hdr as *mut _ as *mut c_void,
            &mut p_sync as *mut _ as *mut c_void,
            &mut p_ntm as *mut _ as *mut c_void,
            &mut p_ntn as *mut _ as *mut c_void,
            &mut p_tpt as *mut _ as *mut c_void,
            &mut p_m as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
        ];
        launch_kernel(self.emit, (1, 1, 1), (1, 1, 1), 0, stream, &mut params)
    }
}
