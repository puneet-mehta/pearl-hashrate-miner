//! `random_int8_seeded` — per-iter random int8 fill (replaces `A.random_`).
//!
//! Deterministic per (seed, iter_idx). Output is int8 in `[-63, 63]` inclusive
//! (127 values; matches torch's `random_(-63, 64)` semantics).
//!
//! Algorithm (per 32-byte output chunk):
//! ```text
//!   message_u32[0]    = iter_idx low
//!   message_u32[1]    = iter_idx high
//!   message_u32[2]    = chunk_idx
//!   message_u32[3..8] = 0
//!   message_u32[8..16] = seed (8 u32)
//!   cv = blake3 IV
//!   compress(message, cv, t=0, block_len=64, CHUNK_START|CHUNK_END|ROOT)
//!   for byte in cv_bytes: out_byte = (uint8(byte) % 127) - 63
//! ```
//!
//! CPU reference uses `blake3::hash(message_64_bytes)` — same primitive as
//! `blake3::keyed_hash` but with the default IV (no KEYED_HASH flag).

use std::ffi::c_void;

use cudarc::driver::sys::{CUdeviceptr, CUfunction, CUstream};

use crate::driver::{launch_kernel, Module};
use crate::error::MinerError;
use crate::fatbin::symbols;

pub struct RandomInt8 {
    kernel: CUfunction,
}

impl RandomInt8 {
    pub const BLOCK_X: u32 = 128;

    pub fn new(module: &Module) -> Result<Self, MinerError> {
        Ok(Self {
            kernel: module.get_function(symbols::RANDOM_INT8_SEEDED)?,
        })
    }

    /// Underlying CUfunction — exposed so callers can identify this kernel's
    /// node inside a captured graph (for `cuGraphExecKernelNodeSetParams`).
    pub fn func(&self) -> CUfunction {
        self.kernel
    }

    /// Grid x for a launch over `total_bytes`. Mirrors the math in `launch`,
    /// kept in sync so graph-node mutators can rebuild params identically.
    pub fn grid_x(total_bytes: i32) -> u32 {
        let num_chunks = total_bytes / 32;
        ((num_chunks as u32) + Self::BLOCK_X - 1) / Self::BLOCK_X
    }

    /// Fill `out` with `total_bytes` int8 values derived deterministically
    /// from `(seed, iter_idx)`.
    ///
    /// `total_bytes` must be a multiple of 32 (production A is 8 MB which is
    /// divisible). `seed` must point to 32 bytes (8 u32) on device.
    ///
    /// # Safety
    /// `out` must point to a `total_bytes`-byte device allocation. `seed` must
    /// be 32 bytes on the same context. `total_bytes % 32 == 0`.
    pub unsafe fn launch(
        &self,
        total_bytes: i32,
        seed: CUdeviceptr,
        iter_idx: u64,
        out: CUdeviceptr,
        stream: CUstream,
    ) -> Result<(), MinerError> {
        assert!(
            total_bytes > 0 && (total_bytes % 32) == 0,
            "total_bytes must be a positive multiple of 32"
        );
        let block_x: u32 = Self::BLOCK_X;
        let grid_x: u32 = Self::grid_x(total_bytes);

        let mut p_total = total_bytes;
        let mut p_seed = seed;
        let mut p_iter = iter_idx;
        let mut p_out = out;
        let mut params: [*mut c_void; 4] = [
            &mut p_total as *mut _ as *mut c_void,
            &mut p_seed as *mut _ as *mut c_void,
            &mut p_iter as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
        ];
        launch_kernel(
            self.kernel,
            (grid_x, 1, 1),
            (block_x, 1, 1),
            0,
            stream,
            &mut params,
        )
    }
}

/// CPU reference. Returns a `Vec<i8>` of length `total_bytes`.
pub fn reference(total_bytes: usize, seed: &[u8; 32], iter_idx: u64) -> Vec<i8> {
    assert!(total_bytes > 0 && (total_bytes % 32) == 0);
    let num_chunks = total_bytes / 32;
    let mut out = vec![0i8; total_bytes];
    for chunk in 0..num_chunks {
        // Build 64-byte message.
        let mut msg = [0u8; 64];
        msg[0..4].copy_from_slice(&((iter_idx & 0xFFFFFFFF) as u32).to_le_bytes());
        msg[4..8].copy_from_slice(&((iter_idx >> 32) as u32).to_le_bytes());
        msg[8..12].copy_from_slice(&(chunk as u32).to_le_bytes());
        // msg[12..32] stays zero
        msg[32..64].copy_from_slice(seed);

        // blake3::hash with no keyed_hash flag matches the kernel's compress
        // (IV as starting CV, single-block CHUNK_START|CHUNK_END|ROOT).
        let cv = *blake3::hash(&msg).as_bytes();

        let out_chunk = &mut out[chunk * 32..(chunk + 1) * 32];
        for (i, b) in cv.iter().enumerate() {
            let v = (*b as u32 % 127u32) as i32 - 63;
            out_chunk[i] = v as i8;
        }
    }
    out
}
