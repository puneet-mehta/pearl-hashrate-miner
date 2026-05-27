//! `commitment_hash_from_merkle_roots` — Blake3 commitment chain.
//!
//! Math (mirrors `pearl_gemm_sm80_torch.cu:489-498`):
//! ```text
//!   B_commit = Blake3(IV, key || B_root, CHUNK_START|CHUNK_END|ROOT)
//!   A_commit = Blake3(IV, B_commit || A_root, CHUNK_START|CHUNK_END|ROOT)
//! ```
//!
//! All inputs are 32-byte tensors. Launched on a single thread on a single
//! block — the work is 16 Blake3 round operations total, fits comfortably.

use std::ffi::c_void;

use cudarc::driver::sys::{CUdeviceptr, CUfunction, CUstream};

use crate::driver::{launch_kernel, Module};
use crate::error::MinerError;
use crate::fatbin::symbols;

/// Cached kernel handle. One-time lookup per process; cheap thereafter.
pub struct CommitmentHash {
    kernel: CUfunction,
}

impl CommitmentHash {
    pub fn new(module: &Module) -> Result<Self, MinerError> {
        Ok(Self {
            kernel: module.get_function(symbols::COMMITMENT_HASH)?,
        })
    }

    /// Run the kernel.
    ///
    /// Buffers `a_root`, `b_root`, `key` are read; `a_commit`, `b_commit` are
    /// written. All must be 32-byte device allocations.
    ///
    /// # Safety
    /// All `CUdeviceptr` arguments must point to valid 32-byte device
    /// allocations on the same context as `module` was loaded into.
    pub unsafe fn launch(
        &self,
        a_root: CUdeviceptr,
        b_root: CUdeviceptr,
        key: CUdeviceptr,
        a_commit: CUdeviceptr,
        b_commit: CUdeviceptr,
        stream: CUstream,
    ) -> Result<(), MinerError> {
        // Stack copies so `&mut` references stay valid for the param array.
        let mut p_a_root = a_root;
        let mut p_b_root = b_root;
        let mut p_key = key;
        let mut p_a_commit = a_commit;
        let mut p_b_commit = b_commit;
        let mut params: [*mut c_void; 5] = [
            &mut p_a_root as *mut _ as *mut c_void,
            &mut p_b_root as *mut _ as *mut c_void,
            &mut p_key as *mut _ as *mut c_void,
            &mut p_a_commit as *mut _ as *mut c_void,
            &mut p_b_commit as *mut _ as *mut c_void,
        ];
        launch_kernel(self.kernel, (1, 1, 1), (1, 1, 1), 0, stream, &mut params)
    }
}

/// CPU-side reference (uses the standard `blake3` crate). Bit-exact with the
/// device kernel by [[project-sm80-blake3-quirk]] in memory.
pub fn reference(a_root: &[u8; 32], b_root: &[u8; 32], key: &[u8; 32]) -> ([u8; 32], [u8; 32]) {
    let mut concat_b = [0u8; 64];
    concat_b[..32].copy_from_slice(key);
    concat_b[32..].copy_from_slice(b_root);
    let b_commit = *blake3::hash(&concat_b).as_bytes();

    let mut concat_a = [0u8; 64];
    concat_a[..32].copy_from_slice(&b_commit);
    concat_a[32..].copy_from_slice(a_root);
    let a_commit = *blake3::hash(&concat_a).as_bytes();

    (a_commit, b_commit)
}
