//! `tensor_hash` — keyed Blake3 Merkle root of a 1D byte buffer on GPU.
//!
//! Equivalent to `pearl_gemm.tensor_hash(data, key, out, roots)`. Constraints:
//! `data.len()` must be a positive multiple of `CHUNK_LEN` (1024 bytes). The
//! mining hot path always satisfies this — A is (m, k) int8 with m*k a multiple
//! of 1024, B same.
//!
//! Math (mirrors `merkle_root_device` in `kernels/merkle_sm80.cuh`):
//!
//! 1. Launch `chunk_cv_kernel_noroot` over `num_chunks` threads, one per
//!    1024-byte chunk, producing leaf CVs into `bufA[0..num_chunks*8]`.
//! 2. Iteratively launch `merkle_layer_kernel` ping-ponging between `bufA`
//!    and `bufB` until `size <= 2`. Each layer reduces by half, carrying
//!    orphans (odd nodes) through.
//! 3. Final layer with `is_top_pair_root=true` gets the ROOT flag on its
//!    one compress, producing the 32-byte merkle root.
//! 4. `cuMemcpyDtoDAsync` 32 bytes from the final buf into `d_out`.
//!
//! Single-chunk fast path: launch `chunk_cv_kernel_root` (kAddRoot=true)
//! directly — it sets the ROOT flag on the last block of the chunk.
//!
//! CPU reference: standard `blake3::keyed_hash(key, data)` is byte-identical
//! to this when `data.len()` is a multiple of 64 bytes (validated in
//! `[[project-sm80-blake3-quirk]]`).

use std::ffi::c_void;

use cudarc::driver::sys as cu;
use cudarc::driver::sys::{CUdeviceptr, CUfunction, CUstream};

use crate::driver::{launch_kernel, DevBuf, Module};
use crate::error::{cu_check, MinerError};
use crate::fatbin::symbols;

/// Chunk length in bytes — matches `pearl::sm80::merkle::CHUNK_LEN` (1024).
pub const CHUNK_LEN: usize = 1024;
/// Output length in u32 — matches `pearl::sm80::merkle::OUT_LEN_U32` (8).
pub const OUT_LEN_U32: usize = 8;

/// One-time kernel handles + persistent scratchpad allocation.
///
/// The scratchpad holds (2 × num_chunks × OUT_LEN_U32) u32s, sized once at
/// construction for the largest expected input. Mining inputs are bounded
/// (A: 8 KB, B: ~117 MB for production shape m=2048, n=28672, k=4096) so a
/// single scratchpad sized for B works for all per-iter tensor_hash calls.
pub struct TensorHash {
    chunk_cv_root: CUfunction,
    chunk_cv_noroot: CUfunction,
    merkle_layer: CUfunction,
    scratch: DevBuf,
    scratch_u32_capacity: usize,
}

impl TensorHash {
    /// Construct, sizing the scratchpad for `max_bytes` (data length will not
    /// exceed this). Round up to a chunk boundary.
    pub fn new(module: &Module, max_bytes: usize) -> Result<Self, MinerError> {
        let max_chunks = (max_bytes + CHUNK_LEN - 1) / CHUNK_LEN;
        // Scratch holds 2 × num_chunks × OUT_LEN_U32 u32s.
        let scratch_u32 = 2 * max_chunks * OUT_LEN_U32;
        // Minimum of 16 u32s for the single-chunk path.
        let scratch_u32 = scratch_u32.max(16);
        let scratch = DevBuf::alloc(scratch_u32 * std::mem::size_of::<u32>())?;

        Ok(Self {
            chunk_cv_root: module.get_function(symbols::CHUNK_CV_ROOT)?,
            chunk_cv_noroot: module.get_function(symbols::CHUNK_CV_NOROOT)?,
            merkle_layer: module.get_function(symbols::MERKLE_LAYER)?,
            scratch,
            scratch_u32_capacity: scratch_u32,
        })
    }

    /// Run the merkle root on `data` (must be a multiple of CHUNK_LEN, on
    /// device), with `key` (32 bytes on device), writing 32 bytes into
    /// `out` (also on device).
    ///
    /// # Safety
    /// All `CUdeviceptr` arguments must point to valid allocations on the same
    /// context as `module` was loaded. `data_len` must be a positive multiple
    /// of CHUNK_LEN, and `data` must point to at least that many bytes.
    pub unsafe fn launch(
        &self,
        data: CUdeviceptr,
        data_len: usize,
        key: CUdeviceptr,
        out: CUdeviceptr,
        stream: CUstream,
    ) -> Result<(), MinerError> {
        assert!(
            data_len > 0 && data_len % CHUNK_LEN == 0,
            "data_len must be a positive multiple of CHUNK_LEN (1024)"
        );
        let num_chunks: i32 = (data_len / CHUNK_LEN) as i32;
        let needed_u32 = 2 * (num_chunks as usize) * OUT_LEN_U32;
        assert!(
            needed_u32 <= self.scratch_u32_capacity,
            "scratchpad too small ({} > {}); construct TensorHash with larger max_bytes",
            needed_u32,
            self.scratch_u32_capacity
        );

        // Scratchpad layout: bufA at [0, num_chunks*8), bufB at [num_chunks*8, ...).
        let scratch_base = self.scratch.ptr;
        let buf_a: CUdeviceptr = scratch_base;
        let buf_b: CUdeviceptr = scratch_base
            + ((num_chunks as u64) * (OUT_LEN_U32 as u64) * std::mem::size_of::<u32>() as u64);

        if num_chunks == 1 {
            // Single chunk: chunk_cv_kernel<kAddRoot=true> produces the root.
            // Args: (padded_data, num_chunks, key, leaf_cvs)
            let mut p_data = data;
            let mut p_nc = num_chunks;
            let mut p_key = key;
            let mut p_out = buf_a;
            let mut params: [*mut c_void; 4] = [
                &mut p_data as *mut _ as *mut c_void,
                &mut p_nc as *mut _ as *mut c_void,
                &mut p_key as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
            ];
            launch_kernel(
                self.chunk_cv_root,
                (1, 1, 1),
                (1, 1, 1),
                0,
                stream,
                &mut params,
            )?;
            // Copy 32 bytes from buf_a to out.
            cu_check(
                cu::cuMemcpyDtoDAsync_v2(out, buf_a, 32, stream),
                "cuMemcpyDtoDAsync(out<-bufA)",
            )?;
            return Ok(());
        }

        // Multi-chunk: chunk_cv_kernel<kAddRoot=false> populates bufA.
        {
            let mut p_data = data;
            let mut p_nc = num_chunks;
            let mut p_key = key;
            let mut p_out = buf_a;
            let mut params: [*mut c_void; 4] = [
                &mut p_data as *mut _ as *mut c_void,
                &mut p_nc as *mut _ as *mut c_void,
                &mut p_key as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
            ];
            let block_x: u32 = 128;
            let grid_x: u32 = ((num_chunks as u32) + block_x - 1) / block_x;
            launch_kernel(
                self.chunk_cv_noroot,
                (grid_x, 1, 1),
                (block_x, 1, 1),
                0,
                stream,
                &mut params,
            )?;
        }

        // Ping-pong merkle_layer_kernel until size <= 2, then final with ROOT.
        let mut in_buf: CUdeviceptr = buf_a;
        let mut out_buf: CUdeviceptr = buf_b;
        let mut size: i32 = num_chunks;

        while size > 2 {
            let num_pairs = size >> 1;
            let has_orphan = size & 1;
            let total_threads = num_pairs + has_orphan;
            let block_x: u32 = 128;
            let grid_x: u32 = ((total_threads as u32) + block_x - 1) / block_x;

            let mut p_in = in_buf;
            let mut p_n = size;
            let mut p_key = key;
            let mut p_root: u8 = 0; // is_top_pair_root=false
            let mut p_out = out_buf;
            // C++ `bool` is 1 byte, padded; we pass a u8.
            let mut params: [*mut c_void; 5] = [
                &mut p_in as *mut _ as *mut c_void,
                &mut p_n as *mut _ as *mut c_void,
                &mut p_key as *mut _ as *mut c_void,
                &mut p_root as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
            ];
            launch_kernel(
                self.merkle_layer,
                (grid_x, 1, 1),
                (block_x, 1, 1),
                0,
                stream,
                &mut params,
            )?;

            std::mem::swap(&mut in_buf, &mut out_buf);
            size = num_pairs + has_orphan;
        }

        // size == 2: final pair gets ROOT.
        {
            let mut p_in = in_buf;
            let mut p_n: i32 = 2;
            let mut p_key = key;
            let mut p_root: u8 = 1; // is_top_pair_root=true
            let mut p_out = out_buf;
            let mut params: [*mut c_void; 5] = [
                &mut p_in as *mut _ as *mut c_void,
                &mut p_n as *mut _ as *mut c_void,
                &mut p_key as *mut _ as *mut c_void,
                &mut p_root as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
            ];
            launch_kernel(
                self.merkle_layer,
                (1, 1, 1),
                (32, 1, 1),
                0,
                stream,
                &mut params,
            )?;
        }

        // Final root is in out_buf[0..32].
        cu_check(
            cu::cuMemcpyDtoDAsync_v2(out, out_buf, 32, stream),
            "cuMemcpyDtoDAsync(out<-merkle_root)",
        )?;
        Ok(())
    }
}

/// CPU reference. Uses standard `blake3::keyed_hash`, which is byte-exact
/// with the SM80 merkle implementation for inputs that are a multiple of 64
/// bytes (per [[project-sm80-blake3-quirk]]).
pub fn reference(data: &[u8], key: &[u8; 32]) -> [u8; 32] {
    *blake3::keyed_hash(key, data).as_bytes()
}
