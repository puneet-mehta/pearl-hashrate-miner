//! `noise_gen` — keyed Blake3 noise tensors used by `noisy_gemm`.
//!
//! Four kernel variants, all at R=128 (the production noise_rank):
//!   - `noise_gen_dense_int8_R128`   → EAL, EBR (int8 rows × R)
//!   - `noise_gen_dense_fp16_R128`   → EAL_fp16, EBR_fp16 (fp16 rows × R, scaled)
//!   - `noise_gen_sparse_R128`       → EAR/EBL_R_major ((k, R) int8, 2 non-zeros/row)
//!   - `transpose_kr_kernel`         → (k, R) → (R, k) for K-major sparse variants
//!
//! Per the kernel comment in `noise_generation_sm80.cuh`, each output chunk is
//! produced by a single keyed Blake3 compress (single-block-keyed flags,
//! counter=0, block_len=64). Bit-exact CPU reference uses
//! `blake3::keyed_hash(key, 64-byte message)`, which is identical to the GPU
//! kernel's compress for the same key/message.
//!
//! The "message" is 8 LE u32 zeros followed by the 8-u32 seed, with one
//! position patched per kernel:
//!   - dense (int8/fp16): message[0] = chunk_idx + 1
//!   - sparse:            message[1] = chunk_idx + 1

use std::ffi::c_void;

use cudarc::driver::sys::{CUdeviceptr, CUfunction, CUstream};

use crate::driver::{launch_kernel, Module};
use crate::error::MinerError;
use crate::fatbin::symbols;

pub const R: usize = 128;

/// All four cached kernel handles.
pub struct NoiseGen {
    dense_int8: CUfunction,
    dense_fp16: CUfunction,
    sparse: CUfunction,
    transpose: CUfunction,
}

impl NoiseGen {
    pub fn new(module: &Module) -> Result<Self, MinerError> {
        Ok(Self {
            dense_int8: module.get_function(symbols::NOISE_GEN_DENSE_INT8_R128)?,
            dense_fp16: module.get_function(symbols::NOISE_GEN_DENSE_FP16_R128)?,
            sparse: module.get_function(symbols::NOISE_GEN_SPARSE_R128)?,
            transpose: module.get_function(symbols::TRANSPOSE_KR)?,
        })
    }

    /// Dense int8: out = (rows × R) int8 noise tensor.
    ///
    /// # Safety
    /// `out` must point to a `rows * R`-byte device allocation. `key` and `seed`
    /// must be 32-byte (8 u32) device allocations.
    pub unsafe fn launch_dense_int8(
        &self,
        rows: i32,
        key: CUdeviceptr,
        seed: CUdeviceptr,
        out: CUdeviceptr,
        stream: CUstream,
    ) -> Result<(), MinerError> {
        let num_chunks = (rows as usize * R) / 32;
        let block_x: u32 = 128;
        let grid_x: u32 = ((num_chunks as u32) + block_x - 1) / block_x;

        let mut p_rows = rows;
        let mut p_key = key;
        let mut p_seed = seed;
        let mut p_out = out;
        let mut params: [*mut c_void; 4] = [
            &mut p_rows as *mut _ as *mut c_void,
            &mut p_key as *mut _ as *mut c_void,
            &mut p_seed as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
        ];
        launch_kernel(
            self.dense_int8,
            (grid_x, 1, 1),
            (block_x, 1, 1),
            0,
            stream,
            &mut params,
        )
    }

    /// Dense fp16: out = (rows × R) fp16 noise tensor scaled by `scale_factor`.
    ///
    /// # Safety
    /// `out` must point to a `rows * R * 2`-byte device allocation (fp16).
    pub unsafe fn launch_dense_fp16(
        &self,
        rows: i32,
        key: CUdeviceptr,
        seed: CUdeviceptr,
        scale_factor: i32,
        out: CUdeviceptr,
        stream: CUstream,
    ) -> Result<(), MinerError> {
        let num_chunks = (rows as usize * R) / 32;
        let block_x: u32 = 128;
        let grid_x: u32 = ((num_chunks as u32) + block_x - 1) / block_x;

        let mut p_rows = rows;
        let mut p_key = key;
        let mut p_seed = seed;
        let mut p_scale = scale_factor;
        let mut p_out = out;
        let mut params: [*mut c_void; 5] = [
            &mut p_rows as *mut _ as *mut c_void,
            &mut p_key as *mut _ as *mut c_void,
            &mut p_seed as *mut _ as *mut c_void,
            &mut p_scale as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
        ];
        launch_kernel(
            self.dense_fp16,
            (grid_x, 1, 1),
            (block_x, 1, 1),
            0,
            stream,
            &mut params,
        )
    }

    /// Sparse: out = (k, R) int8 R-major with exactly 2 non-zeros (±1) per row.
    /// **Caller must zero-init `out_r_major` before this call** (the kernel only
    /// writes the two non-zeros per row, leaving other cells alone).
    ///
    /// # Safety
    /// `out_r_major` must point to a `k * R`-byte device allocation, zero-filled.
    pub unsafe fn launch_sparse(
        &self,
        k: i32,
        key: CUdeviceptr,
        seed: CUdeviceptr,
        out_r_major: CUdeviceptr,
        stream: CUstream,
    ) -> Result<(), MinerError> {
        let num_chunks = ((k as usize) + 7) / 8;
        let block_x: u32 = 128;
        let grid_x: u32 = ((num_chunks as u32) + block_x - 1) / block_x;

        let mut p_k = k;
        let mut p_key = key;
        let mut p_seed = seed;
        let mut p_out = out_r_major;
        let mut params: [*mut c_void; 4] = [
            &mut p_k as *mut _ as *mut c_void,
            &mut p_key as *mut _ as *mut c_void,
            &mut p_seed as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
        ];
        launch_kernel(
            self.sparse,
            (grid_x, 1, 1),
            (block_x, 1, 1),
            0,
            stream,
            &mut params,
        )
    }

    /// Transpose: dst_rk = src_kr.T. `src_kr` is (k, R), `dst_rk` is (R, k).
    ///
    /// # Safety
    /// Both buffers must hold `k * R` int8 elements.
    pub unsafe fn launch_transpose(
        &self,
        k: i32,
        r: i32,
        src_kr: CUdeviceptr,
        dst_rk: CUdeviceptr,
        stream: CUstream,
    ) -> Result<(), MinerError> {
        let block_x: u32 = 16;
        let block_y: u32 = 16;
        let grid_x: u32 = ((k as u32) + block_x - 1) / block_x;
        let grid_y: u32 = ((r as u32) + block_y - 1) / block_y;

        let mut p_k = k;
        let mut p_r = r;
        let mut p_src = src_kr;
        let mut p_dst = dst_rk;
        let mut params: [*mut c_void; 4] = [
            &mut p_k as *mut _ as *mut c_void,
            &mut p_r as *mut _ as *mut c_void,
            &mut p_src as *mut _ as *mut c_void,
            &mut p_dst as *mut _ as *mut c_void,
        ];
        launch_kernel(
            self.transpose,
            (grid_x, grid_y, 1),
            (block_x, block_y, 1),
            0,
            stream,
            &mut params,
        )
    }
}

// =============================================================================
//   CPU reference implementations (bit-exact vs the GPU kernels)
// =============================================================================

/// Compute one keyed Blake3 single-block-keyed compress, byte-exact with the
/// GPU `keyed_compress` helper. Returns the 8-u32 chaining value as 32 bytes.
fn keyed_compress(key_bytes: &[u8; 32], message_bytes: &[u8; 64]) -> [u8; 32] {
    // blake3::keyed_hash with a 64-byte message produces:
    //   cv = compress(IV=key, msg, t=0, block_len=64,
    //                 flags=KEYED_HASH|CHUNK_START|CHUNK_END|ROOT)
    // — which is exactly the kernel's single-block-keyed compress.
    *blake3::keyed_hash(key_bytes, message_bytes).as_bytes()
}

/// Build the 64-byte message for a noise chunk:
///   8 LE u32 zeros, then 8 LE u32 from `seed`, then patch position `slot`
///   (slot=0 for dense, slot=1 for sparse) with `chunk_idx + 1`.
fn make_message(seed: &[u8; 32], slot: usize, chunk_idx_plus_one: u32) -> [u8; 64] {
    let mut msg = [0u8; 64];
    // bytes 32..64 = seed (already in LE u32 form as bytes)
    msg[32..64].copy_from_slice(seed);
    // Patch position [slot] (a u32) with chunk_idx_plus_one (LE).
    let word_off = slot * 4;
    msg[word_off..word_off + 4].copy_from_slice(&chunk_idx_plus_one.to_le_bytes());
    msg
}

/// Reference: dense int8 noise. Same algorithm as `pearl_noise_gen_dense_int8_R128`.
pub fn reference_dense_int8(rows: i32, key: &[u8; 32], seed: &[u8; 32]) -> Vec<i8> {
    let num_chunks = (rows as usize * R) / 32;
    let mut out = vec![0i8; rows as usize * R];
    for chunk in 0..num_chunks {
        let msg = make_message(seed, /*slot=*/ 0, (chunk + 1) as u32);
        let cv = keyed_compress(key, &msg);

        let row = (chunk * 32) / R;
        let col0 = (chunk * 32) % R;
        let out_row = &mut out[row * R + col0..row * R + col0 + 32];
        for (i, b) in cv.iter().enumerate() {
            let hb = *b as i8;
            // ((int32(hb) + 128) % 64) - 32 ∈ [-32, 32)
            let v = ((hb as i32 + 128).rem_euclid(64)) - 32;
            out_row[i] = v as i8;
        }
    }
    out
}

/// Reference: dense fp16 noise. Returns Vec<u16> (raw fp16 bit patterns) since
/// no f16 type exists in std. Each `u16` is the IEEE 754 binary16 encoding of
/// `(byte_value as f32) * scale_factor`.
///
/// Same byte decoding as the int8 path, then scaled and packed as fp16.
pub fn reference_dense_fp16(
    rows: i32,
    key: &[u8; 32],
    seed: &[u8; 32],
    scale_factor: i32,
) -> Vec<u16> {
    let num_chunks = (rows as usize * R) / 32;
    let mut out = vec![0u16; rows as usize * R];
    for chunk in 0..num_chunks {
        let msg = make_message(seed, 0, (chunk + 1) as u32);
        let cv = keyed_compress(key, &msg);
        let row = (chunk * 32) / R;
        let col0 = (chunk * 32) % R;
        let out_row = &mut out[row * R + col0..row * R + col0 + 32];
        for (i, b) in cv.iter().enumerate() {
            let hb = *b as i8;
            let v = ((hb as i32 + 128).rem_euclid(64)) - 32;
            let scaled = (v * scale_factor) as f32;
            out_row[i] = f32_to_f16_bits(scaled);
        }
    }
    out
}

/// Reference: sparse noise. Output is `k * R` i8, two non-zeros per row.
pub fn reference_sparse(k: i32, key: &[u8; 32], seed: &[u8; 32]) -> Vec<i8> {
    let num_chunks = ((k as usize) + 7) / 8;
    let mut out = vec![0i8; k as usize * R];
    for chunk in 0..num_chunks {
        let msg = make_message(seed, /*slot=*/ 1, (chunk + 1) as u32);
        let cv = keyed_compress(key, &msg);
        let k_base = chunk * 8;
        for j in 0..8usize {
            let row = k_base + j;
            if row >= k as usize {
                break;
            }
            // Each u32 word = cv[j*4..j*4+4] in LE.
            let word_bytes: [u8; 4] = cv[j * 4..j * 4 + 4].try_into().unwrap();
            let u = u32::from_le_bytes(word_bytes);
            let r_mask = (R as u32) - 1; // 127
            let k0 = u & r_mask;
            let mul_hi = ((r_mask as u64 * u as u64) >> 32) as u32;
            let k1 = (k0 ^ (1u32.wrapping_add(mul_hi))) % (R as u32);
            let off = row * R;
            out[off + k0 as usize] = 1;
            out[off + k1 as usize] = -1;
        }
    }
    out
}

/// IEEE 754 binary32 → binary16 (round-to-nearest-even). Matches CUDA's
/// `__float2half` for the values we produce (small integers; no subnormal /
/// overflow cases triggered).
fn f32_to_f16_bits(x: f32) -> u16 {
    // Use half-precision conversion if half crate is available, else implement
    // a minimal RTNE converter inline.
    half::f16::from_f32(x).to_bits()
}
