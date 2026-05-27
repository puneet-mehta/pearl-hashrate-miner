//! `noisy_gemm` pipeline — the rest of the per-iter sequence.
//!
//! Three kernels here, all reused on both noising-A and noising-B paths:
//!
//! 1. **`add_gemm_int8_smem`** — `Out = wrap_int8(X + Y @ Z^T)` in int8. Used
//!    to produce `ApEA` from `(A, EAL, EAR)` and `BpEB` from `(B, EBR, EBL)`.
//! 2. **`gemm_int8_int32_smem`** — `C = A @ B` with int32 accumulation. Used
//!    by the denoise side of the pipeline. `B` is (K, N) row-major.
//! 3. **`int32_to_fp16_scaled`** — `dst = float16(src * 2^scale_power)`,
//!    element-wise. Bit-equivalent to division by `2^|scale_power|` for the
//!    denoise scale factors.
//!
//! These kernels live at file scope in their .cuh headers (not in anon
//! namespaces), so their Itanium-mangled C++ symbol names are stable across
//! rebuilds. The mangled-name constants below come from `cuobjdump
//! --dump-elf-symbols` on the the reference image fatbin.
//!
//! The companion search kernel (`pearl_gemm_search_perthread_smem_pipelined_kernel<R>`)
//! is handled in [`crate::kernels::search`] because it needs a
//! `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, 96 KB)` opt-in.

use std::ffi::c_void;

use cudarc::driver::sys::{CUdeviceptr, CUfunction, CUstream};

use crate::driver::{launch_kernel, Module};
use crate::error::MinerError;
use crate::fatbin::symbols;

/// Tile constants matching `kernels/noising_smem_sm80.cuh`.
pub const TILE_M: i32 = 128;
pub const TILE_N: i32 = 128;
pub const CTA_THREADS: u32 = 256;

pub struct NoisyGemm {
    add_gemm: CUfunction,
    gemm_int32: CUfunction,
    int32_to_fp16: CUfunction,
}

impl NoisyGemm {
    pub fn new(module: &Module) -> Result<Self, MinerError> {
        Ok(Self {
            add_gemm: module.get_function(symbols::NOISING_ADD_GEMM_INT8_SMEM)?,
            gemm_int32: module.get_function(symbols::NOISING_GEMM_INT8_INT32_SMEM)?,
            int32_to_fp16: module.get_function(symbols::NOISING_INT32_TO_FP16_SCALED)?,
        })
    }

    /// `Out = wrap_int8(X + Y @ Z^T)`.
    /// `X`: (M, N) int8.  `Y`: (M, K_inner) int8.  `Z`: (N, K_inner) int8.
    /// `Out`: (M, N) int8 — wrap-around on overflow.
    ///
    /// Requires M % 128 == 0 and N % 128 == 0 (smem kernel constraint).
    ///
    /// # Safety
    /// All buffers must be valid device allocations of the matching sizes on
    /// the same context.
    pub unsafe fn launch_add_gemm(
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
        assert!(
            m % TILE_M == 0 && n % TILE_N == 0,
            "add_gemm requires M%128==0 && N%128==0; got M={} N={}",
            m,
            n
        );
        let grid_x: u32 = (n / TILE_N) as u32;
        let grid_y: u32 = (m / TILE_M) as u32;

        let mut p_m = m;
        let mut p_n = n;
        let mut p_k = k_inner;
        let mut p_x = x;
        let mut p_y = y;
        let mut p_z = z;
        let mut p_out = out;
        let mut params: [*mut c_void; 7] = [
            &mut p_m as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut p_x as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut p_z as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
        ];
        launch_kernel(
            self.add_gemm,
            (grid_x, grid_y, 1),
            (CTA_THREADS, 1, 1),
            0,
            stream,
            &mut params,
        )
    }

    /// `C = A @ B`, int32 accumulation. `B` is (K, N) row-major.
    ///
    /// Requires M % 128 == 0 and N % 128 == 0.
    ///
    /// # Safety
    /// All buffers must be valid device allocations of the matching sizes on
    /// the same context.
    pub unsafe fn launch_gemm_int32(
        &self,
        m: i32,
        n: i32,
        k: i32,
        a: CUdeviceptr,
        b: CUdeviceptr,
        c: CUdeviceptr,
        stream: CUstream,
    ) -> Result<(), MinerError> {
        assert!(
            m % TILE_M == 0 && n % TILE_N == 0,
            "gemm_int32 requires M%128==0 && N%128==0; got M={} N={}",
            m,
            n
        );
        let grid_x: u32 = (n / TILE_N) as u32;
        let grid_y: u32 = (m / TILE_M) as u32;

        let mut p_m = m;
        let mut p_n = n;
        let mut p_k = k;
        let mut p_a = a;
        let mut p_b = b;
        let mut p_c = c;
        let mut params: [*mut c_void; 6] = [
            &mut p_m as *mut _ as *mut c_void,
            &mut p_n as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut p_a as *mut _ as *mut c_void,
            &mut p_b as *mut _ as *mut c_void,
            &mut p_c as *mut _ as *mut c_void,
        ];
        launch_kernel(
            self.gemm_int32,
            (grid_x, grid_y, 1),
            (CTA_THREADS, 1, 1),
            0,
            stream,
            &mut params,
        )
    }

    /// Element-wise: `dst[i] = float16(src[i] * 2^scale_power)`.
    ///
    /// # Safety
    /// `src` and `dst` must be valid device allocations of N elements each.
    pub unsafe fn launch_int32_to_fp16(
        &self,
        n: i32,
        src: CUdeviceptr,
        dst: CUdeviceptr,
        scale_power: i32,
        stream: CUstream,
    ) -> Result<(), MinerError> {
        let block_x: u32 = 256;
        let grid_x: u32 = ((n as u32) + block_x - 1) / block_x;

        let mut p_n = n;
        let mut p_src = src;
        let mut p_dst = dst;
        let mut p_scale = scale_power;
        let mut params: [*mut c_void; 4] = [
            &mut p_n as *mut _ as *mut c_void,
            &mut p_src as *mut _ as *mut c_void,
            &mut p_dst as *mut _ as *mut c_void,
            &mut p_scale as *mut _ as *mut c_void,
        ];
        launch_kernel(
            self.int32_to_fp16,
            (grid_x, 1, 1),
            (block_x, 1, 1),
            0,
            stream,
            &mut params,
        )
    }
}

// =============================================================================
//   CPU references
// =============================================================================

/// CPU reference for `add_gemm_int8_smem`:
///   Out[i, j] = wrap_int8( X[i, j] + sum_r Y[i, r] * Z[j, r] )
pub fn reference_add_gemm(
    m: usize,
    n: usize,
    k_inner: usize,
    x: &[i8],
    y: &[i8],
    z: &[i8],
) -> Vec<i8> {
    assert_eq!(x.len(), m * n);
    assert_eq!(y.len(), m * k_inner);
    assert_eq!(z.len(), n * k_inner);
    let mut out = vec![0i8; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = x[i * n + j] as i32;
            for r in 0..k_inner {
                acc = acc.wrapping_add((y[i * k_inner + r] as i32) * (z[j * k_inner + r] as i32));
            }
            // int8 wrap: take low 8 bits, sign-extend.
            out[i * n + j] = (acc & 0xff) as i8;
        }
    }
    out
}

/// CPU reference for `gemm_int8_int32_smem`:
///   C[i, j] = sum_k A[i, k] * B[k, j]
/// where A is (M, K) row-major and B is (K, N) row-major.
pub fn reference_gemm_int32(m: usize, n: usize, k: usize, a: &[i8], b: &[i8]) -> Vec<i32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = 0;
            for kk in 0..k {
                acc = acc.wrapping_add((a[i * k + kk] as i32) * (b[kk * n + j] as i32));
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// CPU reference for `int32_to_fp16_scaled_kernel`:
///   dst[i] = fp16( src[i] as f32 * 2^scale_power )
pub fn reference_int32_to_fp16(src: &[i32], scale_power: i32) -> Vec<u16> {
    // 2^scale_power as exact f32 via ldexp.
    let scale = (2.0f32).powi(scale_power);
    src.iter()
        .map(|s| {
            let v = (*s as f32) * scale;
            half::f16::from_f32(v).to_bits()
        })
        .collect()
}
