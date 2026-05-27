//! Rust equivalent of the Python `MinerBufs` class — owns all the GPU and
//! pinned-host buffers the per-iter mining loop reads/writes, plus cached
//! kernel handles.
//!
//! Layout mirrors `pearl_hashrate_miner.main.MinerBufs`:
//!
//! - **Per-iter ring** (`ring_size` slots, default 8): each slot holds an
//!   `A` activation tensor, the Blake3 `A_tensor_hash`, and the per-iter
//!   `commit_A` / `commit_B` keys. A separate pinned-host snapshot ring
//!   captures these for the async hit callback to read without racing
//!   the GPU writer.
//! - **Per-job state**: `B`, `B_tensor_hash`, `key_tensor`, `pow_target`,
//!   refreshed by [`MinerBufs::ensure_for_job`] when the chain tip advances.
//! - **Shared persistent**: noise tensors (`EAL`, `EBR`, …), search workspaces
//!   (`pow_workspace_hit`, `pow_workspace_hash`, `pow_workspace_scan`),
//!   intermediates (`ApEA`, `BpEB`, …).
//! - **Kernel handles**: one of each ported kernel module, owned by MinerBufs.
//!
//! The next milestone wires up the per-iter sequence (`mine_one`) that drives
//! the kernel chain. This commit lands the data structures + allocation.

use crate::driver::{DevBuf, Module, PinnedHostBuf};
use crate::error::MinerError;
use crate::kernels::{
    commitment_hash::CommitmentHash,
    noise_gen::NoiseGen,
    noisy_gemm::NoisyGemm,
    pow_scan_emit::PowScanEmit,
    random_int8::RandomInt8,
    search::Search,
    tensor_hash::TensorHash,
    triton::{
        noising_ptx, search_norotl_ptx, TritonNoising, TritonPostpass, TritonSearchNorotl,
        BLOCK_M as TRITON_BLOCK_M, BLOCK_N as TRITON_BLOCK_N,
        HASH_CANDIDATES as TRITON_HASH_CANDIDATES, JACKPOT_SIZE as TRITON_JACKPOT_SIZE,
    },
};

/// Host-signal header size in bytes (mirrors `pearl_gemm::get_host_signal_header_size()`).
/// The kernel writes the first 640 bytes; the rest is allocation padding to
/// match the Python side. Validated 2026-05-25 against pearl_gemm in
/// the the reference image image (returns 1024).
pub const HOST_SIGNAL_HEADER_SIZE: usize = 1024;
/// Tile shape (mirrors `MinerSettings.tile_size_m/n`).
pub const TILE_M: usize = 128;
pub const TILE_N: usize = 128;
pub const TILE_K: usize = 128;
/// Threads per (search) tile.
pub const THREADS_PER_TILE: usize = 256;
/// Ring depth — must exceed the worst-case async hit-callback latency.
pub const RING_SIZE: usize = 8;

/// Production noise rank.
pub const NOISE_RANK: usize = 128;

pub struct MinerBufs {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub r: usize,
    pub ring_size: usize,
    pub num_tiles: usize,

    // ----- Per-iter ring (ring_size slots) -----
    pub a_pool: Vec<DevBuf>,                         // (m, k) int8
    pub a_tensor_hash_pool: Vec<DevBuf>,             // 32 u8
    pub commit_a_pool: Vec<DevBuf>,                  // 32 u8
    pub commit_b_pool: Vec<DevBuf>,                  // 32 u8
    pub host_signal_header_pool: Vec<PinnedHostBuf>, // 640 u8, pinned host (UVA mapped)

    // Per-slot CPU-pinned snapshots populated by stream-ordered D2H after the
    // replay. The async hit callback reads only these, never the live ring.
    pub a_snapshot_pool: Vec<PinnedHostBuf>, // (m, k) int8
    pub commit_a_snapshot_pool: Vec<PinnedHostBuf>, // 32 u8
    pub commit_b_snapshot_pool: Vec<PinnedHostBuf>, // 32 u8

    // ----- Per-job (refreshed in ensure_for_job) -----
    pub b: DevBuf,                 // (n, k) int8
    pub b_tensor_hash: DevBuf,     // 32 u8
    pub key_tensor: DevBuf,        // 32 u8 (Blake3 key)
    pub pow_target_tensor: DevBuf, // 32 u8
    pub b_pinned: PinnedHostBuf,   // (n, k) int8 pinned
    /// Per-job session seed for `random_int8_seeded` (32 bytes).
    pub seed_tensor: DevBuf,
    /// Cached incomplete_header_bytes to detect job changes.
    pub cached_header: Option<Vec<u8>>,
    pub adjusted_target: Option<u128>, // store low 128 bits; full 256 lives in pow_target_tensor

    // Persistent noise_gen seed labels (matches the Python wrapper's default
    // when seed_A/seed_B kwargs are omitted: "A_tensor" + 24 zero bytes).
    pub seed_label_a: DevBuf, // 32 bytes
    pub seed_label_b: DevBuf,

    // ----- Shared persistent -----
    pub a_scales: DevBuf, // (m,) f32
    pub b_scales: DevBuf, // (n,) f32

    // Noise + intermediates.
    pub eal: DevBuf,         // (m, r) i8
    pub ebr: DevBuf,         // (n, r) i8
    pub ear_r_major: DevBuf, // (k, r) i8
    pub ebl_r_major: DevBuf, // (k, r) i8
    pub ear_k_major: DevBuf, // (r, k) i8
    pub ebl_k_major: DevBuf, // (r, k) i8
    pub eal_fp16: DevBuf,    // (m, r) fp16
    pub ebr_fp16: DevBuf,    // (n, r) fp16
    pub bp_eb: DevBuf,       // (n, k) i8 — produced by noising-B
    pub earx_bp_eb: DevBuf,  // (n, r) fp16
    pub ap_ea: DevBuf,       // (m, k) i8 — produced by noising-A
    pub a_e_bl: DevBuf,      // (m, r) fp16

    // Search workspaces.
    pub pow_workspace_hit: DevBuf,  // num_tiles * 256 u8
    pub pow_workspace_hash: DevBuf, // num_tiles * 256 * 8 u32
    pub pow_workspace_scan: DevBuf, // 1 u32 (atomicMin sentinel)

    // ----- Kernel handles -----
    pub random_int8: RandomInt8,
    pub tensor_hash: TensorHash,
    pub commitment_hash: CommitmentHash,
    pub noise_gen: NoiseGen,
    pub noisy_gemm: NoisyGemm,
    pub search: Search,
    pub pow_scan_emit: PowScanEmit,

    /// Optional: Triton kernels + buffers for the no-rotl path.
    /// `None` for the C++ search path (production shape).
    pub triton: Option<TritonPath>,
}

/// Triton no-rotl kernels + the per-iter transcript buffer.
/// Cubins are loaded as a separate CUmodule (kept alive in `_noising_mod`
/// / `_search_mod`).
pub struct TritonPath {
    pub noising: TritonNoising,
    pub search: TritonSearchNorotl,
    pub postpass: TritonPostpass,
    /// (num_tiles, HASH_CANDIDATES=64, JACKPOT_SIZE=16) uint32. Caller
    /// must pre-zero before each search launch; the search kernel only
    /// writes to slots it actually visits.
    pub transcripts: DevBuf,
    pub num_triton_tile_m: i32,
    pub num_triton_tile_n: i32,
    /// Keep the modules alive — dropped after the kernel handles.
    _noising_mod: Module,
    _search_mod: Module,
}

/// Which search-path the miner is configured for. Affects shape,
/// pattern, sizing of pow_workspace, AND which kernels mine_one calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MiningPath {
    /// C++ search_perthread kernel (production). Pattern is
    /// (rows=[0,8], cols=[0,1,8,9,…,120,121]).
    CppSearch,
    /// Triton no-rotl. Pattern is (rows=[0,1], cols=range(128)).
    /// Requires default shape (K=2048, R=128, K/R==JACKPOT_SIZE).
    TritonNorotl,
}

/// Build params — equivalent to the Python `MinerBufs.__init__` args.
#[derive(Clone, Copy)]
pub struct MinerBufsConfig {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub r: usize,
    pub ring_size: usize,
    pub path: MiningPath,
}

impl MinerBufsConfig {
    /// Legacy production: m=2048, n=28672, k=4096, r=128. C++ search kernel.
    pub fn production() -> Self {
        Self {
            m: 2048,
            n: 28672,
            k: 4096,
            r: 128,
            ring_size: 8,
            path: MiningPath::CppSearch,
        }
    }

    /// no-rotl: default shape m=8192 n=32768 k=2048 r=128. Triton path.
    /// K/R = 16 = JACKPOT_SIZE (no-rotl invariant).
    pub fn shape_default() -> Self {
        Self {
            m: 8192,
            n: 32768,
            k: 2048,
            r: 128,
            ring_size: 8,
            path: MiningPath::TritonNorotl,
        }
    }

    /// no-rotl with 2× M dimension: m=16384 n=32768 k=2048 r=128.
    /// More search tiles per iter → potentially better SM utilization on
    /// Blackwell-class GPUs that have spare SM capacity at the default shape.
    /// K/R == JACKPOT_SIZE preserved.
    pub fn shape_big_m() -> Self {
        Self {
            m: 16384,
            n: 32768,
            k: 2048,
            r: 128,
            ring_size: 8,
            path: MiningPath::TritonNorotl,
        }
    }

    /// no-rotl with 4× M dimension: m=32768 n=32768 k=2048 r=128.
    pub fn shape_huge_m() -> Self {
        Self {
            m: 32768,
            n: 32768,
            k: 2048,
            r: 128,
            ring_size: 8,
            path: MiningPath::TritonNorotl,
        }
    }

    /// no-rotl with 2× M AND 2× N: m=16384 n=65536 k=2048 r=128.
    pub fn shape_big_mn() -> Self {
        Self {
            m: 16384,
            n: 65536,
            k: 2048,
            r: 128,
            ring_size: 8,
            path: MiningPath::TritonNorotl,
        }
    }

    /// no-rotl giga: m=32768 n=65536 k=2048 r=128 (8× MACs).
    pub fn shape_giga() -> Self {
        Self {
            m: 32768,
            n: 65536,
            k: 2048,
            r: 128,
            ring_size: 8,
            path: MiningPath::TritonNorotl,
        }
    }

    /// no-rotl very wide: m=49152 n=49152 k=2048 r=128 (intermediate).
    pub fn shape_49k_n() -> Self {
        Self {
            m: 49152,
            n: 49152,
            k: 2048,
            r: 128,
            ring_size: 8,
            path: MiningPath::TritonNorotl,
        }
    }

    /// no-rotl: m=49152 n=32768 k=2048 r=128 (1.5× huge-M's M).
    pub fn shape_49k_m() -> Self {
        Self {
            m: 49152,
            n: 32768,
            k: 2048,
            r: 128,
            ring_size: 8,
            path: MiningPath::TritonNorotl,
        }
    }
}

/// Query the compute capability of the current CUDA context's device.
/// Returns `(major, minor)`, e.g. `(8, 9)` for an RTX 4090.
fn query_current_device_cc() -> Result<(i32, i32), MinerError> {
    use cudarc::driver::sys as cu;
    let mut device: cu::CUdevice = 0;
    unsafe {
        crate::error::cu_check(cu::cuCtxGetDevice(&mut device), "cuCtxGetDevice")?;
    }
    let mut major: i32 = 0;
    let mut minor: i32 = 0;
    unsafe {
        crate::error::cu_check(
            cu::cuDeviceGetAttribute(
                &mut major,
                cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device,
            ),
            "cuDeviceGetAttribute(CC_MAJOR)",
        )?;
        crate::error::cu_check(
            cu::cuDeviceGetAttribute(
                &mut minor,
                cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                device,
            ),
            "cuDeviceGetAttribute(CC_MINOR)",
        )?;
    }
    Ok((major, minor))
}

impl MinerBufs {
    /// Allocate every buffer + load every kernel handle.
    pub fn new(module: &Module, cfg: MinerBufsConfig) -> Result<Self, MinerError> {
        let MinerBufsConfig {
            m,
            n,
            k,
            r,
            ring_size,
            path,
        } = cfg;
        assert!(m % TILE_M == 0, "m must be a multiple of {}", TILE_M);
        assert!(n % TILE_N == 0, "n must be a multiple of {}", TILE_N);
        assert_eq!(
            r, NOISE_RANK,
            "Rust port currently fixed at R={}",
            NOISE_RANK
        );
        let num_tiles = (m / TILE_M) * (n / TILE_N);

        // For the Triton path the search kernel emits HASH_CANDIDATES=64
        // hashes per tile instead of THREADS_PER_TILE=256 like the C++
        // path. pow_workspace sizing needs to cover whichever is larger.
        let max_hashes_per_tile = match path {
            MiningPath::CppSearch => THREADS_PER_TILE,
            MiningPath::TritonNorotl => TRITON_HASH_CANDIDATES,
        };
        // Triton uses different tile shape (BM=BN=128); num_tiles still
        // matches because we share TILE_M==TILE_N==128 between paths.

        // ---- Per-iter ring ----
        let mk_dev_ring = |bytes: usize| -> Result<Vec<DevBuf>, MinerError> {
            (0..ring_size).map(|_| DevBuf::alloc(bytes)).collect()
        };
        let mk_host_ring = |bytes: usize| -> Result<Vec<PinnedHostBuf>, MinerError> {
            (0..ring_size)
                .map(|_| PinnedHostBuf::alloc(bytes))
                .collect()
        };

        let a_pool = mk_dev_ring(m * k)?;
        let a_tensor_hash_pool = mk_dev_ring(32)?;
        let commit_a_pool = mk_dev_ring(32)?;
        let commit_b_pool = mk_dev_ring(32)?;
        let host_signal_header_pool = mk_host_ring(HOST_SIGNAL_HEADER_SIZE)?;
        let a_snapshot_pool = mk_host_ring(m * k)?;
        let commit_a_snapshot_pool = mk_host_ring(32)?;
        let commit_b_snapshot_pool = mk_host_ring(32)?;

        // ---- Per-job ----
        let b = DevBuf::alloc(n * k)?;
        let b_tensor_hash = DevBuf::alloc(32)?;
        let key_tensor = DevBuf::alloc(32)?;
        let pow_target_tensor = DevBuf::alloc(32)?;
        let b_pinned = PinnedHostBuf::alloc(n * k)?;
        let seed_tensor = DevBuf::alloc(32)?;

        // ---- Shared persistent ----
        let a_scales = DevBuf::alloc(m * 4)?; // f32
        let b_scales = DevBuf::alloc(n * 4)?;

        let eal = DevBuf::alloc(m * r)?;
        let ebr = DevBuf::alloc(n * r)?;
        let ear_r_major = DevBuf::alloc(k * r)?;
        let ebl_r_major = DevBuf::alloc(k * r)?;
        let ear_k_major = DevBuf::alloc(r * k)?;
        let ebl_k_major = DevBuf::alloc(r * k)?;
        let eal_fp16 = DevBuf::alloc(m * r * 2)?; // fp16
        let ebr_fp16 = DevBuf::alloc(n * r * 2)?;
        let bp_eb = DevBuf::alloc(n * k)?;
        let earx_bp_eb = DevBuf::alloc(n * r * 2)?; // fp16
        let ap_ea = DevBuf::alloc(m * k)?;
        let a_e_bl = DevBuf::alloc(m * r * 2)?; // fp16

        let pow_workspace_hit = DevBuf::alloc(num_tiles * max_hashes_per_tile)?;
        let pow_workspace_hash = DevBuf::alloc(num_tiles * max_hashes_per_tile * 8 * 4)?; // u32
        let pow_workspace_scan = DevBuf::alloc(4)?; // 1 u32

        // Persistent noise_gen seed labels: "A_tensor" / "B_tensor" + 24 zeros.
        let mut seed_label_a_bytes = [0u8; 32];
        seed_label_a_bytes[..8].copy_from_slice(b"A_tensor");
        let mut seed_label_b_bytes = [0u8; 32];
        seed_label_b_bytes[..8].copy_from_slice(b"B_tensor");
        let mut seed_label_a = DevBuf::alloc(32)?;
        let mut seed_label_b = DevBuf::alloc(32)?;
        seed_label_a.copy_from(&seed_label_a_bytes)?;
        seed_label_b.copy_from(&seed_label_b_bytes)?;

        // ---- Kernel handles ----
        let random_int8 = RandomInt8::new(module)?;
        let tensor_hash = TensorHash::new(module, n * k)?; // size scratch for largest input (B)
        let commitment_hash = CommitmentHash::new(module)?;
        let noise_gen = NoiseGen::new(module)?;
        let noisy_gemm = NoisyGemm::new(module)?;
        let search = Search::new(module)?;
        let pow_scan_emit = PowScanEmit::new(module)?;

        // Optional: load Triton cubins + allocate transcript buffer.
        let triton = match path {
            MiningPath::CppSearch => None,
            MiningPath::TritonNorotl => {
                assert_eq!(
                    k / r,
                    TRITON_JACKPOT_SIZE,
                    "TritonNorotl requires K/R == JACKPOT_SIZE ({}); got K={} R={}",
                    TRITON_JACKPOT_SIZE,
                    k,
                    r,
                );
                // PTX (per-arch, runtime-selected); driver JITs at load.
                let (cc_maj, cc_min) = query_current_device_cc()?;
                let noising_mod = Module::load_data(&noising_ptx(cc_maj, cc_min))?;
                let search_mod = Module::load_data(&search_norotl_ptx(cc_maj, cc_min))?;
                let noising = TritonNoising::new(&noising_mod)?;
                let search_tri = TritonSearchNorotl::new(&search_mod)?;
                let postpass = TritonPostpass::new(module)?;
                let num_triton_tile_m = (m / TRITON_BLOCK_M) as i32;
                let num_triton_tile_n = (n / TRITON_BLOCK_N) as i32;
                let num_triton_tiles = (num_triton_tile_m as usize) * (num_triton_tile_n as usize);
                let transcripts = DevBuf::alloc(
                    num_triton_tiles * TRITON_HASH_CANDIDATES * TRITON_JACKPOT_SIZE * 4,
                )?;
                Some(TritonPath {
                    noising,
                    search: search_tri,
                    postpass,
                    transcripts,
                    num_triton_tile_m,
                    num_triton_tile_n,
                    _noising_mod: noising_mod,
                    _search_mod: search_mod,
                })
            }
        };

        Ok(MinerBufs {
            m,
            n,
            k,
            r,
            ring_size,
            num_tiles,
            a_pool,
            a_tensor_hash_pool,
            commit_a_pool,
            commit_b_pool,
            host_signal_header_pool,
            a_snapshot_pool,
            commit_a_snapshot_pool,
            commit_b_snapshot_pool,
            b,
            b_tensor_hash,
            key_tensor,
            pow_target_tensor,
            b_pinned,
            seed_tensor,
            cached_header: None,
            adjusted_target: None,
            seed_label_a,
            seed_label_b,
            a_scales,
            b_scales,
            eal,
            ebr,
            ear_r_major,
            ebl_r_major,
            ear_k_major,
            ebl_k_major,
            eal_fp16,
            ebr_fp16,
            bp_eb,
            earx_bp_eb,
            ap_ea,
            a_e_bl,
            pow_workspace_hit,
            pow_workspace_hash,
            pow_workspace_scan,
            random_int8,
            tensor_hash,
            commitment_hash,
            noise_gen,
            noisy_gemm,
            search,
            pow_scan_emit,
            triton,
        })
    }

    /// Compute which ring slot iteration `iter_idx` lives in.
    #[inline]
    pub fn slot(&self, iter_idx: u64) -> usize {
        (iter_idx as usize) % self.ring_size
    }

    /// Refresh per-job device state when the chain tip changes.
    ///
    /// - `header_bytes`: the job's incomplete-header bytes; used to detect
    ///   job changes via cached equality.
    /// - `key`: 32-byte Blake3 key derived from the header on the CPU.
    /// - `target`: 32-byte adjusted target (u256 little-endian).
    /// - `seed`: 32-byte session seed for `random_int8_seeded`.
    /// - `b_iter_idx`: passed to `random_int8_seeded` for B's fill (any
    ///    deterministic value; we use 0).
    ///
    /// On a hit, this also stages a fresh `b_pinned` snapshot (D2H copy)
    /// so the async callback has stable host-side bytes to submit.
    ///
    /// # Safety
    /// `stream` must be a valid CUstream (or null for the default stream).
    pub unsafe fn ensure_for_job(
        &mut self,
        header_bytes: &[u8],
        key: &[u8; 32],
        target: &[u8; 32],
        seed: &[u8; 32],
        b_iter_idx: u64,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<(), MinerError> {
        if let Some(cached) = &self.cached_header {
            if cached.as_slice() == header_bytes {
                return Ok(());
            }
        }

        // Copy job state to device.
        self.key_tensor.copy_from(key)?;
        self.pow_target_tensor.copy_from(target)?;
        self.seed_tensor.copy_from(seed)?;

        // Generate B deterministically from seed.
        self.random_int8.launch(
            (self.n * self.k) as i32,
            self.seed_tensor.ptr,
            b_iter_idx,
            self.b.ptr,
            stream,
        )?;
        // Merkle root of B.
        self.tensor_hash.launch(
            self.b.ptr,
            self.n * self.k,
            self.key_tensor.ptr,
            self.b_tensor_hash.ptr,
            stream,
        )?;
        // D2H snapshot of B for callback use (async on the stream).
        let r = cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
            self.b_pinned.host_ptr,
            self.b.ptr,
            self.n * self.k,
            stream,
        );
        crate::error::cu_check(r, "cuMemcpyDtoHAsync(B->b_pinned)")?;

        self.cached_header = Some(header_bytes.to_vec());
        Ok(())
    }

    /// Step 1 only: eager `random_int8_seeded` for slot's A (different per
    /// `iter_idx`). Kept out of the captureable region so iter_idx isn't
    /// baked into the graph.
    ///
    /// # Safety
    /// `stream` must be a valid CUstream (or null).
    pub unsafe fn random_fill_a(
        &mut self,
        iter_idx: u64,
        slot: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<(), MinerError> {
        self.random_int8.launch(
            (self.m * self.k) as i32,
            self.seed_tensor.ptr,
            iter_idx,
            self.a_pool[slot].ptr,
            stream,
        )
    }

    /// Steps 2-9 of the per-iter sequence: tensor_hash → commitment → noise_gen
    /// → noising → search → scan/emit → D2H snapshots. Stream-ordered, suitable
    /// for `cuStreamBeginCapture`. Does NOT zero the pinned header (CPU
    /// operation; do that before launching).
    ///
    /// # Safety
    /// `stream` must be a valid (non-default) CUstream when used inside graph
    /// capture. `random_fill_a` must have already populated `a_pool[slot]`.
    pub unsafe fn mine_one_post_random(
        &mut self,
        slot: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<(), MinerError> {
        // 2. Merkle root of A.
        self.tensor_hash.launch(
            self.a_pool[slot].ptr,
            self.m * self.k,
            self.key_tensor.ptr,
            self.a_tensor_hash_pool[slot].ptr,
            stream,
        )?;

        // 3. Commitment hash: chain (A_hash, B_hash, key) → commit_A, commit_B.
        self.commitment_hash.launch(
            self.a_tensor_hash_pool[slot].ptr,
            self.b_tensor_hash.ptr,
            self.key_tensor.ptr,
            self.commit_a_pool[slot].ptr,
            self.commit_b_pool[slot].ptr,
            stream,
        )?;

        // 4. Noise generation. Uses commit_A as key for the A-side noise
        //    tensors and commit_B for the B-side; persistent "A_tensor" /
        //    "B_tensor" labels are the seeds.
        //
        //    EAL (m, r) i8:                  dense_int8 with key=commit_A
        //    EBR (n, r) i8:                  dense_int8 with key=commit_B
        //    EAR_R_major (k, r) i8 sparse:   sparse + zeroed first
        //    EAR_K_major (r, k):             transpose of EAR_R_major
        //    EBL_R_major (k, r) i8 sparse:   sparse with key=commit_B
        //    EBL_K_major (r, k):             transpose of EBL_R_major
        //    EAL_fp16, EBR_fp16:             dense_fp16 (scale 1)
        self.noise_gen.launch_dense_int8(
            self.m as i32,
            self.commit_a_pool[slot].ptr,
            self.seed_label_a.ptr,
            self.eal.ptr,
            stream,
        )?;
        self.noise_gen.launch_dense_int8(
            self.n as i32,
            self.commit_b_pool[slot].ptr,
            self.seed_label_b.ptr,
            self.ebr.ptr,
            stream,
        )?;
        self.noise_gen.launch_dense_fp16(
            self.m as i32,
            self.commit_a_pool[slot].ptr,
            self.seed_label_a.ptr,
            /*scale=*/ 1,
            self.eal_fp16.ptr,
            stream,
        )?;
        self.noise_gen.launch_dense_fp16(
            self.n as i32,
            self.commit_b_pool[slot].ptr,
            self.seed_label_b.ptr,
            /*scale=*/ 1,
            self.ebr_fp16.ptr,
            stream,
        )?;
        // Sparse R-major outputs need pre-zeroing. cuMemsetD8Async is stream-
        // ordered so this serializes with the subsequent sparse launch.
        let zero_async = |ptr, bytes| -> Result<(), MinerError> {
            crate::error::cu_check(
                cudarc::driver::sys::cuMemsetD8Async(ptr, 0, bytes, stream),
                "cuMemsetD8Async",
            )
        };
        zero_async(self.ear_r_major.ptr, self.k * self.r)?;
        zero_async(self.ebl_r_major.ptr, self.k * self.r)?;
        self.noise_gen.launch_sparse(
            self.k as i32,
            self.commit_a_pool[slot].ptr,
            self.seed_label_a.ptr,
            self.ear_r_major.ptr,
            stream,
        )?;
        self.noise_gen.launch_sparse(
            self.k as i32,
            self.commit_b_pool[slot].ptr,
            self.seed_label_b.ptr,
            self.ebl_r_major.ptr,
            stream,
        )?;
        self.noise_gen.launch_transpose(
            self.k as i32,
            self.r as i32,
            self.ear_r_major.ptr,
            self.ear_k_major.ptr,
            stream,
        )?;
        self.noise_gen.launch_transpose(
            self.k as i32,
            self.r as i32,
            self.ebl_r_major.ptr,
            self.ebl_k_major.ptr,
            stream,
        )?;

        // 5-8. Path branch — either the C++ noising+search+emit, or the
        // Triton no-rotl pipeline.
        if let Some(triton) = &self.triton {
            // -- Triton path --
            // 5. Triton noising A → ApEA, (m, k) = wrap_int8(A + EAL @ EAR_R^T)
            triton.noising.launch(
                self.m as i32,
                self.k as i32,
                self.r as i32,
                self.a_pool[slot].ptr,
                self.eal.ptr,
                self.ear_r_major.ptr,
                self.ap_ea.ptr,
                stream,
            )?;
            // 6. Triton noising B → BpEB
            triton.noising.launch(
                self.n as i32,
                self.k as i32,
                self.r as i32,
                self.b.ptr,
                self.ebr.ptr,
                self.ebl_r_major.ptr,
                self.bp_eb.ptr,
                stream,
            )?;
            // 7. Pre-zero transcripts (kernel only writes hit slots), then
            //    launch the Triton search kernel.
            zero_async(triton.transcripts.ptr, triton.transcripts.size)?;
            triton.search.launch(
                self.m as i32,
                self.n as i32,
                self.k as i32,
                self.ap_ea.ptr,
                self.bp_eb.ptr,
                triton.transcripts.ptr,
                stream,
            )?;
            // 8a. Postpass: Blake3+compare → hash+hit per candidate.
            let num_triton_tiles =
                (triton.num_triton_tile_m as i32) * (triton.num_triton_tile_n as i32);
            let total_candidates = num_triton_tiles * TRITON_HASH_CANDIDATES as i32;
            // Reset scan sentinel.
            crate::error::cu_check(
                cudarc::driver::sys::cuMemsetD32Async(
                    self.pow_workspace_scan.ptr,
                    0xFFFFFFFFu32,
                    1,
                    stream,
                ),
                "cuMemsetD32Async(pow_workspace_scan = 0xFFFFFFFF)",
            )?;
            triton.postpass.launch_blake3_compare(
                triton.transcripts.ptr,
                self.commit_a_pool[slot].ptr, // pow_key (u32 view)
                self.pow_target_tensor.ptr,
                self.pow_workspace_hash.ptr,
                self.pow_workspace_hit.ptr,
                total_candidates,
                stream,
            )?;
            // 8b. Scan hits.
            triton.postpass.launch_scan(
                self.pow_workspace_hit.ptr,
                total_candidates,
                self.pow_workspace_scan.ptr,
                stream,
            )?;
            // 8c. Emit (paired-pattern h=2, w=128).
            triton.postpass.launch_emit(
                self.pow_workspace_scan.ptr,
                self.pow_target_tensor.ptr,
                self.host_signal_header_pool[slot].device_ptr,
                triton.num_triton_tile_m,
                triton.num_triton_tile_n,
                TRITON_HASH_CANDIDATES as i32,
                self.m as i32,
                self.n as i32,
                self.k as i32,
                TRITON_BLOCK_M as i32,
                TRITON_BLOCK_N as i32,
                /*block_k=*/ 64,
                stream,
            )?;
        } else {
            // -- C++ search path (production) --
            // 5. Noising-A.
            self.noisy_gemm.launch_add_gemm(
                self.m as i32,
                self.k as i32,
                self.r as i32,
                self.a_pool[slot].ptr,
                self.eal.ptr,
                self.ear_r_major.ptr,
                self.ap_ea.ptr,
                stream,
            )?;
            // 6. Noising-B.
            self.noisy_gemm.launch_add_gemm(
                self.n as i32,
                self.k as i32,
                self.r as i32,
                self.b.ptr,
                self.ebr.ptr,
                self.ebl_r_major.ptr,
                self.bp_eb.ptr,
                stream,
            )?;
            // 7. Reset scan sentinel + launch C++ search.
            zero_async(
                self.pow_workspace_hit.ptr,
                self.num_tiles * THREADS_PER_TILE,
            )?;
            crate::error::cu_check(
                cudarc::driver::sys::cuMemsetD32Async(
                    self.pow_workspace_scan.ptr,
                    0xFFFFFFFFu32,
                    1,
                    stream,
                ),
                "cuMemsetD32Async(pow_workspace_scan = 0xFFFFFFFF)",
            )?;
            self.search.launch_r128(
                self.m as i32,
                self.n as i32,
                self.k as i32,
                self.ap_ea.ptr,
                self.bp_eb.ptr,
                self.commit_a_pool[slot].ptr,
                self.pow_target_tensor.ptr,
                self.pow_workspace_hash.ptr,
                self.pow_workspace_hit.ptr,
                /*transcript=*/ 0,
                stream,
            )?;
            // 8. Scan + emit.
            let total = (self.num_tiles * THREADS_PER_TILE) as i32;
            self.pow_scan_emit.launch_scan(
                self.pow_workspace_hit.ptr,
                total,
                self.pow_workspace_scan.ptr,
                stream,
            )?;
            let num_tile_m = (self.m / TILE_M) as i32;
            let num_tile_n = (self.n / TILE_N) as i32;
            self.pow_scan_emit.launch_emit(
                self.pow_workspace_scan.ptr,
                self.pow_target_tensor.ptr,
                self.host_signal_header_pool[slot].device_ptr,
                /*pinned_sync=*/ 0,
                num_tile_m,
                num_tile_n,
                THREADS_PER_TILE as i32,
                self.m as i32,
                self.n as i32,
                self.k as i32,
                stream,
            )?;
        }

        // 9. Per-slot D2H snapshots — NOT captured in the graph. Live
        //    miner calls [`snapshot_slot`] eagerly after each replay,
        //    so they run on the same stream but outside the captured
        //    region. Matches Python's main.py mine_one() which does
        //    them OUTSIDE the captured `_run_iter_body`.
        Ok(())
    }

    /// Async D2H copy of slot's A tensor (m*k bytes) onto `stream`. Issued
    /// ONLY when the host detects status=1 in the header — A is ~16 MB at
    /// default shape so per-iter unconditional snapshots add ~1 ms of PCIe
    /// time that the GPU pipeline can't hide. Hits are rare; this path is
    /// taken once per ~10s.
    ///
    /// Caller must `cuStreamSynchronize` before reading `a_snapshot_pool[slot]`.
    ///
    /// # Safety: `stream` must be valid.
    pub unsafe fn snapshot_a_for_hit(
        &mut self,
        slot: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<(), MinerError> {
        crate::error::cu_check(
            cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                self.a_snapshot_pool[slot].host_ptr,
                self.a_pool[slot].ptr,
                self.m * self.k,
                stream,
            ),
            "cuMemcpyDtoHAsync(A snapshot, on-hit)",
        )
    }

    /// Convenience: full per-iter sequence in eager mode. Equivalent to
    /// `zero_cpu(host_signal_header[slot]) + random_fill_a + mine_one_post_random`.
    ///
    /// # Safety
    /// `stream` must be a valid CUstream (or null). `ensure_for_job` must
    /// have populated B / B_tensor_hash / key / target / seed first.
    pub unsafe fn mine_one(
        &mut self,
        iter_idx: u64,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<(), MinerError> {
        let slot = self.slot(iter_idx);
        self.host_signal_header_pool[slot].zero_cpu();
        self.random_fill_a(iter_idx, slot, stream)?;
        self.mine_one_post_random(slot, stream)
    }

    /// Capture one [`CapturedGraph`] per ring slot by running
    /// `random_fill_a(slot)` followed by `mine_one_post_random(slot)` inside
    /// `cuStreamBeginCapture` / `cuStreamEndCapture`. Returns a
    /// `Vec<CapturedGraph>` indexed by slot.
    ///
    /// `random_int8_seeded` IS captured but its `iter_idx` arg is mutable
    /// per replay via [`crate::driver::CapturedGraph::launch_with_iter_idx`]
    /// (which calls `cuGraphExecKernelNodeSetParams`). The kernel-node
    /// handle is located post-instantiate by matching the `CUfunction`.
    ///
    /// # Safety
    /// `stream` must be a fresh non-default `Stream`. `ensure_for_job` must
    /// have populated per-job tensors before this call so the captured kernels
    /// reference valid bytes.
    pub unsafe fn capture_all_slots(
        &mut self,
        stream: &crate::driver::Stream,
    ) -> Result<Vec<crate::driver::CapturedGraph>, MinerError> {
        let mut graphs = Vec::with_capacity(self.ring_size);
        for slot in 0..self.ring_size {
            crate::driver::CapturedGraph::begin(stream)?;
            // random_fill_a is part of the captured graph. iter_idx is the
            // 3rd kernel arg; we'll mutate it per-replay below.
            self.random_fill_a(0, slot, stream.handle)?;
            self.mine_one_post_random(slot, stream.handle)?;
            let mut g = crate::driver::CapturedGraph::end(stream)?;
            // Locate the random_int8 node so launch_with_iter_idx can patch
            // it without re-capturing the whole graph.
            let total_bytes = (self.m * self.k) as i32;
            g.record_mutable_random_int8(
                self.random_int8.func(),
                total_bytes,
                self.seed_tensor.ptr,
                self.a_pool[slot].ptr,
                crate::kernels::random_int8::RandomInt8::grid_x(total_bytes),
                crate::kernels::random_int8::RandomInt8::BLOCK_X,
            )?;
            graphs.push(g);
        }
        Ok(graphs)
    }

    /// Replay-path mining iteration. `random_int8` is inside the captured
    /// graph; we mutate its `iter_idx` arg via cuGraphExecKernelNodeSetParams
    /// before each replay so the bytes are slot-fresh.
    ///
    /// # Safety
    /// `graphs` must come from [`Self::capture_all_slots`] on the same context.
    /// `stream` should be the same stream used during capture.
    pub unsafe fn mine_one_with_graphs(
        &mut self,
        iter_idx: u64,
        graphs: &mut [crate::driver::CapturedGraph],
        stream: &crate::driver::Stream,
    ) -> Result<(), MinerError> {
        let slot = self.slot(iter_idx);
        self.host_signal_header_pool[slot].zero_cpu();
        graphs[slot].launch_with_iter_idx(stream, iter_idx)
        // No per-iter D2H of A — that's deferred to `snapshot_a_for_hit`
        // (called only when status=1). At default shape the unconditional
        // snapshot was ~16 MB/iter of PCIe overhead the GPU couldn't hide.
    }

    /// Total bytes resident on device (rough estimate for diagnostics).
    pub fn approx_device_bytes(&self) -> usize {
        let per_slot = self.m * self.k                              // a_pool
            + 32 + 32 + 32; // hash + commits
        let ring_dev = self.ring_size * per_slot;
        let per_job = self.n * self.k + 32 + 32 + 32 + 32; // B + hash + key + target + seed
        let shared = self.m * 4 + self.n * 4                        // scales
            + self.m * self.r * 5                                    // eal + ear_r/k + eal_fp16 + a_e_bl(approx)
            + self.n * self.r * 4                                    // ebr + ebr_fp16 + earx
            + self.k * self.r * 4                                    // ear_r + ear_k + ebl_r + ebl_k
            + self.n * self.k * 2                                    // bp_eb + ap_ea (ap_ea = m*k actually)
            + self.num_tiles * THREADS_PER_TILE * (1 + 8 * 4)        // hit + hash workspaces
            + 4; // scan
        ring_dev + per_job + shared
    }
}
