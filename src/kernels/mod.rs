//! Typed Rust launchers for pearl-gemm CUDA kernels.
//!
//! Each submodule corresponds to one Python `pearl_gemm.*` call site that the
//! original miner used (e.g. `pearl_gemm.commitment_hash_from_merkle_roots`
//! → [`commitment_hash`]).
//!
//! All launchers take pre-allocated device buffers (callers are expected to
//! reuse them across iters via the ring buffer / `MinerBufs` analogue —
//! that's a future commit).

pub mod commitment_hash;
pub mod noise_gen;
pub mod noisy_gemm;
pub mod pow_scan_emit;
pub mod random_int8;
pub mod search;
pub mod tensor_hash;
pub mod triton;
