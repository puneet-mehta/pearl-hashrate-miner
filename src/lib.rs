//! All-Rust Pearl PoW miner.
//!
//! Drives the sequence of CUDA kernels (Blake3 hashing, random fill, noise
//! generation, noisy GEMM, search, scan+emit) directly via cudart, with no
//! PyTorch dispatch layer in the path. Kernels are loaded from a multi-arch
//! fatbin built from `csrc/` and SASS-resident at runtime.
//!
//! ## Module layout
//!
//! - [`error`] — single `MinerError` for all failure modes.
//! - [`driver`] — thin safe wrappers around raw `cudarc::driver::sys` calls
//!   (`cuModuleLoadFatBinary`, `cuLaunchKernel` with arbitrary args, etc.).
//! - [`fatbin`] — opens the pearl-gemm fatbin and lets callers grab functions
//!   by stable extern-C name (e.g. `pearl_commitment_hash_kernel`).
//! - [`kernels`] — one submodule per kernel (`commitment_hash`, `tensor_hash`,
//!   `noise_gen`, `noisy_gemm`, `search`, `pow_scan_emit`, `triton`). Each
//!   exposes a typed Rust launcher.
//! - [`miner_bufs`] — `MinerBufs`: per-device ring buffer, kernel handles,
//!   captured graphs.
//! - [`gateway`] — pearld JSON-RPC client + `MiningJob` builder.
//! - [`proof`] — `PlainProof` builder and signal-header parser.

#[cfg(feature = "cuda")]
pub mod driver;
pub mod error;
#[cfg(feature = "cuda")]
pub mod fatbin;
pub mod gateway;
#[cfg(feature = "cuda")]
pub mod kernels;
#[cfg(feature = "cuda")]
pub mod miner_bufs;
pub mod proof;

pub use error::MinerError;
#[cfg(feature = "cuda")]
pub use miner_bufs::{MinerBufs, MinerBufsConfig};
