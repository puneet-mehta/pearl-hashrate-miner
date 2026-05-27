//! Hit → submission path: header decode → PlainProof → ZK prove → PearlBlock.
//!
//! Mirrors the Python `StatusCheckCallback → SubmissionService` flow but
//! runs entirely in the Rust miner process (no UDS hop, no gateway).
//!
//! Code lives under `proof/` so it can be unit-tested without CUDA
//! (cfg-gated `cuda` feature controls only the GPU-touching bits).

pub mod plain_proof;
pub mod signal_header;
pub mod submit;

pub use plain_proof::build_plain_proof;
pub use signal_header::{extract_indices, ParsedSignalHeader};
pub use submit::submit_plain_proof;
