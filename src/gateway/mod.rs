//! Pearl-node JSON-RPC client + block-template caching.
//!
//! Replaces the Python `pearl_gateway` process for the docker-image build.
//! The Rust miner talks directly to pearld via HTTPS JSON-RPC instead of
//! going through a UDS gateway. Saves one process and ~4 GB of Python
//! dependencies in the image.
//!
//! Two endpoints exposed:
//!
//! - [`PearldClient::get_block_template`] — fetches `getblocktemplate` and
//!   parses it into [`BlockTemplate`].
//! - [`PearldClient::submit_block`] — submits hex-encoded block bytes via
//!   `submitblock`; returns `Ok(())` on accept, `Err(rejected_reason)` on
//!   reject.
//!
//! HTTP transport is `ureq` (sync). Per-iter polling work happens on a
//! background thread; the main mining loop never blocks on the network.

pub mod block;
pub mod client;
pub mod mining_job;
pub mod rpc_types;

pub use block::{
    bits_to_target, calculate_merkle_root, create_coinbase_transaction, double_sha256,
    target_be_to_le, PearlBlock, PearlHeader, ZKCertificate,
};
pub use client::{PearldClient, PearldConfig};
pub use mining_job::{
    build_mining_config_cpp_search, build_mining_config_triton_norotl, MiningConfig, MiningJob,
};
pub use rpc_types::{BlockTemplateResponse, BlockTemplateTx};
