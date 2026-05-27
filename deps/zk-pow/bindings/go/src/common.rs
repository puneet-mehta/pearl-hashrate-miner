//! C-compatible structs and utilities for Go FFI.

use anyhow::Result;
use std::os::raw::c_char;
use std::panic::AssertUnwindSafe;
use std::slice;
use std::sync::Mutex;

use zk_pow::api::proof::{MiningConfiguration, PublicProofParams};
use zk_pow::circuit::pearl_circuit::{PearlRecursion, RecursionCircuit};

/// Size of reserved field in MiningConfiguration (exported to C header).
pub const MINING_CONFIG_RESERVED_SIZE: usize = 32;

/// Size of serialized MiningConfiguration in bytes (exported to C header).
/// Note: IncompleteBlockHeader (76) + MiningConfiguration (52) = 128 bytes = 2 blake3 blocks.
pub const MINING_CONFIG_SERIALIZED_SIZE: usize = 52;

/// Maximum size of the error message buffer passed from Go (exported to C header).
pub const ERROR_MSG_MAX_SIZE: usize = 128;

/// Maximum size of a serialized ZK proof blob (excluding IncompleteBlockHeader and MiningConfiguration, including everything else).
pub const MAX_ZK_PROOF_SIZE: usize = 60000;

// Compile-time assertions to ensure constants stay in sync
const _: () = assert!(MINING_CONFIG_RESERVED_SIZE == MiningConfiguration::RESERVED_SIZE);
const _: () = assert!(MINING_CONFIG_SERIALIZED_SIZE == MiningConfiguration::SERIALIZED_SIZE);

type CircuitCache = <PearlRecursion as RecursionCircuit>::CircuitCache;

lazy_static::lazy_static! {
    /// Global circuit cache shared across Go FFI functions (verify and prove).
    /// Protected by a Mutex for thread-safe access from multiple Go goroutines.
    pub static ref CIRCUIT_CACHE: Mutex<CircuitCache> = {
        use zk_pow::circuit::embedded_cache;
        Mutex::new(CircuitCache::from_bytes(embedded_cache::CACHE_DATA).unwrap_or_default())
    };
}

/// Acquires the circuit cache. Recovers from poisoned mutex if a prior panic occurred.
/// The cache data is still valid for verifier after a panic, since the CircuitCache is read only.
pub(crate) fn acquire_cache() -> std::sync::MutexGuard<'static, CircuitCache> {
    CIRCUIT_CACHE.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Catches panics from a closure and returns Ok(result) or Err(panic_message).
/// The closure is wrapped in AssertUnwindSafe internally.
pub(crate) fn catch_panic<F, R>(f: F) -> Result<R>
where
    F: FnOnce() -> R,
{
    std::panic::catch_unwind(AssertUnwindSafe(f)).map_err(|e| {
        let msg = e
            .downcast::<String>()
            .map(|s| *s)
            .or_else(|e| e.downcast::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|_| "Unknown panic".to_string());
        let first_line = msg.lines().next().unwrap_or(&msg).to_string();
        anyhow::anyhow!(first_line)
    })
}

/// Size of the committed public data in bytes (exported to C header).
pub const PUBLICDATA_SIZE: usize = 164;
const _: () = assert!(PUBLICDATA_SIZE == PublicProofParams::PUBLICDATA_SIZE);

/// Go-owned ZK proof structure.
#[repr(C)]
pub struct CZKProof {
    pub public_data: [u8; PUBLICDATA_SIZE],
    pub proof_blob_len: usize,
    pub proof_blob: *mut u8,
}

/// Writes an error message into a caller-allocated buffer of ERROR_MSG_MAX_SIZE bytes.
/// The message is always null-terminated. Truncation respects UTF-8 char boundaries.
/// # Safety
/// `out` must be null or a valid pointer to a buffer of at least `ERROR_MSG_MAX_SIZE` bytes.
pub(crate) unsafe fn set_error_msg(out: *mut c_char, msg: &str) {
    if out.is_null() {
        return;
    }
    let buf = slice::from_raw_parts_mut(out as *mut u8, ERROR_MSG_MAX_SIZE);
    // Truncate at a UTF-8 char boundary that fits in ERROR_MSG_MAX_SIZE-1 bytes (reserve 1 for null)
    let max_len = ERROR_MSG_MAX_SIZE - 1;
    let mut end = msg.len().min(max_len);
    while end > 0 && !msg.is_char_boundary(end) {
        end -= 1;
    }
    buf[..end].copy_from_slice(&msg.as_bytes()[..end]);
    buf[end] = 0;
}
