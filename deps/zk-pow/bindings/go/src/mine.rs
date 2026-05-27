//! Mining FFI - performs mining and returns a ZK proof directly.

use std::os::raw::c_char;
use std::slice;

use zk_pow::api::proof::{IncompleteBlockHeader, MiningConfiguration};
use zk_pow::api::prove;
use zk_pow::ffi::mine::mine as ffi_mine;

use crate::common::{acquire_cache, catch_panic, set_error_msg, CZKProof, MAX_ZK_PROOF_SIZE};

/// Perform mining and generate a ZK proof in one step.
///
/// Internally: mines a PlainProof, then generates a ZK proof from it.
///
/// # Returns
/// - 0: Mining and proof generation successful
/// - 1: Invalid input
/// - 2: System error
///
/// # Safety
/// - All pointers must be valid
/// - `zk_proof_out.proof_blob` must have capacity `MAX_ZK_PROOF_SIZE`
/// - `error_msg_out` must be null or a valid pointer to a caller-allocated buffer of `ERROR_MSG_MAX_SIZE` bytes
#[no_mangle]
pub unsafe extern "C" fn mine(
    m: u32,
    n: u32,
    block_header: *const IncompleteBlockHeader,
    mining_config: *const [u8; crate::common::MINING_CONFIG_SERIALIZED_SIZE],
    zk_proof_out: *mut CZKProof,
    error_msg_out: *mut c_char,
) -> i32 {
    if block_header.is_null() || mining_config.is_null() || zk_proof_out.is_null() {
        set_error_msg(error_msg_out, "Null pointer");
        return 2;
    }

    let config = match MiningConfiguration::from_bytes(&*mining_config) {
        Ok(c) => c,
        Err(e) => {
            set_error_msg(error_msg_out, &format!("Invalid mining config: {}", e));
            return 2;
        }
    };

    let k = config.common_dim as usize;
    let header = *block_header;

    let out = &mut *zk_proof_out;
    if out.proof_blob.is_null() {
        set_error_msg(error_msg_out, "proof_blob buffer is null");
        return 2;
    }

    // Step 1: Mine a PlainProof
    let plain_proof = match catch_panic(|| ffi_mine(m as usize, n as usize, k, header, config, None, false)) {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            set_error_msg(error_msg_out, &format!("Mining failed: {}", e));
            return 2;
        }
        Err(panic_msg) => {
            set_error_msg(error_msg_out, &format!("Mining panic: {}", panic_msg));
            return 2;
        }
    };

    // Step 2: Parse, prove, and serialize
    let mut cache = acquire_cache();
    let result = match catch_panic(|| prove::zk_prove_plain_proof(header, &plain_proof, &mut cache, false)) {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            set_error_msg(error_msg_out, &format!("Prove failed: {}", e));
            return 2;
        }
        Err(panic_msg) => {
            set_error_msg(error_msg_out, &format!("Prove panic: {}", panic_msg));
            return 2;
        }
    };

    out.public_data = result.public_data;

    let buffer = slice::from_raw_parts_mut(out.proof_blob, MAX_ZK_PROOF_SIZE);
    buffer[..result.proof_data.len()].copy_from_slice(&result.proof_data);
    out.proof_blob_len = result.proof_data.len();

    set_error_msg(error_msg_out, "Mining and proof generation successful");
    0
}
