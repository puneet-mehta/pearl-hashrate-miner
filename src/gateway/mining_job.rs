//! Per-job derived state.
//!
//! From a `BlockTemplateResponse` + coinbase tx we derive:
//!
//! - `header_bytes`: 76-byte canonical Pearl header (no proof commitment).
//! - `mining_config`: MMA type + (rows_pattern, cols_pattern) for the
//!   running search kernel.
//! - `key`: 32-byte Blake3 keying material (= blake3(header || cfg.to_bytes())).
//! - `target`: raw 256-bit target from the template's `bits` field.
//! - `adjusted_target`: `target * h * w * rounded_common_dim`, the value
//!   the kernel compares hashes against.
//!
//! Mirrors `MinerBufs.ensure_for_job` from main.py with USE_TRITON=0.
//! The Rust port uses the C++ search kernel pattern (rows=[0,8],
//! cols=[0,1,8,9,…,120,121]) — see `build_mining_config_cpp_search`.

use bitcoin::blockdata::transaction::Transaction;

use zk_pow::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern};

use crate::error::MinerError;
use crate::gateway::block::{bits_to_target, calculate_merkle_root, target_be_to_le};
use crate::gateway::rpc_types::BlockTemplateResponse;

/// Re-export so callers don't need to depend on zk-pow directly.
pub type MiningConfig = MiningConfiguration;

/// One mining job derived from a `getblocktemplate` response.
///
/// All the per-job device-bound state in [`crate::miner_bufs::MinerBufs::ensure_for_job`]
/// is derived from these fields.
pub struct MiningJob {
    /// 76 bytes ready to feed `IncompleteBlockHeader::from_bytes` and
    /// `blake3(header_bytes || config.to_bytes())`.
    pub header_bytes: [u8; 76],
    pub incomplete_header: IncompleteBlockHeader,
    pub mining_config: MiningConfig,
    /// 32-byte Blake3 keying material for tensor_hash + commitment_hash.
    pub key: [u8; 32],
    /// 32-byte raw target from `bits` (big-endian 256-bit int as written
    /// in the block header). Useful for sanity logging.
    pub target_be: [u8; 32],
    /// 32-byte adjusted target as the kernel needs it (little-endian).
    /// `adjusted_target_le = target_be * H * W * rounded_common_dim`,
    /// converted to LE byte order.
    pub adjusted_target_le: [u8; 32],

    /// Block template (kept so the hit callback can rebuild the block).
    pub template: BlockTemplateResponse,
    /// Coinbase transaction the miner committed to in the merkle root.
    pub coinbase_tx: Transaction,
    /// Non-coinbase txs from the template, in template order.
    pub other_txs: Vec<Transaction>,
}

impl MiningJob {
    /// Construct a fresh job. Builds the merkle root over (coinbase +
    /// other_txs) and assembles the canonical incomplete header.
    pub fn build(
        template: BlockTemplateResponse,
        coinbase_tx: Transaction,
        other_txs: Vec<Transaction>,
        mining_config: MiningConfig,
    ) -> Result<Self, MinerError> {
        // Merkle root over [coinbase, other_txs...].
        let mut all = Vec::with_capacity(1 + other_txs.len());
        all.push(coinbase_tx.clone());
        all.extend(other_txs.iter().cloned());
        let merkle_root_be = calculate_merkle_root(&all);

        // prev_block_hash from hex (display order, big-endian).
        let prev_block_be = hex_to_array32(&template.previousblockhash)?;
        let nbits = u32::from_str_radix(&template.bits, 16).map_err(|e| MinerError::Rpc {
            method: "build_job".to_string(),
            msg: format!("bad bits {}: {}", template.bits, e),
        })?;

        let incomplete_header = IncompleteBlockHeader {
            version: template.version as u32,
            prev_block: prev_block_be,
            merkle_root: merkle_root_be,
            timestamp: template.curtime as u32,
            nbits,
        };
        let header_bytes = incomplete_header.to_bytes();

        // key = blake3(header_bytes || mining_config.to_bytes())
        let mut keying = Vec::with_capacity(76 + 52);
        keying.extend_from_slice(&header_bytes);
        keying.extend_from_slice(&mining_config.to_bytes());
        let key = *blake3::hash(&keying).as_bytes();

        // Target derivation
        let target_be = bits_to_target(nbits);
        let h = mining_config.rows_pattern.to_list().len() as u128;
        let w = mining_config.cols_pattern.to_list().len() as u128;
        let rounded_k = mining_config.dot_product_length() as u128;
        let adjustment = h
            .checked_mul(w)
            .and_then(|x| x.checked_mul(rounded_k))
            .ok_or_else(|| MinerError::Rpc {
                method: "build_job".to_string(),
                msg: format!("difficulty adjustment overflow h={h} w={w} k={rounded_k}"),
            })?;
        let adjusted_be = mul_u256_u128(&target_be, adjustment)?;
        let adjusted_target_le = target_be_to_le(&adjusted_be);

        Ok(MiningJob {
            header_bytes,
            incomplete_header,
            mining_config,
            key,
            target_be,
            adjusted_target_le,
            template,
            coinbase_tx,
            other_txs,
        })
    }
}

// -----------------------------------------------------------------------------
//   Mining config: C++ search kernel pattern
// -----------------------------------------------------------------------------

/// The sm_80 C++ `pearl_gemm_search_perthread_smem_pipelined_kernel<R>`
/// uses an 8-warp × 1-warp CTA with TILE_M=128, TILE_N=128. Per thread
/// it works on h=2 unique rows × w=32 unique cols, arranged in pairs:
///
///   rows = [0, 8]                    # stride 8, length 2
///   cols = [0, 1, 8, 9, 16, 17, ..., 120, 121]   # 16 pairs, pair-stride 8, intra-pair stride 1
///
/// This is the pattern that MUST match what the kernel actually emits
/// from `host_signal_header` and what the verifier reconstructs from the
/// submitted MiningConfiguration. Diverging the two = JACKPOT bug
/// the noise commitment chain.
pub fn build_mining_config_cpp_search(k: u32, noise_rank: u16) -> Result<MiningConfig, MinerError> {
    let rows: Vec<u32> = vec![0, 8];
    let mut cols: Vec<u32> = Vec::with_capacity(32);
    for pair in 0..16 {
        cols.push(pair * 8);
        cols.push(pair * 8 + 1);
    }
    let rows_pattern = PeriodicPattern::from_list(&rows).map_err(|e| MinerError::Rpc {
        method: "mining_config".to_string(),
        msg: format!("rows pattern: {e}"),
    })?;
    let cols_pattern = PeriodicPattern::from_list(&cols).map_err(|e| MinerError::Rpc {
        method: "mining_config".to_string(),
        msg: format!("cols pattern: {e}"),
    })?;
    Ok(MiningConfig {
        common_dim: k,
        rank: noise_rank,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern,
        cols_pattern,
        reserved: MiningConfiguration::RESERVED_VALUE,
    })
}

/// Triton no-rotl pattern: `rows=[0,1]`, `cols=range(128)`.
/// (h=2, w=128) — what the Triton paired search kernel emits per
/// candidate. Verifier-equivalent for the Triton path only.
pub fn build_mining_config_triton_norotl(
    k: u32,
    noise_rank: u16,
) -> Result<MiningConfig, MinerError> {
    let rows: Vec<u32> = vec![0, 1];
    let cols: Vec<u32> = (0..128).collect();
    let rows_pattern = PeriodicPattern::from_list(&rows).map_err(|e| MinerError::Rpc {
        method: "mining_config_triton".to_string(),
        msg: format!("rows pattern: {e}"),
    })?;
    let cols_pattern = PeriodicPattern::from_list(&cols).map_err(|e| MinerError::Rpc {
        method: "mining_config_triton".to_string(),
        msg: format!("cols pattern: {e}"),
    })?;
    Ok(MiningConfig {
        common_dim: k,
        rank: noise_rank,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern,
        cols_pattern,
        reserved: MiningConfiguration::RESERVED_VALUE,
    })
}

// -----------------------------------------------------------------------------
//   Internal helpers
// -----------------------------------------------------------------------------

fn hex_to_array32(s: &str) -> Result<[u8; 32], MinerError> {
    let bytes = hex::decode(s).map_err(|e| MinerError::Rpc {
        method: "build_job".to_string(),
        msg: format!("hex decode {s}: {e}"),
    })?;
    if bytes.len() != 32 {
        return Err(MinerError::Rpc {
            method: "build_job".to_string(),
            msg: format!("expected 32 bytes, got {}", bytes.len()),
        });
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

/// 256-bit big-endian × 128-bit. Errors on overflow.
fn mul_u256_u128(a_be: &[u8; 32], b: u128) -> Result<[u8; 32], MinerError> {
    // Schoolbook: a is stored in big-endian; multiply low-to-high in
    // little-endian form to make carries easy.
    let mut a_le = *a_be;
    a_le.reverse();
    let mut out = [0u128; 33];
    for i in 0..32 {
        let prod = (a_le[i] as u128) * b;
        out[i] += prod & 0xff;
        // Spread the upper bits forward.
        let mut carry = prod >> 8;
        let mut j = i + 1;
        while carry > 0 && j < out.len() {
            let s = out[j] + (carry & 0xff);
            out[j] = s & 0xff;
            carry = (carry >> 8) + (s >> 8);
            j += 1;
        }
    }
    // Carry propagation in `out[0..32]` may have left non-byte values; tidy up.
    let mut carry: u128 = 0;
    let mut le = [0u8; 32];
    for i in 0..32 {
        let s = out[i] + carry;
        le[i] = (s & 0xff) as u8;
        carry = s >> 8;
    }
    // out[32] holds any final overflow, plus carry leftover.
    let overflow = out[32] + carry;
    if overflow > 0 {
        return Err(MinerError::Rpc {
            method: "build_job".to_string(),
            msg: "adjusted target overflow (> 2^256)".to_string(),
        });
    }
    let mut be = le;
    be.reverse();
    Ok(be)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpp_pattern_round_trip() {
        let cfg = build_mining_config_cpp_search(4096, 128).unwrap();
        let rows = cfg.rows_pattern.to_list();
        assert_eq!(rows, vec![0, 8]);
        let cols = cfg.cols_pattern.to_list();
        let expected: Vec<u32> = (0..16).flat_map(|p| [p * 8, p * 8 + 1]).collect();
        assert_eq!(cols, expected);

        // 52-byte serialize must round-trip via from_bytes.
        let bytes = cfg.to_bytes();
        let parsed = MiningConfiguration::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.common_dim, 4096);
        assert_eq!(parsed.rank, 128);
        assert_eq!(parsed.rows_pattern.to_list(), vec![0, 8]);
    }

    #[test]
    fn mul_u256_u128_simple() {
        // (1u256) * 7 = 7
        let mut a = [0u8; 32];
        a[31] = 1;
        let p = mul_u256_u128(&a, 7).unwrap();
        let mut expected = [0u8; 32];
        expected[31] = 7;
        assert_eq!(p, expected);
    }
}
