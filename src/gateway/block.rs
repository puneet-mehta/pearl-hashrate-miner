//! Pearl block construction.
//!
//! Mirrors:
//! - `miner/pearl-gateway/src/pearl_gateway/blockchain_utils/blockchain_utils.py`
//! - `miner/pearl-gateway/src/pearl_gateway/blockchain_utils/pearl_header.py`
//! - `miner/pearl-gateway/src/pearl_gateway/blockchain_utils/zk_certificate.py`
//! - `miner/pearl-gateway/src/pearl_gateway/blockchain_utils/pearl_block.py`
//!
//! A Pearl block on the wire is:
//!
//!   ZK_CERTIFICATE | PEARL_HEADER (76 + 32 = 108 B) | TX_COUNT (varint) | TX_BYTES
//!
//! Pearl extends bitcoin's block: an additional `proof_commitment` field in
//! the header (32 B) and a leading `ZKCertificate` blob carrying the SNARK.

use bitcoin::absolute::LockTime;
use bitcoin::blockdata::script::{Builder, ScriptBuf};
use bitcoin::blockdata::transaction::{OutPoint, Sequence, Transaction, TxIn, TxOut, Version};
use bitcoin::blockdata::witness::Witness;
use bitcoin::consensus::Encodable;
use bitcoin::hashes::{sha256d, Hash};
use bitcoin::opcodes::all::OP_RETURN;
use bitcoin::script::PushBytesBuf;
use bitcoin::{Amount, Txid};

use zk_pow::api::proof::{IncompleteBlockHeader, PublicProofParams};

use crate::error::MinerError;

/// Pearl-specific extension to bitcoin's block header.
pub const PROOF_COMMITMENT_SIZE: usize = 32;
pub const PEARL_HEADER_SIZE: usize = 76 + PROOF_COMMITMENT_SIZE;

/// Pearl block header. Wraps `IncompleteBlockHeader` (the 76 B canonical
/// portion, identical to bitcoin) and adds `proof_commitment` (32 B).
#[derive(Debug, Clone)]
pub struct PearlHeader {
    pub incomplete: IncompleteBlockHeader,
    pub proof_commitment: Option<[u8; PROOF_COMMITMENT_SIZE]>,
}

impl PearlHeader {
    pub fn serialize_without_proof_commitment(&self) -> [u8; 76] {
        self.incomplete.to_bytes()
    }

    /// Full 108 B header. Panics if `proof_commitment` is `None`.
    pub fn serialize(&self) -> [u8; PEARL_HEADER_SIZE] {
        let inc = self.serialize_without_proof_commitment();
        let pc = self.proof_commitment.expect("proof_commitment not set");
        let mut out = [0u8; PEARL_HEADER_SIZE];
        out[..76].copy_from_slice(&inc);
        out[76..].copy_from_slice(&pc);
        out
    }
}

/// Wraps a ZK proof + binds it to the block via `header_hash`.
/// On-wire layout (mirrors numpy struct from Python):
///
///   version       u32 little-endian   = 1 (`ZK_CERTIFICATE_VERSION`)
///   header_hash   [u8; 32]            = double_sha256(pearl_header.serialize())
///   public_data   [u8; PUBLICDATA]    = 164 bytes
///   proof_data_len u32 little-endian
///   proof_data    Vec<u8>             (proof_data_len bytes)
pub struct ZKCertificate {
    pub header_hash: [u8; 32],
    pub public_data: Vec<u8>,
    pub proof_data: Vec<u8>,
}

pub const ZK_CERTIFICATE_VERSION: u32 = 1;
pub const ZK_MAX_PROOF_DATA_SIZE: usize = 60_000;

impl ZKCertificate {
    pub fn serialize(&self) -> Vec<u8> {
        assert_eq!(self.public_data.len(), PublicProofParams::PUBLICDATA_SIZE);
        assert!(self.proof_data.len() <= ZK_MAX_PROOF_DATA_SIZE);

        let mut out = Vec::with_capacity(
            4 + 32 + PublicProofParams::PUBLICDATA_SIZE + 4 + self.proof_data.len(),
        );
        out.extend_from_slice(&ZK_CERTIFICATE_VERSION.to_le_bytes());
        out.extend_from_slice(&self.header_hash);
        out.extend_from_slice(&self.public_data);
        out.extend_from_slice(&(self.proof_data.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.proof_data);
        out
    }

    /// proof_commitment = double_sha256(version_le4 || public_data).
    /// Computed pre-cert (no header_hash dependency) so it can populate the
    /// header's `proof_commitment` before we hash the header.
    pub fn compute_proof_commitment(public_data: &[u8]) -> [u8; 32] {
        let mut buf = Vec::with_capacity(4 + public_data.len());
        buf.extend_from_slice(&ZK_CERTIFICATE_VERSION.to_le_bytes());
        buf.extend_from_slice(public_data);
        double_sha256(&buf)
    }
}

/// Pearl block: ZKCertificate | PearlHeader | tx_count varint | tx bytes.
pub struct PearlBlock {
    pub header: PearlHeader,
    /// Transactions in submission order (coinbase first).
    pub txns: Vec<Transaction>,
    pub zk_certificate: ZKCertificate,
}

impl PearlBlock {
    /// Serialize for `submitblock` RPC. Returns the byte stream which the
    /// caller hex-encodes.
    pub fn serialize(&self) -> Result<Vec<u8>, MinerError> {
        let mut out = self.zk_certificate.serialize();
        out.extend_from_slice(&self.header.serialize());

        // CompactSize tx count.
        encode_varint(self.txns.len() as u64, &mut out);

        // Each tx serialized in its canonical (segwit-aware) form.
        for tx in &self.txns {
            tx.consensus_encode(&mut out).map_err(|e| MinerError::Rpc {
                method: "tx_serialize".to_string(),
                msg: format!("tx encode: {e}"),
            })?;
        }
        Ok(out)
    }
}

// -----------------------------------------------------------------------------
//   Helpers
// -----------------------------------------------------------------------------

pub fn double_sha256(data: &[u8]) -> [u8; 32] {
    sha256d::Hash::hash(data).to_byte_array()
}

pub fn bits_to_target(bits: u32) -> [u8; 32] {
    let exponent = (bits >> 24) & 0xff;
    let mantissa: u64 = (bits & 0x00_ff_ff_ff) as u64;
    // target = mantissa * 2^(8*(exponent-3))
    // Build as 256-bit big-endian then reverse.
    let mut be = [0u8; 32];
    if exponent >= 3 {
        let shift_bytes = (exponent - 3) as usize;
        let mantissa_be = mantissa.to_be_bytes(); // 8 bytes, leading zeros
                                                  // mantissa fits in 24 bits → only last 3 bytes are non-zero
        let mantissa3 = &mantissa_be[5..]; // 3 bytes
        let start = 32 - 3 - shift_bytes;
        if start <= 32 - 3 {
            be[start..start + 3].copy_from_slice(mantissa3);
        }
    } else {
        // exponent < 3 means the mantissa is shifted right — unused in
        // practice but handle for correctness.
        let mantissa_shifted = mantissa >> (8 * (3 - exponent));
        be[24..].copy_from_slice(&mantissa_shifted.to_be_bytes());
    }
    be
}

/// Convert `target` (256-bit big-endian) to little-endian u256 for the
/// PoW comparison the kernel does.
pub fn target_be_to_le(target_be: &[u8; 32]) -> [u8; 32] {
    let mut le = *target_be;
    le.reverse();
    le
}

/// Bitcoin CompactSize varint.
pub fn encode_varint(n: u64, out: &mut Vec<u8>) {
    if n < 0xfd {
        out.push(n as u8);
    } else if n <= 0xffff {
        out.push(0xfd);
        out.extend_from_slice(&(n as u16).to_le_bytes());
    } else if n <= 0xffff_ffff {
        out.push(0xfe);
        out.extend_from_slice(&(n as u32).to_le_bytes());
    } else {
        out.push(0xff);
        out.extend_from_slice(&n.to_le_bytes());
    }
}

// -----------------------------------------------------------------------------
//   Merkle root
// -----------------------------------------------------------------------------

/// Bitcoin-style transaction Merkle root.
///
/// - Input txids are `little-endian` (Bitcoin internal byte order; reverse
///   of `Txid::to_string()`).
/// - Pairs are concatenated and double-SHA-256'd.
/// - Odd levels duplicate the last hash.
/// - The result is reversed at the end to be the **display-order**
///   merkle root that goes into the block header.
pub fn calculate_merkle_root(txs: &[Transaction]) -> [u8; 32] {
    assert!(
        !txs.is_empty(),
        "calculate_merkle_root requires at least one tx"
    );

    // Get raw little-endian txids.
    let mut level: Vec<[u8; 32]> = txs
        .iter()
        .map(|t| t.compute_txid().to_byte_array())
        .collect();

    if level.len() == 1 {
        let mut r = level[0];
        r.reverse();
        return r;
    }

    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let n = level.len();
        let mut i = 0;
        while i < n {
            let left = level[i];
            let right = if i + 1 < n { level[i + 1] } else { left };
            let mut buf = [0u8; 64];
            buf[..32].copy_from_slice(&left);
            buf[32..].copy_from_slice(&right);
            next.push(double_sha256(&buf));
            i += 2;
        }
        level = next;
    }
    let mut r = level[0];
    r.reverse();
    r
}

// -----------------------------------------------------------------------------
//   Coinbase tx
// -----------------------------------------------------------------------------

/// Construct a Pearl coinbase transaction.
///
/// Mirrors Python's `create_coinbase_transaction`:
/// - BIP34: block height encoded as the first script-number push in scriptSig.
/// - Followed by `0x00` extra-nonce byte (matches pearld's behavior).
/// - Followed by the `coinbaseaux.flags` blob from getblocktemplate.
/// - Output 0: P2TR pay-to-mining-address.
/// - Output 1 (only if `default_witness_commitment` set): OP_RETURN
///   `aa21a9ed || commitment`, segwit-marker.
///
/// `mining_address` is a bech32m Taproot address. We use rust-bitcoin's
/// address parser to validate + extract the witness program.
pub fn create_coinbase_transaction(
    height: u64,
    coinbase_value: u64,
    mining_address: &str,
    coinbase_aux_flags: Option<&[u8]>,
    default_witness_commitment: Option<&[u8; 32]>,
) -> Result<Transaction, MinerError> {
    // ScriptSig is RAW script bytes (no outer push wrapper):
    //   <height as scriptnum push> 0x00 [aux_flags…]
    //
    // Python's gateway emits `Script([raw_hex_str])` which bitcoinutils
    // interprets as raw script bytes (not as data to push). pearld's
    // BIP34 check then sees OP_11 (for height ≤ 16) as the first opcode
    // and decodes the scriptnum correctly.
    let mut scriptsig_bytes: Vec<u8> = Vec::new();
    push_script_number(height as i64, &mut scriptsig_bytes);
    scriptsig_bytes.push(0x00);
    if let Some(flags) = coinbase_aux_flags {
        scriptsig_bytes.extend_from_slice(flags);
    }

    let prevout = OutPoint {
        txid: Txid::all_zeros(),
        vout: 0xffff_ffff,
    };
    let mut tx_in = TxIn {
        previous_output: prevout,
        script_sig: ScriptBuf::from_bytes(scriptsig_bytes),
        sequence: Sequence::MAX,
        witness: Witness::new(),
    };

    // P2TR scriptPubKey. Pearl uses non-standard HRPs (rprl/tprl/mprl)
    // that rust-bitcoin's Address parser rejects; decode bech32m
    // manually.
    let script_pubkey = p2tr_script_pubkey_from_address(mining_address)?;
    let outputs = {
        let mut v = vec![TxOut {
            value: Amount::from_sat(coinbase_value),
            script_pubkey,
        }];
        if let Some(commitment) = default_witness_commitment {
            // OP_RETURN aa21a9ed||commitment (38 bytes)
            let mut data = Vec::with_capacity(4 + 32);
            data.extend_from_slice(&[0xaa, 0x21, 0xa9, 0xed]);
            data.extend_from_slice(commitment);
            let pb = PushBytesBuf::try_from(data).expect("36 bytes fits PushBytes");
            let script = Builder::new()
                .push_opcode(OP_RETURN)
                .push_slice(&pb)
                .into_script();
            v.push(TxOut {
                value: Amount::ZERO,
                script_pubkey: script,
            });
            // Witness: 32 zero bytes (the canonical empty-segwit witness
            // reservation matching what Python writes).
            tx_in.witness.push(vec![0u8; 32]);
        }
        v
    };

    let tx = Transaction {
        version: Version(1),
        lock_time: LockTime::ZERO,
        input: vec![tx_in],
        output: outputs,
    };
    Ok(tx)
}

/// Decode a Pearl Taproot bech32m address (HRP can be `rprl`, `tprl`,
/// `mprl`, etc.) and return its P2TR scriptPubKey:
///   `OP_1 (0x51) | PUSH32 (0x20) | <32-byte witness program>`.
///
/// Validates witness_version == 1 (Taproot) and program length == 32.
fn p2tr_script_pubkey_from_address(addr: &str) -> Result<ScriptBuf, MinerError> {
    use bech32::primitives::decode::CheckedHrpstring;
    use bech32::primitives::iter::Fe32IterExt;
    use bech32::Bech32m;

    let checked = CheckedHrpstring::new::<Bech32m>(addr).map_err(|e| MinerError::Rpc {
        method: "address".to_string(),
        msg: format!("bech32m decode of '{addr}': {e}"),
    })?;
    // The first field element (5 bits) is the witness version; the rest are
    // the witness-program 5-bit groups that get unpacked into 8-bit bytes.
    // `byte_iter()` does that conversion on EVERYTHING — wrong for our use.
    // Take the first Fe32 as version, then convert the remainder.
    // `fe32_iter`'s type param is unused; pass any Iterator<Item=u8>.
    let mut fes = checked.fe32_iter::<std::iter::Empty<u8>>();
    let witness_version_fe = fes.next().ok_or_else(|| MinerError::Rpc {
        method: "address".to_string(),
        msg: "empty data".to_string(),
    })?;
    let witness_version: u8 = witness_version_fe.to_u8();
    if witness_version != 1 {
        return Err(MinerError::Rpc {
            method: "address".to_string(),
            msg: format!("expected witness version 1 (Taproot), got {witness_version}"),
        });
    }
    let program: Vec<u8> = fes.fes_to_bytes().collect();
    if program.len() != 32 {
        return Err(MinerError::Rpc {
            method: "address".to_string(),
            msg: format!("Taproot program must be 32 bytes, got {}", program.len()),
        });
    }
    let mut script = Vec::with_capacity(34);
    script.push(0x51); // OP_1
    script.push(0x20); // PUSHBYTES_32
    script.extend_from_slice(&program);
    Ok(ScriptBuf::from_bytes(script))
}

/// Push a script number (encoded as the minimal bitcoin scriptnum) onto
/// `out`. This is the BIP34 height encoding.
fn push_script_number(n: i64, out: &mut Vec<u8>) {
    if n == 0 {
        out.push(0); // OP_0
        return;
    }
    // OP_1..OP_16 short forms.
    if (1..=16).contains(&n) {
        out.push(0x50 + n as u8);
        return;
    }
    // Otherwise encode as little-endian magnitude with explicit sign bit.
    let mut absn = n.unsigned_abs();
    let mut buf = Vec::new();
    while absn != 0 {
        buf.push((absn & 0xff) as u8);
        absn >>= 8;
    }
    if buf.last().map(|b| b & 0x80 != 0).unwrap_or(false) {
        buf.push(if n < 0 { 0x80 } else { 0x00 });
    } else if n < 0 {
        let last = buf.last_mut().unwrap();
        *last |= 0x80;
    }
    push_data(&buf, out);
}

/// Push `data` as a minimal-encoding push opcode + bytes (OP_PUSHBYTES_n
/// for small, OP_PUSHDATA1/2/4 for larger).
fn push_data(data: &[u8], out: &mut Vec<u8>) {
    let n = data.len();
    if n <= 75 {
        out.push(n as u8);
    } else if n <= 0xff {
        out.push(0x4c); // OP_PUSHDATA1
        out.push(n as u8);
    } else if n <= 0xffff {
        out.push(0x4d);
        out.extend_from_slice(&(n as u16).to_le_bytes());
    } else {
        out.push(0x4e);
        out.extend_from_slice(&(n as u32).to_le_bytes());
    }
    out.extend_from_slice(data);
}

// -----------------------------------------------------------------------------
//   Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrip() {
        let cases = [
            0u64,
            1,
            252,
            253,
            254,
            0xffff,
            0x10000,
            0xffff_ffff,
            0x1_0000_0000,
        ];
        for c in cases {
            let mut out = Vec::new();
            encode_varint(c, &mut out);
            // Spot-check the length matches the spec
            let expected_len = if c < 0xfd {
                1
            } else if c <= 0xffff {
                3
            } else if c <= 0xffff_ffff {
                5
            } else {
                9
            };
            assert_eq!(out.len(), expected_len, "varint len mismatch for {c}");
        }
    }

    #[test]
    fn bits_to_target_examples() {
        // bitcoin genesis bits 0x1d00ffff -> target = 00...00ffff << (8*(0x1d-3))
        let target = bits_to_target(0x1d00_ffff);
        // Last 4 bytes should be `ff ff 00 00` ... wait. Easier check:
        // top byte of bits is 0x1d=29, mantissa=0x00ffff. target =
        // 0x00ffff * 2^(8*26) which has its highest non-zero byte at index
        // (32 - 1 - 26) = 5; we expect target[4]=0xff target[5]=0xff target[6]=0x00.
        assert_eq!(target[4], 0xff);
        assert_eq!(target[5], 0xff);
        assert_eq!(target[6], 0x00);

        // The Pearl simnet bits we just saw: 0x1e010000.
        // exponent=0x1e=30, mantissa=0x010000.
        // target = 0x010000 * 2^(8*27) → highest bit at idx (32-1-27)=4, value 0x01.
        let target = bits_to_target(0x1e01_0000);
        assert_eq!(target[2], 0x01);
        assert_eq!(target[3], 0x00);
    }

    #[test]
    fn p2tr_script_pubkey_simnet_address() {
        // Known simnet mining address (HRP=rprl, witness v1, 32 B program).
        let addr = "rprl1p94k8ffwc4ufn78r9cz5ln8zrxjvdeqraecpzu4vuvz36wrszy04qtcg0d2";
        let script = p2tr_script_pubkey_from_address(addr).unwrap();
        let bytes = script.as_bytes();
        assert_eq!(bytes.len(), 34);
        assert_eq!(bytes[0], 0x51); // OP_1
        assert_eq!(bytes[1], 0x20); // PUSH32
    }

    #[test]
    fn double_sha256_known_vector() {
        // Bitcoin sha256d(""): 5df6e0e2761359d30a8275058e299fcc0381534545f55cf43e41983f5d4c9456
        let got = double_sha256(b"");
        let expected = [
            0x5d, 0xf6, 0xe0, 0xe2, 0x76, 0x13, 0x59, 0xd3, 0x0a, 0x82, 0x75, 0x05, 0x8e, 0x29,
            0x9f, 0xcc, 0x03, 0x81, 0x53, 0x45, 0x45, 0xf5, 0x5c, 0xf4, 0x3e, 0x41, 0x98, 0x3f,
            0x5d, 0x4c, 0x94, 0x56,
        ];
        assert_eq!(got, expected);
    }
}
