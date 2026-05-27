//! End-to-end submission: PlainProof → ZK prove → PearlBlock → pearld submit.

use zk_pow::api::prove::zk_prove_plain_proof;
use zk_pow::circuit::pearl_circuit::{PearlRecursion, RecursionCircuit};
use zk_pow::ffi::plain_proof::PlainProof;

use crate::error::MinerError;
use crate::gateway::block::{double_sha256, PearlBlock, PearlHeader, ZKCertificate};
use crate::gateway::client::PearldClient;
use crate::gateway::mining_job::MiningJob;

/// ZK-prove `plain_proof`, build the full PearlBlock, submit it via
/// `submitblock`. Returns `Ok(())` on accept; on reject the pearld reason
/// string is wrapped in [`MinerError::Rpc`].
///
/// Single function so the hit-callback hot path stays linear and the
/// failure modes (ZK fail, encode fail, RPC reject) are visible in one
/// place.
pub fn submit_plain_proof(
    plain_proof: &PlainProof,
    job: &MiningJob,
    client: &PearldClient,
    circuit_cache: &mut <PearlRecursion as RecursionCircuit>::CircuitCache,
) -> Result<(), MinerError> {
    // 1. ZK-prove.
    let zk_proof = zk_prove_plain_proof(job.incomplete_header, plain_proof, circuit_cache, true)
        .map_err(|e| MinerError::Rpc {
            method: "zk_prove".to_string(),
            msg: format!("{e}"),
        })?;

    // 2. Build the PearlHeader with proof_commitment set from public_data.
    let proof_commitment = ZKCertificate::compute_proof_commitment(&zk_proof.public_data);
    let header = PearlHeader {
        incomplete: job.incomplete_header,
        proof_commitment: Some(proof_commitment),
    };

    // 3. Hash the full 108-byte serialized header for the ZK certificate.
    let header_hash = double_sha256(&header.serialize());
    let cert = ZKCertificate {
        header_hash,
        public_data: zk_proof.public_data.to_vec(),
        proof_data: zk_proof.proof_data,
    };

    // 4. Assemble all transactions (coinbase first).
    let mut txns = Vec::with_capacity(1 + job.other_txs.len());
    txns.push(job.coinbase_tx.clone());
    txns.extend(job.other_txs.iter().cloned());

    let block = PearlBlock {
        header,
        txns,
        zk_certificate: cert,
    };
    let block_bytes = block.serialize()?;
    let block_hex = hex::encode(&block_bytes);

    // 5. Submit.
    client.submit_block(&block_hex)
}
