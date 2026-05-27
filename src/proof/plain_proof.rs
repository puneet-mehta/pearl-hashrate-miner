//! PlainProof construction from a hit's A/B snapshots + extracted indices.
//!
//! `PlainProof` is the wire format the verifier uses to (re)compute the
//! commitment hash + jackpot hash and check that the submitter actually
//! computed the matmul they claim. It's what the ZK prover takes as
//! private witness.

use pearl_blake3::MerkleTree;
use zk_pow::ffi::plain_proof::{MatrixMerkleProof, PlainProof};

use crate::error::MinerError;

/// Build a PlainProof from raw A bytes, raw B bytes, and the hit indices.
///
/// - `a_bytes`: `m * k` int8 (reinterpreted as u8) row-major. Same buffer
///   that the kernel hashed with `tensor_hash(A, key)`.
/// - `b_bytes`: `n * k` int8 (reinterpreted as u8). Convention: B is
///   stored s.t. `B[i, :]` is the i-th column of the matmul B. The kernel
///   hashed `tensor_hash(B, key)`; the verifier reconstructs from
///   "B^T" semantically but the bytes on the wire ARE the same `(n, k)`
///   row-major. (See miner_base/commitment_hash.py docstring.)
/// - `a_row_indices`: global row indices selected by the kernel.
/// - `b_col_indices`: global column indices selected by the kernel.
/// - `key`: per-job Blake3 key (= same value passed to `tensor_hash`).
///
/// Returns a PlainProof whose serialized form (`bincode::serialize +
/// base64`) matches what Python's `_run_iter_body` → gateway path
/// produces.
pub fn build_plain_proof(
    m: usize,
    n: usize,
    k: usize,
    noise_rank: usize,
    a_bytes: &[u8],
    b_bytes: &[u8],
    a_row_indices: Vec<usize>,
    b_col_indices: Vec<usize>,
    key: [u8; 32],
) -> Result<PlainProof, MinerError> {
    if a_bytes.len() != m * k {
        return Err(MinerError::Rpc {
            method: "plain_proof".to_string(),
            msg: format!("A length {} != m*k = {}", a_bytes.len(), m * k),
        });
    }
    if b_bytes.len() != n * k {
        return Err(MinerError::Rpc {
            method: "plain_proof".to_string(),
            msg: format!("B length {} != n*k = {}", b_bytes.len(), n * k),
        });
    }

    // --- A side ---
    let a_leaf_indices = MerkleTree::compute_leaf_indices_from_rows(&a_row_indices, (m, k));
    let a_tree = MerkleTree::new(a_bytes, key);
    let a_proof = a_tree.get_multileaf_proof(&a_leaf_indices);

    // --- B^t side (B stored (n, k) row-major; rows = columns of the matmul B) ---
    let bt_leaf_indices = MerkleTree::compute_leaf_indices_from_rows(&b_col_indices, (n, k));
    let bt_tree = MerkleTree::new(b_bytes, key);
    let bt_proof = bt_tree.get_multileaf_proof(&bt_leaf_indices);

    Ok(PlainProof {
        m,
        n,
        k,
        noise_rank,
        a: MatrixMerkleProof {
            proof: a_proof,
            row_indices: a_row_indices,
        },
        bt: MatrixMerkleProof {
            proof: bt_proof,
            row_indices: b_col_indices,
        },
    })
}
