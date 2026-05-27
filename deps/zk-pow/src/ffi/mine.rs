//! Shared mining implementation for both Go FFI and Python bindings.

use anyhow::Result;
use primitive_types::U256;
use rand::Rng;

use crate::api::proof::{IncompleteBlockHeader, MiningConfiguration, PeriodicPattern};
use crate::api::proof_utils::compute_jackpot_hash;
use crate::api::sanity_checks::extract_difficulty_bound;
use crate::circuit::pearl_noise::compute_noise_for_indices;
use crate::circuit::pearl_program::{JACKPOT_SIZE, LROT_PER_TILE};
use crate::ffi::plain_proof::{MatrixMerkleProof, PlainProof};
use pearl_blake3::blake3_digest;

const SIGNAL_MIN: i8 = -64;
const SIGNAL_MAX: i8 = 64;

#[allow(clippy::too_many_arguments)]
pub fn try_mine_one<R: Rng>(
    rng: &mut R,
    m: usize,
    n: usize,
    k: usize,
    header: IncompleteBlockHeader,
    config: MiningConfiguration,
    signal_range: Option<(i8, i8)>,
    wrong_jackpot_hash: bool,
) -> Result<Option<PlainProof>> {
    let rank = config.rank as usize;
    let (signal_min, signal_max) = signal_range.unwrap_or((SIGNAL_MIN, SIGNAL_MAX));

    // Generate random matrices A (m×k) and B (k×n)
    let a_matrix: Vec<Vec<i8>> = (0..m)
        .map(|_| (0..k).map(|_| rng.random_range(signal_min..=signal_max)).collect())
        .collect();

    let b_matrix: Vec<Vec<i8>> = (0..k)
        .map(|_| (0..n).map(|_| rng.random_range(signal_min..=signal_max)).collect())
        .collect();

    // Transpose B for column-major format
    let b_transposed: Vec<Vec<i8>> = (0..n).map(|i| (0..k).map(|j| b_matrix[j][i]).collect()).collect();

    let job_key = compute_job_key(&header, &config);

    let a_row_major = pearl_blake3::pad_to_chunk_boundary(&flatten_matrix(&a_matrix));
    let b_col_major = pearl_blake3::pad_to_chunk_boundary(&flatten_matrix(&b_transposed));
    let (b_noise_seed, a_noise_seed) = compute_commitment_hash(&job_key, &a_row_major, &b_col_major);

    // Compute noise using shared implementation from pearl_noise
    let a_all_rows: Vec<usize> = (0..m).collect();
    let b_all_cols: Vec<usize> = (0..n).collect();
    let noise = compute_noise_for_indices(k, rank, (b_noise_seed, a_noise_seed), &a_all_rows, &b_all_cols);

    // Add noise to matrices (noise.a is m×k, noise.b is n×k as transposed columns)
    let a_noised: Vec<Vec<i32>> = a_matrix
        .iter()
        .zip(&noise.a)
        .map(|(a_row, n_row)| a_row.iter().zip(n_row).map(|(&a, &n)| a as i32 + n as i32).collect())
        .collect();

    // noise.b contains columns of B's noise as rows, need to transpose for b_matrix (k×n)
    let b_noised: Vec<Vec<i32>> = b_matrix
        .iter()
        .enumerate()
        .map(|(row_idx, b_row)| {
            b_row
                .iter()
                .enumerate()
                .map(|(col_idx, &b)| b as i32 + noise.b[col_idx][row_idx] as i32)
                .collect()
        })
        .collect();

    let b_noised_t: Vec<Vec<i32>> = (0..n).map(|i| (0..k).map(|j| b_noised[j][i]).collect()).collect();

    // Mine using pattern partitions
    for a_rows in threads_partition(&config.rows_pattern, m) {
        for b_cols in threads_partition(&config.cols_pattern, n) {
            // same as compute_jackpot but with a and b matrices pre-noised
            let tile_h = a_rows.len();
            let tile_w = b_cols.len();
            let mut jackpot_tile: Vec<Vec<i32>> = vec![vec![0; tile_w]; tile_h];
            let mut jackpot: [u32; 16] = [0; 16];

            for ll in (rank..=k).step_by(rank) {
                for (u, &a_idx) in a_rows.iter().enumerate() {
                    for (v, &b_idx) in b_cols.iter().enumerate() {
                        for l in ll - rank..ll {
                            jackpot_tile[u][v] += a_noised[a_idx][l] * b_noised_t[b_idx][l];
                        }
                    }
                }

                let xored_tile = jackpot_tile.iter().flatten().fold(0u32, |a, &x| a ^ x as u32);
                let tid = (ll / rank - 1) % JACKPOT_SIZE;
                jackpot[tid] = jackpot[tid].rotate_left(LROT_PER_TILE) ^ xored_tile;
            }
            let jackpot_hash = compute_jackpot_hash(&jackpot, a_noise_seed);
            let jackpot_bound = extract_difficulty_bound(header.nbits, &config);
            if (U256::from_little_endian(&jackpot_hash) <= jackpot_bound) != wrong_jackpot_hash {
                let a_proof = build_matrix_proof(&a_matrix, &job_key, &a_rows, k);
                let b_proof = build_matrix_proof(&b_transposed, &job_key, &b_cols, k);

                return Ok(Some(PlainProof {
                    m,
                    n,
                    k,
                    noise_rank: rank,
                    a: a_proof,
                    bt: b_proof,
                }));
            }
        }
    }

    Ok(None)
}

/// Mines a proof for the given block header and configuration.
///
/// * `signal_range` - Optional custom signal range [min, max] for testing. Default: (-64, 64)
/// * `wrong_jackpot_hash` - Accept wrong jackpot hash (for testing only)
#[allow(clippy::too_many_arguments)]
pub fn mine(
    m: usize,
    n: usize,
    k: usize,
    header: IncompleteBlockHeader,
    config: MiningConfiguration,
    signal_range: Option<(i8, i8)>,
    wrong_jackpot_hash: bool,
) -> Result<PlainProof> {
    let mut rng = rand::rng();

    loop {
        let proof = try_mine_one(&mut rng, m, n, k, header, config, signal_range, wrong_jackpot_hash)?;
        if let Some(proof) = proof {
            return Ok(proof);
        }
    }
}

/// Build a MatrixMerkleProof for the given matrix and row indices using pearl_blake3.
fn build_matrix_proof(matrix: &[Vec<i8>], job_key: &[u8; 32], row_indices: &[usize], num_cols: usize) -> MatrixMerkleProof {
    let padded = pearl_blake3::pad_to_chunk_boundary(&flatten_matrix(matrix));
    let tree = pearl_blake3::MerkleTree::new(&padded, *job_key);
    let leaf_indices = pearl_blake3::MerkleTree::compute_leaf_indices_from_rows(row_indices, (matrix.len(), num_cols));
    let proof = tree.get_multileaf_proof(&leaf_indices);
    MatrixMerkleProof {
        proof,
        row_indices: row_indices.to_vec(),
    }
}

fn compute_job_key(header: &IncompleteBlockHeader, config: &MiningConfiguration) -> [u8; 32] {
    let mut data = Vec::with_capacity(128);
    data.extend_from_slice(&header.to_bytes());
    data.extend_from_slice(&config.to_bytes());
    blake3_digest(&data, None)
}

fn compute_commitment_hash(job_key: &[u8; 32], a_row_major: &[u8], b_col_major: &[u8]) -> ([u8; 32], [u8; 32]) {
    let hash_a = blake3_digest(a_row_major, Some(*job_key));
    let hash_b = blake3_digest(b_col_major, Some(*job_key));

    let mut b_seed_input = [0u8; 64];
    b_seed_input[..32].copy_from_slice(job_key);
    b_seed_input[32..].copy_from_slice(&hash_b);
    let b_noise_seed = blake3_digest(&b_seed_input, None);

    let mut a_seed_input = [0u8; 64];
    a_seed_input[..32].copy_from_slice(&b_noise_seed);
    a_seed_input[32..].copy_from_slice(&hash_a);
    let a_noise_seed = blake3_digest(&a_seed_input, None);

    (b_noise_seed, a_noise_seed)
}

fn flatten_matrix(matrix: &[Vec<i8>]) -> Vec<u8> {
    matrix.iter().flatten().map(|&x| x as u8).collect()
}

fn threads_partition(pattern: &PeriodicPattern, total_dimension: usize) -> Vec<Vec<usize>> {
    let period = pattern.period() as usize;
    if !total_dimension.is_multiple_of(period) {
        panic!("total_dimension must be divisible by pattern period");
    }

    let base_indices: Vec<usize> = pattern.to_list().iter().map(|&i| i as usize).collect();

    (0..total_dimension)
        .filter(|&i| pattern.offset_is_valid(i as u32))
        .map(|offset| base_indices.iter().map(|&d| offset + d).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use blake3::CHUNK_LEN;

    const TEST_MATRIX_MOD: usize = 251;

    fn test_matrix(num_rows: usize, num_cols: usize) -> Vec<Vec<i8>> {
        (0..num_rows)
            .map(|r| (0..num_cols).map(|c| ((r * num_cols + c) % TEST_MATRIX_MOD) as i8).collect())
            .collect()
    }

    #[test]
    fn test_build_matrix_proof_pads_to_chunk_boundary() {
        // 3 rows x 500 cols = 1500 bytes, not a multiple of CHUNK_LEN (1024)
        let num_rows = 3;
        let num_cols = 500;
        assert_ne!((num_rows * num_cols) % CHUNK_LEN, 0);

        let matrix = test_matrix(num_rows, num_cols);
        let key = [42u8; 32];

        let proof = build_matrix_proof(&matrix, &key, &[0, 2], num_cols);

        let padded = pearl_blake3::pad_to_chunk_boundary(&flatten_matrix(&matrix));
        let expected_root = pearl_blake3::Blake3Hasher::with_key(key).hash(&padded);
        assert_eq!(
            proof.proof.root, expected_root,
            "Merkle root must equal blake3 of chunk-padded data"
        );
    }

    #[test]
    fn test_build_matrix_proof_aligned_unchanged() {
        // 4 rows x 256 cols = 1024 bytes, exactly one chunk
        let num_rows = 4;
        let num_cols = 256;
        assert_eq!((num_rows * num_cols) % CHUNK_LEN, 0);

        let matrix = test_matrix(num_rows, num_cols);
        let key = [7u8; 32];

        let proof = build_matrix_proof(&matrix, &key, &[1, 3], num_cols);

        let flat = flatten_matrix(&matrix);
        let expected_root = pearl_blake3::Blake3Hasher::with_key(key).hash(&flat);
        assert_eq!(
            proof.proof.root, expected_root,
            "Aligned data should produce identical root with or without padding"
        );
    }

    #[test]
    fn test_padded_blake3_hash_equals_merkle_root() {
        // The commitment hash derives noise seeds from blake3(padded_data, key).
        // This must equal MerkleTree::new(padded_data, key).root() so that
        // the verifier's Merkle proof check is consistent with the miner's
        // commitment.
        let num_rows = 3;
        let num_cols = 500;
        let matrix = test_matrix(num_rows, num_cols);
        let key = [99u8; 32];

        let padded = pearl_blake3::pad_to_chunk_boundary(&flatten_matrix(&matrix));
        let hash_via_digest = blake3_digest(&padded, Some(key));
        let hash_via_tree = pearl_blake3::MerkleTree::new(&padded, key).root();
        assert_eq!(hash_via_digest, hash_via_tree);
    }
}
