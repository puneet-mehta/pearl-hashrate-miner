// merkle_sm80.cuh
//
// Standalone Ampere/Ada port of pearl-blake3's MerkleTree, i.e. the
// algorithm behind the upstream merkle_tree_roots_kernel.hpp. Produces a
// byte-identical Merkle root to `pearl_blake3::MerkleTree::root()` for any
// keyed-Blake3 tree over (zero-padded-to-1024-byte) tensor data.
//
// Structure
// ---------
// One kernel does the heavy lifting (chunk-CV hashing), one host helper
// does the tiny tree-combine pass:
//
//   chunk_cv_kernel<bool kAddRoot>:
//     One thread per 1024-byte chunk. 16 chained Blake3 compress() calls
//     (first with CHUNK_START, last with CHUNK_END (+ROOT iff kAddRoot)).
//     Writes the 8-u32 chaining value out.
//
//   merkle_combine_host:
//     Pair-wise reduction of leaf CVs using compress_msg_block_u32 (which
//     is __host__ __device__). Top pair gets the ROOT flag. log2(N) levels
//     of O(N/2) compresses; for the largest case in the project (512
//     chunks) this is ~511 compresses on CPU — sub-millisecond.
//
// The split is for engineering pragmatism, not perf: tree-combine is so
// small relative to leaf hashing that running it on the GPU adds launch
// overhead with no payoff. The same algorithm written as a single GPU
// kernel would produce identical bytes.

#pragma once

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

#include "blake3_sm80.cuh"
#include "merkle_combine_sm80.cuh"

namespace pearl::sm80::merkle {

inline constexpr int CHUNK_LEN     = 1024;
inline constexpr int BLOCK_LEN     = 64;
inline constexpr int CHUNK_BLOCKS  = CHUNK_LEN / BLOCK_LEN;  // 16
inline constexpr int OUT_LEN_U32   = 8;

// =============================================================================
//   GPU: per-chunk Blake3 chunk CV
// =============================================================================
//
// `padded_data` must be a multiple of CHUNK_LEN bytes long. Each thread
// processes one chunk. When `kAddRoot` is true and a chunk is the only chunk
// in the tree, this kernel also produces the root in one shot (the
// upstream's MerkleTree::new short-circuits the single-chunk path that way).

template <bool kAddRoot>
__global__ void chunk_cv_kernel(const uint8_t* __restrict__ padded_data,
                                int num_chunks,
                                const uint32_t* __restrict__ key,
                                uint32_t* __restrict__ leaf_cvs) {
  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk >= num_chunks) return;

  // Load key once per thread.
  uint32_t cv[OUT_LEN_U32];
  #pragma unroll
  for (int i = 0; i < OUT_LEN_U32; ++i) cv[i] = key[i];

  // Process 16 blocks of 64 bytes each, chaining the CV.
  const uint8_t* chunk_ptr = padded_data + chunk * CHUNK_LEN;

  #pragma unroll 1
  for (int blk = 0; blk < CHUNK_BLOCKS; ++blk) {
    // Load 64 bytes = 16 u32 LE words into a register array. We do byte
    // reads + manual pack so this works on any alignment.
    uint32_t msg[16];
    const uint8_t* p = chunk_ptr + blk * BLOCK_LEN;
    #pragma unroll
    for (int w = 0; w < 16; ++w) {
      uint32_t b0 = static_cast<uint32_t>(p[w * 4 + 0]);
      uint32_t b1 = static_cast<uint32_t>(p[w * 4 + 1]);
      uint32_t b2 = static_cast<uint32_t>(p[w * 4 + 2]);
      uint32_t b3 = static_cast<uint32_t>(p[w * 4 + 3]);
      msg[w] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }

    uint32_t flags = pearl::sm80::blake3::KEYED_HASH;
    if (blk == 0)                  flags |= pearl::sm80::blake3::CHUNK_START;
    if (blk == CHUNK_BLOCKS - 1)   flags |= pearl::sm80::blake3::CHUNK_END;
    if (kAddRoot && blk == CHUNK_BLOCKS - 1)
                                   flags |= pearl::sm80::blake3::ROOT;

    pearl::sm80::blake3::CompressParams params{
        static_cast<uint64_t>(chunk), BLOCK_LEN, flags};
    pearl::sm80::blake3::compress_msg_block_u32(msg, cv, params);
  }

  // Store this chunk's CV (8 u32 = 32 bytes).
  uint32_t* out = leaf_cvs + chunk * OUT_LEN_U32;
  #pragma unroll
  for (int i = 0; i < OUT_LEN_U32; ++i) out[i] = cv[i];
}

// =============================================================================
//   Host: single-chunk hash (also handles sub-chunk inputs)
// =============================================================================
//
// Matches pearl-blake3's `hasher.hash(data)` for data <= CHUNK_LEN: process
// `len` bytes as ceil(len/64) blocks, zero-padding the final partial block
// to 64 bytes but recording block_len = BLOCK_LEN throughout. This is what
// blake3_merkle_ref.py's chunk_cv path does — it differs from "standard"
// Blake3 (which records the actual partial-block length) but matches what
// the upstream Rust MerkleTree builds against. Our golden fixtures all use
// inputs that are multiples of 64 bytes, so the discrepancy never bites.

inline void hash_single_chunk_host(const uint8_t* data, size_t len,
                                   const uint32_t key[8],
                                   uint64_t chunk_index, bool is_root,
                                   uint32_t out_cv[8]) {
  for (int i = 0; i < 8; ++i) out_cv[i] = key[i];

  const size_t num_blocks =
      len == 0 ? 1 : (len + BLOCK_LEN - 1) / BLOCK_LEN;

  for (size_t blk = 0; blk < num_blocks; ++blk) {
    uint8_t block_bytes[BLOCK_LEN] = {0};
    const size_t off = blk * BLOCK_LEN;
    const size_t copy_n = (off < len) ? std::min<size_t>(BLOCK_LEN, len - off)
                                      : 0;
    if (copy_n > 0) std::memcpy(block_bytes, data + off, copy_n);

    uint32_t msg[16];
    for (int w = 0; w < 16; ++w) {
      msg[w] = static_cast<uint32_t>(block_bytes[w * 4 + 0])
             | (static_cast<uint32_t>(block_bytes[w * 4 + 1]) << 8)
             | (static_cast<uint32_t>(block_bytes[w * 4 + 2]) << 16)
             | (static_cast<uint32_t>(block_bytes[w * 4 + 3]) << 24);
    }

    uint32_t flags = pearl::sm80::blake3::KEYED_HASH;
    if (blk == 0)                  flags |= pearl::sm80::blake3::CHUNK_START;
    if (blk == num_blocks - 1) {
      flags                              |= pearl::sm80::blake3::CHUNK_END;
      if (is_root)                 flags |= pearl::sm80::blake3::ROOT;
    }

    pearl::sm80::blake3::CompressParams params{chunk_index, BLOCK_LEN, flags};
    pearl::sm80::blake3::compress_msg_block_u32(msg, out_cv, params);
  }
}

// =============================================================================
//   Host: tree combine
// =============================================================================

// Combine two 8-u32 CVs into a parent CV. `is_root` adds the ROOT flag.
inline void parent_cv_host(const uint32_t left[8], const uint32_t right[8],
                           const uint32_t key[8], bool is_root,
                           uint32_t out[8]) {
  // Build the 64-byte message block (left || right as 16 u32).
  uint32_t msg[16];
  for (int i = 0; i < 8; ++i) msg[i]     = left[i];
  for (int i = 0; i < 8; ++i) msg[8 + i] = right[i];

  // Initialise CV from key.
  for (int i = 0; i < 8; ++i) out[i] = key[i];

  const uint32_t flags = pearl::sm80::blake3::KEYED_HASH
                       | pearl::sm80::blake3::PARENT
                       | (is_root ? pearl::sm80::blake3::ROOT : 0);
  pearl::sm80::blake3::CompressParams params{0ull, BLOCK_LEN, flags};
  pearl::sm80::blake3::compress_msg_block_u32(msg, out, params);
}

// Iteratively combine leaf CVs into a single root CV, mirroring
// pearl-blake3's MerkleTree::new logic (odd elements pass through; the
// final pair gets ROOT).
inline std::vector<uint32_t> merkle_combine_host(
    std::vector<uint32_t> layer /* (num_leaves * 8) u32 */,
    const uint32_t key[8]) {
  if (layer.empty()) {
    // Empty tree: 32 zero bytes (mirrors the Rust impl).
    return std::vector<uint32_t>(8, 0u);
  }
  if (layer.size() == 8) {
    // Single leaf — that leaf IS the root.
    return layer;
  }
  // While the layer has more than 2 nodes, pair-combine non-root.
  while (layer.size() > 2 * 8) {
    std::vector<uint32_t> next;
    next.reserve((layer.size() / 16) * 8 + 8);
    size_t n = layer.size() / 8;
    size_t i = 0;
    while (i + 1 < n) {
      uint32_t out[8];
      parent_cv_host(&layer[i * 8], &layer[(i + 1) * 8], key,
                     /*is_root=*/false, out);
      for (int j = 0; j < 8; ++j) next.push_back(out[j]);
      i += 2;
    }
    if (i < n) {
      // Carry the orphan.
      for (int j = 0; j < 8; ++j) next.push_back(layer[i * 8 + j]);
    }
    layer = std::move(next);
  }
  // Top pair gets ROOT.
  std::vector<uint32_t> root(8);
  if (layer.size() == 16) {
    parent_cv_host(&layer[0], &layer[8], key, /*is_root=*/true,
                   root.data());
  } else {
    // size == 8: single leaf, already the root.
    root.assign(layer.begin(), layer.end());
  }
  return root;
}

// =============================================================================
//   Top-level orchestrator
// =============================================================================

// Compute the Merkle root of `raw_data` (any length) with `key` (32 bytes).
// Performs zero-padding to a 1024-byte boundary on the host, uploads to
// device, runs chunk_cv_kernel, downloads leaf CVs, and combines on host.
//
//   * d_key:    8 u32 device pointer to the key.
//   * raw_data: pointer to host bytes (any length, will be padded).
//   * key:      8 u32 host pointer to the key (same data as d_key).
//   * out_root: host destination for 32 bytes.
inline void merkle_root(const uint32_t* d_key, const uint32_t* key,
                        const uint8_t* raw_data, size_t raw_len,
                        uint8_t out_root[32],
                        cudaStream_t stream = nullptr) {
  if (raw_len == 0) {
    std::memset(out_root, 0, 32);
    return;
  }

  // Sub-chunk fast path: mirrors `MerkleTree::new` short-circuit. We hash
  // the RAW bytes (not the zero-padded buffer) with ROOT on the final
  // block. Cheap enough to do on host — avoids the kernel-launch overhead
  // for trees with one leaf.
  if (raw_len <= CHUNK_LEN) {
    (void)d_key;  // d_key path unused here; host-only.
    uint32_t cv[OUT_LEN_U32];
    hash_single_chunk_host(raw_data, raw_len, key, /*chunk_index=*/0,
                           /*is_root=*/true, cv);
    std::memcpy(out_root, cv, 32);
    return;
  }

  // Multi-chunk: zero-pad to chunk boundary, run kernel, combine on host.
  std::vector<uint8_t> padded(raw_data, raw_data + raw_len);
  size_t target = ((raw_len + CHUNK_LEN - 1) / CHUNK_LEN) * CHUNK_LEN;
  padded.resize(target, 0);
  const int num_chunks = static_cast<int>(target / CHUNK_LEN);

  uint8_t* d_data = nullptr;
  uint32_t* d_leaves = nullptr;
  cudaMalloc(&d_data, padded.size());
  cudaMalloc(&d_leaves, num_chunks * OUT_LEN_U32 * sizeof(uint32_t));
  cudaMemcpyAsync(d_data, padded.data(), padded.size(),
                  cudaMemcpyHostToDevice, stream);

  const int block = 128;
  const int grid = (num_chunks + block - 1) / block;
  chunk_cv_kernel</*kAddRoot=*/false><<<grid, block, 0, stream>>>(
      d_data, num_chunks, d_key, d_leaves);

  std::vector<uint32_t> leaves(num_chunks * OUT_LEN_U32);
  cudaMemcpyAsync(leaves.data(), d_leaves,
                  leaves.size() * sizeof(uint32_t),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  cudaFree(d_data);
  cudaFree(d_leaves);

  auto root = merkle_combine_host(std::move(leaves), key);
  std::memcpy(out_root, root.data(), 32);
}

// =============================================================================
//   GPU: fully-device merkle root (no host bounce, fully async)
// =============================================================================
//
// Used by pearl_gemm_tensor_hash on the mining hot path. Skips the device →
// host → device round-trip the original `merkle_root` does (which forces a
// cudaStreamSynchronize). The caller-provided scratchpad is sized for
// ping-pong (2 * num_chunks * 32 bytes); see get_required_scratchpad_bytes
// on the Python side.
//
// Constraints:
//   - raw_len must be a positive multiple of CHUNK_LEN (1024). For mining
//     inputs (A is (m,k) int8 multiple-of-1024 bytes, B same), this always
//     holds. Non-multiple inputs are not supported by this path — callers
//     that need sub-chunk hashing must use the host-bouncing `merkle_root`.
//   - d_scratch must hold at least `2 * num_chunks * OUT_LEN_U32` u32s.
//   - d_out is 32 bytes (8 u32) on device; the final root is written here.

inline void merkle_root_device(
    const uint8_t* d_data,
    size_t raw_len,
    const uint32_t* d_key,
    uint32_t* d_scratch,
    size_t scratch_u32_count,
    uint8_t* d_out,  // 32 bytes
    cudaStream_t stream) {
  // raw_len must be a positive multiple of CHUNK_LEN; this fast path is for
  // mining-shape inputs only.
  const int num_chunks = static_cast<int>(raw_len / CHUNK_LEN);

  // Scratchpad layout: bufA at [0, num_chunks*8), bufB at [num_chunks*8, 2*num_chunks*8).
  // bufB only needs (num_chunks/2 + 1)*8 u32s; allocating 2*num_chunks*8 is
  // a simple overestimate that fits any layer count.
  uint32_t* bufA = d_scratch;
  uint32_t* bufB = d_scratch + static_cast<size_t>(num_chunks) * OUT_LEN_U32;
  (void)scratch_u32_count;  // size enforced by the caller via Python wrapper.

  if (num_chunks == 1) {
    // Single chunk: chunk_cv_kernel<kAddRoot=true> produces the root in
    // one shot. Write directly to bufA, then copy 32 bytes to d_out.
    chunk_cv_kernel</*kAddRoot=*/true><<<1, 1, 0, stream>>>(
        d_data, /*num_chunks=*/1, d_key, bufA);
    cudaMemcpyAsync(d_out, bufA, 32, cudaMemcpyDeviceToDevice, stream);
    return;
  }

  // Multi-chunk: chunk_cv_kernel (no ROOT) populates bufA[0..num_chunks*8).
  {
    const int block = 128;
    const int grid = (num_chunks + block - 1) / block;
    chunk_cv_kernel</*kAddRoot=*/false><<<grid, block, 0, stream>>>(
        d_data, num_chunks, d_key, bufA);
  }

  // Tree-combine in place across log2(num_chunks) layer launches.
  const uint32_t* root_ptr =
      merkle_combine_device(bufA, bufB, num_chunks, d_key, stream);
  cudaMemcpyAsync(d_out, root_ptr, 32, cudaMemcpyDeviceToDevice, stream);
}

}  // namespace pearl::sm80::merkle
