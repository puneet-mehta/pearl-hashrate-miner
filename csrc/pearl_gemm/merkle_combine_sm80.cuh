// merkle_combine_sm80.cuh
//
// GPU-side leaf-combine pass for the keyed-Blake3 Merkle tree, replacing
// `pearl::sm80::merkle::merkle_combine_host`. Used by the fully-async
// `merkle_root_device` variant in merkle_sm80.cuh (which the mining hot
// path uses via pearl_gemm_tensor_hash).
//
// Algorithm mirrors `merkle_combine_host` byte-for-byte:
//   * Repeatedly pair-combine non-root CVs (KEYED_HASH | PARENT) until
//     2 nodes remain.
//   * The final pair gets the ROOT flag.
//   * Orphan nodes (odd count in a layer) carry through unchanged.
//
// Each pass is a separate kernel launch keyed by the layer size, ping-ponging
// between two halves of a contiguous scratchpad. log2(num_leaves) launches
// total (~13 for the largest input in the project); per-launch overhead is
// negligible against the per-iter cost the host path replaced.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "blake3_sm80.cuh"

namespace pearl::sm80::merkle {

inline constexpr int OUT_LEN_U32_GPU = 8;

// One thread per output CV. `num_in` is the number of 8-u32 CVs in the
// input layer; output layer has `num_in/2 + (num_in & 1)` CVs.
//
// `is_top_pair_root`: when true, the LAST pair in this layer is the root
// pair → its parent_cv emits with the ROOT flag (in addition to PARENT |
// KEYED_HASH). Caller arranges this only for the final layer transition.
__global__ inline void merkle_layer_kernel(
    const uint32_t* __restrict__ layer_in,
    int num_in,
    const uint32_t* __restrict__ key,
    bool is_top_pair_root,
    uint32_t* __restrict__ layer_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_pairs = num_in >> 1;
  const int has_orphan = num_in & 1;

  if (i < num_pairs) {
    // parent_cv(left=layer_in[2i], right=layer_in[2i+1], key, ...)
    uint32_t msg[16];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      msg[j]     = layer_in[(2 * i + 0) * 8 + j];
      msg[8 + j] = layer_in[(2 * i + 1) * 8 + j];
    }

    uint32_t cv[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) cv[j] = key[j];

    // Root iff this is the final pair AND we're emitting the root in this
    // layer (i.e. no orphan and is_top_pair_root). When an orphan carries,
    // the *next* layer produces the root from this pair + orphan.
    const bool is_root_call =
        is_top_pair_root && (i == num_pairs - 1) && (has_orphan == 0);
    uint32_t flags = pearl::sm80::blake3::KEYED_HASH
                   | pearl::sm80::blake3::PARENT
                   | (is_root_call ? pearl::sm80::blake3::ROOT : 0u);
    pearl::sm80::blake3::CompressParams params{
        0ull, pearl::sm80::blake3::MSG_BLOCK_SIZE, flags};
    pearl::sm80::blake3::compress_msg_block_u32(msg, cv, params);

    #pragma unroll
    for (int j = 0; j < 8; ++j) layer_out[i * 8 + j] = cv[j];
  } else if (i == num_pairs && has_orphan) {
    // Carry the orphan to the next layer unchanged.
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      layer_out[num_pairs * 8 + j] = layer_in[(num_in - 1) * 8 + j];
    }
  }
}

// Combine `num_leaves` CVs (in `bufA`) into a single 8-u32 root. Uses
// `bufA` and `bufB` as ping-pong scratchpads. The final root ends up in
// either bufA or bufB; the function returns a pointer to it.
//
// `bufA` size must be >= num_leaves * 8 u32. `bufB` size must be >=
// (num_leaves / 2 + 1) * 8 u32.
inline const uint32_t* merkle_combine_device(
    uint32_t* bufA,
    uint32_t* bufB,
    int num_leaves,
    const uint32_t* d_key,
    cudaStream_t stream) {
  if (num_leaves <= 1) {
    // Single leaf is already the root (caller put it there with ROOT flag
    // set by chunk_cv_kernel<kAddRoot=true>); nothing to do.
    return bufA;
  }

  const uint32_t* in_buf = bufA;
  uint32_t* out_buf = bufB;
  int size = num_leaves;

  while (size > 2) {
    const int num_pairs = size >> 1;
    const int has_orphan = size & 1;
    const int next_size = num_pairs + has_orphan;
    const int total_threads = num_pairs + has_orphan;
    const int block = 128;
    const int grid = (total_threads + block - 1) / block;
    merkle_layer_kernel<<<grid, block, 0, stream>>>(
        in_buf, size, d_key, /*is_top_pair_root=*/false, out_buf);

    const uint32_t* tmp_in = in_buf;
    in_buf  = out_buf;
    out_buf = const_cast<uint32_t*>(tmp_in);
    size = next_size;
  }

  // size == 2: final pair gets ROOT.
  merkle_layer_kernel<<<1, 32, 0, stream>>>(
      in_buf, size, d_key, /*is_top_pair_root=*/true, out_buf);
  return out_buf;
}

}  // namespace pearl::sm80::merkle
