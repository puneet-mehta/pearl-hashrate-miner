extern "C" __global__ void pearl_chunk_cv_kernel_root(
    const uint8_t* __restrict__ padded_data,
    int num_chunks,
    const uint32_t* __restrict__ key,
    uint32_t* __restrict__ leaf_cvs) {
  // Forwarder to the templated kernel. Same signature, just a stable symbol.
  // We can't directly call __global__ from __global__, so we duplicate the
  // body — same code as pearl::sm80::merkle::chunk_cv_kernel<true>.
  using namespace pearl::sm80::blake3;
  using pearl::sm80::merkle::CHUNK_LEN;
  using pearl::sm80::merkle::BLOCK_LEN;
  using pearl::sm80::merkle::CHUNK_BLOCKS;
  using pearl::sm80::merkle::OUT_LEN_U32;
  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk >= num_chunks) return;

  uint32_t cv[OUT_LEN_U32];
  #pragma unroll
  for (int i = 0; i < OUT_LEN_U32; ++i) cv[i] = key[i];

  const uint8_t* chunk_ptr = padded_data + chunk * CHUNK_LEN;
  #pragma unroll 1
  for (int blk = 0; blk < CHUNK_BLOCKS; ++blk) {
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
    uint32_t flags = KEYED_HASH;
    if (blk == 0)                       flags |= CHUNK_START;
    if (blk == CHUNK_BLOCKS - 1)        flags |= CHUNK_END | ROOT;  // kAddRoot=true
    CompressParams params{static_cast<uint64_t>(chunk), BLOCK_LEN, flags};
    compress_msg_block_u32(msg, cv, params);
  }
  uint32_t* out = leaf_cvs + chunk * OUT_LEN_U32;
  #pragma unroll
  for (int i = 0; i < OUT_LEN_U32; ++i) out[i] = cv[i];
}

extern "C" __global__ void pearl_chunk_cv_kernel_noroot(
    const uint8_t* __restrict__ padded_data,
    int num_chunks,
    const uint32_t* __restrict__ key,
    uint32_t* __restrict__ leaf_cvs) {
  // Same as pearl_chunk_cv_kernel_root but the final block doesn't get ROOT.
  using namespace pearl::sm80::blake3;
  using pearl::sm80::merkle::CHUNK_LEN;
  using pearl::sm80::merkle::BLOCK_LEN;
  using pearl::sm80::merkle::CHUNK_BLOCKS;
  using pearl::sm80::merkle::OUT_LEN_U32;
  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk >= num_chunks) return;

  uint32_t cv[OUT_LEN_U32];
  #pragma unroll
  for (int i = 0; i < OUT_LEN_U32; ++i) cv[i] = key[i];

  const uint8_t* chunk_ptr = padded_data + chunk * CHUNK_LEN;
  #pragma unroll 1
  for (int blk = 0; blk < CHUNK_BLOCKS; ++blk) {
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
    uint32_t flags = KEYED_HASH;
    if (blk == 0)                       flags |= CHUNK_START;
    if (blk == CHUNK_BLOCKS - 1)        flags |= CHUNK_END;  // kAddRoot=false
    CompressParams params{static_cast<uint64_t>(chunk), BLOCK_LEN, flags};
    compress_msg_block_u32(msg, cv, params);
  }
  uint32_t* out = leaf_cvs + chunk * OUT_LEN_U32;
  #pragma unroll
  for (int i = 0; i < OUT_LEN_U32; ++i) out[i] = cv[i];
}

extern "C" __global__ void pearl_merkle_layer_kernel(
    const uint32_t* __restrict__ layer_in,
    int num_in,
    const uint32_t* __restrict__ key,
    bool is_top_pair_root,
    uint32_t* __restrict__ layer_out) {
  // Same body as pearl::sm80::merkle::merkle_layer_kernel.
  using namespace pearl::sm80::blake3;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_pairs = num_in >> 1;
  const int has_orphan = num_in & 1;

  if (i < num_pairs) {
    uint32_t msg[16];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      msg[j]     = layer_in[(2 * i + 0) * 8 + j];
      msg[8 + j] = layer_in[(2 * i + 1) * 8 + j];
    }
    uint32_t cv[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) cv[j] = key[j];

    const bool is_root_call =
        is_top_pair_root && (i == num_pairs - 1) && (has_orphan == 0);
    uint32_t flags = KEYED_HASH | PARENT | (is_root_call ? ROOT : 0u);
    CompressParams params{0ull, MSG_BLOCK_SIZE, flags};
    compress_msg_block_u32(msg, cv, params);
    #pragma unroll
    for (int j = 0; j < 8; ++j) layer_out[i * 8 + j] = cv[j];
  } else if (i == num_pairs && has_orphan) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      layer_out[num_pairs * 8 + j] = layer_in[(num_in - 1) * 8 + j];
    }
  }
}

// ----- Per-iter random int8 fill (replaces A.random_(-63, 64)) -----
//
// The reference miner uses `A.random_(-63, 64)` (torch curand under the hood) to
// reseed A every iter. The Rust port avoids curand by hashing (seed, iter,
// chunk) through Blake3 — same primitives as the rest of the pipeline. Output
// is int8 in [-63, 63] inclusive (127 values; same range as torch's random_).
//
// Algorithm (per 32-byte output chunk):
//   message = [iter_idx.lo, iter_idx.hi, chunk_idx, 0, 0, 0, 0, 0,
//              seed[0], seed[1], ..., seed[7]]  (16 u32, 64 bytes)
//   key     = blake3 IV (not keyed_hash — seed is in the message)
//   compress with CHUNK_START | CHUNK_END | ROOT
//   for each of 32 output bytes: signed_v = (uint8(b) % 127) - 63  ∈ [-63, 63]
//
// total_bytes must be a multiple of 32 (production A: 2048*4096 = 8 MB, OK).

extern "C" __global__ void pearl_random_int8_seeded_kernel(
    int total_bytes,
    const uint32_t* __restrict__ seed,   // 32 bytes (8 u32)
    uint64_t iter_idx,
    int8_t* __restrict__ out) {
  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_chunks = total_bytes / 32;
  if (chunk >= num_chunks) return;

  // Build the 64-byte message.
  uint32_t msg[16];
  msg[0] = static_cast<uint32_t>(iter_idx & 0xFFFFFFFFull);
  msg[1] = static_cast<uint32_t>((iter_idx >> 32) & 0xFFFFFFFFull);
  msg[2] = static_cast<uint32_t>(chunk);
  #pragma unroll
  for (int i = 3; i < 8; ++i) msg[i] = 0u;
  #pragma unroll
  for (int i = 0; i < 8; ++i) msg[8 + i] = seed[i];

  // Start CV from Blake3 IV (NOT keyed_hash — seed is in the message).
  using namespace pearl::sm80::blake3;
  uint32_t cv[8] = {IV0, IV1, IV2, IV3, IV4, IV5, IV6, IV7};
  constexpr CompressParams params{0u, MSG_BLOCK_SIZE,
                                  CHUNK_START | CHUNK_END | ROOT};
  compress_msg_block_u32(msg, cv, params);

  // Decode 32 bytes (= 8 u32 in LE) into int8 ∈ [-63, 63].
  int8_t* out_chunk = out + chunk * 32;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t w = cv[i];
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
      uint32_t byte = w & 0xff;
      w >>= 8;
      // 127 values: 0..126 → -63..63
      int32_t v = static_cast<int32_t>(byte % 127u) - 63;
      out_chunk[i * 4 + b] = static_cast<int8_t>(v);
    }
  }
}

// ----- PoW scan + emit -----
//
// Two kernels:
//   pearl_pow_scan_hits_kernel — finds the first set bit in d_hit
//   pearl_pow_emit_header_kernel — emits a 640-byte status header into the
//     pinned host buffer when a hit was found (no-op otherwise)
//
// Caller must `cuMemsetD8Async(d_first_hit_idx, 0xFF, 4, stream)` before
// the scan kernel — the scan uses atomicMin against UINT32_MAX sentinel.

extern "C" __global__ void pearl_pow_scan_hits_kernel(
    const uint8_t* __restrict__ d_hit,
    int total,
    uint32_t* __restrict__ g_first_hit_idx) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total) return;
  if (d_hit[tid]) {
    atomicMin(g_first_hit_idx, static_cast<uint32_t>(tid));
  }
}

extern "C" __global__ void pearl_pow_emit_header_kernel(
    const uint32_t* __restrict__ g_first_hit_idx,
    const uint32_t* __restrict__ pow_target,
    uint8_t* __restrict__ pinned_header,
    uint8_t* __restrict__ pinned_sync,
    int num_tile_m,
    int num_tile_n,
    int threads_per_tile,
    int m,
    int n,
    int k) {
  // Same body as pearl::sm80::pow_scan_emit::pow_emit_header_kernel.
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  const uint32_t first_hit = *g_first_hit_idx;
  if (first_hit == 0xFFFFFFFFu) return;

  const int tile_linear = static_cast<int>(first_hit / static_cast<uint32_t>(threads_per_tile));
  const int thread_idx  = static_cast<int>(first_hit % static_cast<uint32_t>(threads_per_tile));
  const int tile_m_idx  = tile_linear / num_tile_n;
  const int tile_n_idx  = tile_linear % num_tile_n;

  auto store_u32 = [&](int off, uint32_t v) {
    *reinterpret_cast<uint32_t*>(pinned_header + off) = v;
  };
  auto store_u16 = [&](int off, uint16_t v) {
    *reinterpret_cast<uint16_t*>(pinned_header + off) = v;
  };
  auto store_i32 = [&](int off, int32_t v) {
    *reinterpret_cast<int32_t*>(pinned_header + off) = v;
  };

  store_u32(4,  static_cast<uint32_t>(num_tile_n));
  store_u32(8,  static_cast<uint32_t>(num_tile_m));
  store_u32(12, 1u);
  store_u32(16, 256u); store_u32(20, 1u); store_u32(24, 1u);
  store_u32(28, static_cast<uint32_t>(tile_n_idx));
  store_u32(32, static_cast<uint32_t>(tile_m_idx));
  store_u32(36, 0u);
  store_u32(40, static_cast<uint32_t>(tile_m_idx));
  store_u32(44, static_cast<uint32_t>(tile_n_idx));
  store_u32(48, 0u);
  store_u32(52, static_cast<uint32_t>(thread_idx));
  store_u32(56, 0u); store_u32(60, 0u);

  store_u16(64, static_cast<uint16_t>(64));

  const int warp_id = thread_idx >> 5;
  const int lane_id = thread_idx & 31;
  const int gid     = lane_id >> 2;
  const int tig     = lane_id & 3;
  const int row_lo  = warp_id * 16 + gid;
  const int row_hi  = row_lo + 8;
  #pragma unroll
  for (int na = 0; na < 16; ++na) {
    const int col0  = na * 8 + 2 * tig;
    const int col1  = col0 + 1;
    const int base  = 66 + na * 4;
    pinned_header[base + 0] = static_cast<uint8_t>(row_lo);
    pinned_header[base + 1] = static_cast<uint8_t>(row_lo);
    pinned_header[base + 2] = static_cast<uint8_t>(row_hi);
    pinned_header[base + 3] = static_cast<uint8_t>(row_hi);
    const int cbase = 322 + na * 4;
    pinned_header[cbase + 0] = static_cast<uint8_t>(col0);
    pinned_header[cbase + 1] = static_cast<uint8_t>(col1);
    pinned_header[cbase + 2] = static_cast<uint8_t>(col0);
    pinned_header[cbase + 3] = static_cast<uint8_t>(col1);
  }

  store_i32(580, m); store_i32(584, n); store_i32(588, k);
  store_i32(592, 128); store_i32(596, 128); store_i32(600, 128);
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    store_u32(604 + i * 4, pow_target[i]);
  }
  if (pinned_sync != nullptr) {
    *reinterpret_cast<uint32_t*>(pinned_sync + 4) = 1u;
  }
  __threadfence_system();
  store_u32(0, 1u);
  __threadfence_system();
}

// ----- Noise generation (R=128 production rank) -----
//
// Four extern-C shims for the noise_gen kernel set, R=128 only (the only
// noise_rank actually used by the production miner per MinerSettings). If
// R=64 is ever needed, add parallel `_R64` shims following the same pattern.

extern "C" __global__ void pearl_noise_gen_dense_int8_R128(
    int rows,
    const uint32_t* __restrict__ key,
    const uint32_t* __restrict__ seed,
    int8_t* __restrict__ out) {
  constexpr int R = 128;
  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_chunks = (rows * R) / 32;
  if (chunk >= num_chunks) return;
  uint32_t key_r[8], seed_r[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) { key_r[i] = key[i]; seed_r[i] = seed[i]; }
  uint32_t msg[16];
  pearl::sm80::noise_gen::detail::make_message(seed_r, msg, /*slot=*/0,
      static_cast<uint32_t>(chunk + 1));
  uint32_t cv[8];
  pearl::sm80::noise_gen::detail::keyed_compress(key_r, msg, cv);
  const int row = (chunk * 32) / R;
  const int col0 = (chunk * 32) % R;
  int8_t* out_row = out + row * R + col0;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t w = cv[i];
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
      int8_t hb = static_cast<int8_t>(w & 0xff);
      w >>= 8;
      int32_t v = (static_cast<int32_t>(hb) + 128) % 64 - 32;
      out_row[i * 4 + b] = static_cast<int8_t>(v);
    }
  }
}

extern "C" __global__ void pearl_noise_gen_dense_fp16_R128(
    int rows,
    const uint32_t* __restrict__ key,
    const uint32_t* __restrict__ seed,
    int32_t scale_factor,
    __half* __restrict__ out) {
  constexpr int R = 128;
  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_chunks = (rows * R) / 32;
  if (chunk >= num_chunks) return;
  uint32_t key_r[8], seed_r[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) { key_r[i] = key[i]; seed_r[i] = seed[i]; }
  uint32_t msg[16];
  pearl::sm80::noise_gen::detail::make_message(seed_r, msg, /*slot=*/0,
      static_cast<uint32_t>(chunk + 1));
  uint32_t cv[8];
  pearl::sm80::noise_gen::detail::keyed_compress(key_r, msg, cv);
  const int row = (chunk * 32) / R;
  const int col0 = (chunk * 32) % R;
  __half* out_row = out + row * R + col0;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t w = cv[i];
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
      int8_t hb = static_cast<int8_t>(w & 0xff);
      w >>= 8;
      int32_t v = (static_cast<int32_t>(hb) + 128) % 64 - 32;
      float scaled = static_cast<float>(v * scale_factor);
      out_row[i * 4 + b] = __float2half(scaled);
    }
  }
}

extern "C" __global__ void pearl_noise_gen_sparse_R128(
    int k,
    const uint32_t* __restrict__ key,
    const uint32_t* __restrict__ seed,
    int8_t* __restrict__ out_r_major) {
  constexpr int R = 128;
  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_chunks = (k + 7) / 8;
  if (chunk >= num_chunks) return;
  uint32_t key_r[8], seed_r[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) { key_r[i] = key[i]; seed_r[i] = seed[i]; }
  uint32_t msg[16];
  pearl::sm80::noise_gen::detail::make_message(seed_r, msg, /*slot=*/1,
      static_cast<uint32_t>(chunk + 1));
  uint32_t cv[8];
  pearl::sm80::noise_gen::detail::keyed_compress(key_r, msg, cv);
  const int k_base = chunk * 8;
  #pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int row = k_base + j;
    if (row >= k) return;
    uint32_t u = cv[j];
    uint32_t k0 = u & static_cast<uint32_t>(R - 1);
    uint32_t k1 = k0 ^ (1u + pearl::sm80::noise_gen::detail::mul_hi_u32(
        static_cast<uint32_t>(R - 1), u));
    k1 = k1 % R;
    int8_t* out_row = out_r_major + row * R;
    out_row[k0] = 1;
    out_row[k1] = -1;
  }
}

// Tiny transpose: (k, R) int8 -> (R, k) int8. Used for K-major sparse variants.
// Not templated — k and R are runtime args.
extern "C" __global__ void pearl_transpose_kr_kernel(
    int k, int R,
    const int8_t* __restrict__ src_kr,
    int8_t* __restrict__ dst_rk) {
  const int kk = blockIdx.x * blockDim.x + threadIdx.x;
  const int rr = blockIdx.y * blockDim.y + threadIdx.y;
  if (kk >= k || rr >= R) return;
  dst_rk[rr * k + kk] = src_kr[kk * R + rr];
}

// ----- Commitment hash -----

extern "C" __global__ void pearl_commitment_hash_kernel(
    const uint8_t* __restrict__ A_root,
    const uint8_t* __restrict__ B_root,
    const uint8_t* __restrict__ key,
    uint8_t* __restrict__ A_commit,
    uint8_t* __restrict__ B_commit) {
  using namespace pearl::sm80::blake3;
  const uint32_t* key_u32   = reinterpret_cast<const uint32_t*>(key);
  const uint32_t* A_u32     = reinterpret_cast<const uint32_t*>(A_root);
  const uint32_t* B_u32     = reinterpret_cast<const uint32_t*>(B_root);
  uint32_t*       A_out_u32 = reinterpret_cast<uint32_t*>(A_commit);
  uint32_t*       B_out_u32 = reinterpret_cast<uint32_t*>(B_commit);

  constexpr CompressParams params{0u, MSG_BLOCK_SIZE,
                                  CHUNK_START | CHUNK_END | ROOT};

  uint32_t blockB[16];
  uint32_t cvB[8] = {IV0, IV1, IV2, IV3, IV4, IV5, IV6, IV7};
  for (int i = 0; i < 8; ++i) blockB[i]     = key_u32[i];
  for (int i = 0; i < 8; ++i) blockB[i + 8] = B_u32[i];
  compress_msg_block_u32(blockB, cvB, params);

  uint32_t blockA[16];
  uint32_t cvA[8] = {IV0, IV1, IV2, IV3, IV4, IV5, IV6, IV7};
  for (int i = 0; i < 8; ++i) blockA[i]     = cvB[i];
  for (int i = 0; i < 8; ++i) blockA[i + 8] = A_u32[i];
  compress_msg_block_u32(blockA, cvA, params);

  for (int i = 0; i < 8; ++i) B_out_u32[i] = cvB[i];
  for (int i = 0; i < 8; ++i) A_out_u32[i] = cvA[i];
}
