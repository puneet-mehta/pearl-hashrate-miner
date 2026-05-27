// blake3_sm80.cuh
//
// Standalone Ampere/Ada port of `csrc/blake3/blake3.cuh`.
//
// The upstream Blake3 implementation is pure 32-bit integer math
// (add / xor / rotate-right) and has no Hopper-specific instructions.
// The only thing that needed lifting was the CUTLASS Tensor abstraction
// the upstream uses to wrap register arrays; here we use raw `uint32_t[]`
// instead. The algorithm — operation count and order — is preserved
// byte-for-byte, so output is bit-identical to upstream by construction.
//
// Notes from upstream we preserve here:
//   * 7 rounds total: 6 (round + permute) followed by one round without
//     permute. Standard Blake3 does 7 rounds + 6 permutes.
//   * After the rounds, only the first 8 chaining-value words are written
//     back (state[i] ^ state[i+8]). The remaining 8 words that real Blake3
//     also XORs the original chaining value into are intentionally NOT
//     touched, because the upstream kernel never reads them. This means
//     this is a *truncated* Blake3 with a 32-byte digest, NOT the full
//     32-or-larger XOF.
//
// Both properties carry through to the sm_80 port; otherwise the network
// would reject shares.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pearl::sm80::blake3 {

// Sizes / flag bits (mirror blake3_constants.hpp).
inline constexpr uint32_t CHAINING_VALUE_SIZE      = 32;
inline constexpr uint32_t CHAINING_VALUE_SIZE_U32  = CHAINING_VALUE_SIZE / sizeof(uint32_t); // 8
inline constexpr uint32_t MSG_BLOCK_SIZE           = 64;
inline constexpr uint32_t MSG_BLOCK_SIZE_U32       = MSG_BLOCK_SIZE / sizeof(uint32_t);     // 16

inline constexpr uint32_t CHUNK_START          = 1u << 0;
inline constexpr uint32_t CHUNK_END            = 1u << 1;
inline constexpr uint32_t PARENT               = 1u << 2;
inline constexpr uint32_t ROOT                 = 1u << 3;
inline constexpr uint32_t KEYED_HASH           = 1u << 4;
inline constexpr uint32_t DERIVE_KEY_CONTEXT   = 1u << 5;
inline constexpr uint32_t DERIVE_KEY_MATERIAL  = 1u << 6;

inline constexpr uint32_t IV0 = 0x6A09E667u;
inline constexpr uint32_t IV1 = 0xBB67AE85u;
inline constexpr uint32_t IV2 = 0x3C6EF372u;
inline constexpr uint32_t IV3 = 0xA54FF53Au;
inline constexpr uint32_t IV4 = 0x510E527Fu;
inline constexpr uint32_t IV5 = 0x9B05688Cu;
inline constexpr uint32_t IV6 = 0x1F83D9ABu;
inline constexpr uint32_t IV7 = 0x5BE0CD19u;

struct CompressParams {
  uint64_t counter;
  uint32_t block_len;
  uint32_t flags;
};

// The three CompressParams pre-sets the upstream kernel uses.
__device__ __host__ inline constexpr CompressParams make_inner_node_params() {
  return CompressParams{0u, MSG_BLOCK_SIZE, KEYED_HASH | PARENT};
}
__device__ __host__ inline constexpr CompressParams make_root_params() {
  return CompressParams{0u, MSG_BLOCK_SIZE, KEYED_HASH | ROOT | PARENT};
}
__device__ __host__ inline constexpr CompressParams make_single_block_keyed_params() {
  return CompressParams{0u, MSG_BLOCK_SIZE,
                        KEYED_HASH | CHUNK_START | CHUNK_END | ROOT};
}

// Host-and-device 32-bit add / right-rotate. The arithmetic is identical on
// both sides — no FP, no rounding — so a CPU reference using these same
// helpers must produce byte-identical results.
__device__ __host__ __forceinline__ uint32_t add32(uint32_t x, uint32_t y) {
  return x + y;
}
__device__ __host__ __forceinline__ uint32_t rightrotate32(uint32_t x,
                                                           uint32_t n) {
  // n is always in (0, 32) for Blake3's rotation constants {16, 12, 8, 7}.
  return (x << (32u - n)) | (x >> n);
}

// One Blake3 round: 8 G operations laid out explicitly so the compiler can
// keep state in registers. Byte-for-byte copy of the upstream macro, with
// the upstream's `rState(i)` Tensor access rewritten as `rState[i]`.
#define PEARL_SM80_BLAKE3_ROUND(rState, rBlock)               \
  do {                                                         \
    rState[0]  = add32(rState[0],  add32(rState[4],  rBlock[0])); \
    rState[12] = rightrotate32(rState[12] ^ rState[0],  16);     \
    rState[8]  = add32(rState[8],  rState[12]);                  \
    rState[4]  = rightrotate32(rState[4]  ^ rState[8],  12);     \
    rState[0]  = add32(rState[0],  add32(rState[4],  rBlock[1])); \
    rState[12] = rightrotate32(rState[12] ^ rState[0],   8);     \
    rState[8]  = add32(rState[8],  rState[12]);                  \
    rState[4]  = rightrotate32(rState[4]  ^ rState[8],   7);     \
    rState[1]  = add32(rState[1],  add32(rState[5],  rBlock[2])); \
    rState[13] = rightrotate32(rState[13] ^ rState[1],  16);     \
    rState[9]  = add32(rState[9],  rState[13]);                  \
    rState[5]  = rightrotate32(rState[5]  ^ rState[9],  12);     \
    rState[1]  = add32(rState[1],  add32(rState[5],  rBlock[3])); \
    rState[13] = rightrotate32(rState[13] ^ rState[1],   8);     \
    rState[9]  = add32(rState[9],  rState[13]);                  \
    rState[5]  = rightrotate32(rState[5]  ^ rState[9],   7);     \
    rState[2]  = add32(rState[2],  add32(rState[6],  rBlock[4])); \
    rState[14] = rightrotate32(rState[14] ^ rState[2],  16);     \
    rState[10] = add32(rState[10], rState[14]);                  \
    rState[6]  = rightrotate32(rState[6]  ^ rState[10], 12);     \
    rState[2]  = add32(rState[2],  add32(rState[6],  rBlock[5])); \
    rState[14] = rightrotate32(rState[14] ^ rState[2],   8);     \
    rState[10] = add32(rState[10], rState[14]);                  \
    rState[6]  = rightrotate32(rState[6]  ^ rState[10],  7);     \
    rState[3]  = add32(rState[3],  add32(rState[7],  rBlock[6])); \
    rState[15] = rightrotate32(rState[15] ^ rState[3],  16);     \
    rState[11] = add32(rState[11], rState[15]);                  \
    rState[7]  = rightrotate32(rState[7]  ^ rState[11], 12);     \
    rState[3]  = add32(rState[3],  add32(rState[7],  rBlock[7])); \
    rState[15] = rightrotate32(rState[15] ^ rState[3],   8);     \
    rState[11] = add32(rState[11], rState[15]);                  \
    rState[7]  = rightrotate32(rState[7]  ^ rState[11],  7);     \
    rState[0]  = add32(rState[0],  add32(rState[5],  rBlock[8])); \
    rState[15] = rightrotate32(rState[15] ^ rState[0],  16);     \
    rState[10] = add32(rState[10], rState[15]);                  \
    rState[5]  = rightrotate32(rState[5]  ^ rState[10], 12);     \
    rState[0]  = add32(rState[0],  add32(rState[5],  rBlock[9])); \
    rState[15] = rightrotate32(rState[15] ^ rState[0],   8);     \
    rState[10] = add32(rState[10], rState[15]);                  \
    rState[5]  = rightrotate32(rState[5]  ^ rState[10],  7);     \
    rState[1]  = add32(rState[1],  add32(rState[6],  rBlock[10])); \
    rState[12] = rightrotate32(rState[12] ^ rState[1],  16);     \
    rState[11] = add32(rState[11], rState[12]);                  \
    rState[6]  = rightrotate32(rState[6]  ^ rState[11], 12);     \
    rState[1]  = add32(rState[1],  add32(rState[6],  rBlock[11])); \
    rState[12] = rightrotate32(rState[12] ^ rState[1],   8);     \
    rState[11] = add32(rState[11], rState[12]);                  \
    rState[6]  = rightrotate32(rState[6]  ^ rState[11],  7);     \
    rState[2]  = add32(rState[2],  add32(rState[7],  rBlock[12])); \
    rState[13] = rightrotate32(rState[13] ^ rState[2],  16);     \
    rState[8]  = add32(rState[8],  rState[13]);                  \
    rState[7]  = rightrotate32(rState[7]  ^ rState[8],  12);     \
    rState[2]  = add32(rState[2],  add32(rState[7],  rBlock[13])); \
    rState[13] = rightrotate32(rState[13] ^ rState[2],   8);     \
    rState[8]  = add32(rState[8],  rState[13]);                  \
    rState[7]  = rightrotate32(rState[7]  ^ rState[8],   7);     \
    rState[3]  = add32(rState[3],  add32(rState[4],  rBlock[14])); \
    rState[14] = rightrotate32(rState[14] ^ rState[3],  16);     \
    rState[9]  = add32(rState[9],  rState[14]);                  \
    rState[4]  = rightrotate32(rState[4]  ^ rState[9],  12);     \
    rState[3]  = add32(rState[3],  add32(rState[4],  rBlock[15])); \
    rState[14] = rightrotate32(rState[14] ^ rState[3],   8);     \
    rState[9]  = add32(rState[9],  rState[14]);                  \
    rState[4]  = rightrotate32(rState[4]  ^ rState[9],   7);     \
  } while (0)

// Standard Blake3 message-word permutation. Byte-for-byte from upstream.
#define PEARL_SM80_BLAKE3_PERMUTE(rBlock)            \
  do {                                                \
    uint32_t tmp[16];                                 \
    for (int _pi = 0; _pi < 16; ++_pi) tmp[_pi] = rBlock[_pi]; \
    rBlock[0]  = tmp[2];   rBlock[1]  = tmp[6];                \
    rBlock[2]  = tmp[3];   rBlock[3]  = tmp[10];               \
    rBlock[4]  = tmp[7];   rBlock[5]  = tmp[0];                \
    rBlock[6]  = tmp[4];   rBlock[7]  = tmp[13];               \
    rBlock[8]  = tmp[1];   rBlock[9]  = tmp[11];               \
    rBlock[10] = tmp[12];  rBlock[11] = tmp[5];                \
    rBlock[12] = tmp[9];   rBlock[13] = tmp[14];               \
    rBlock[14] = tmp[15];  rBlock[15] = tmp[8];                \
  } while (0)

// Compress one 64-byte block into the 8-word chaining value, in place.
// Works on both host and device — same arithmetic, same control flow.
//
// Inputs:
//   block[16]: input message block (will be modified internally; caller
//              should not rely on its post-call value).
//   chaining_value[8]: keyed-hash starting state. Replaced with the
//              compressed output (8 u32 = 32 bytes digest).
//   params:    counter / block_len / flags. Must use one of the
//              make_*_params() variants or a custom CompressParams.
__device__ __host__ inline void compress_msg_block_u32(
    uint32_t block[16], uint32_t chaining_value[8],
    const CompressParams params) {
  uint32_t rState[16];
  uint32_t rBlock[16];

  // Working block starts as a copy of the input block.
  for (int i = 0; i < 16; ++i) rBlock[i] = block[i];

  // State = (chaining_value, IV[0..4], counter_lo, counter_hi, block_len, flags)
  for (int i = 0; i < 8; ++i) rState[i] = chaining_value[i];
  rState[8]  = IV0;
  rState[9]  = IV1;
  rState[10] = IV2;
  rState[11] = IV3;
  rState[12] = static_cast<uint32_t>(params.counter);
  rState[13] = static_cast<uint32_t>(params.counter >> 32);
  rState[14] = params.block_len;
  rState[15] = params.flags;

  // 6 rounds + permute, then a 7th round w/o permute (mirrors upstream).
  #pragma unroll
  for (int r = 0; r < 6; ++r) {
    PEARL_SM80_BLAKE3_ROUND(rState, rBlock);
    PEARL_SM80_BLAKE3_PERMUTE(rBlock);
  }
  PEARL_SM80_BLAKE3_ROUND(rState, rBlock);

  // Write back only the lower 8 chaining-value words, as upstream does.
  // (Real Blake3 also XORs the original chaining value into state[8..15];
  // upstream does not, and we mirror that on purpose.)
  chaining_value[0] = rState[0] ^ rState[8];
  chaining_value[1] = rState[1] ^ rState[9];
  chaining_value[2] = rState[2] ^ rState[10];
  chaining_value[3] = rState[3] ^ rState[11];
  chaining_value[4] = rState[4] ^ rState[12];
  chaining_value[5] = rState[5] ^ rState[13];
  chaining_value[6] = rState[6] ^ rState[14];
  chaining_value[7] = rState[7] ^ rState[15];
}

}  // namespace pearl::sm80::blake3
