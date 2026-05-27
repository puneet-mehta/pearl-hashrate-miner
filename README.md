# pearl-hashrate-miner

A standalone Rust miner for the [Pearl](https://github.com/pearl-research-labs/pearl) PoW chain.

- **Single static binary** — no PyTorch, no Python at runtime
- **Drives CUDA kernels directly** via cudart (`cuModuleLoadFatBinary`,
  `cuLaunchKernel`); kernel SASS lives in a multi-arch fatbin built from `csrc/`
- **Multi-arch out of the box**: ships PTX for sm_86 / sm_89 / sm_120; one
  binary runs on RTX 3090 / 4090 / 5090
- **Multi-GPU**: one OS thread per device, shared pearld RPC / template polling
- **Small image**: distroless / debian-slim container in the ~30 MB range
  (vs ~4 GB for a PyTorch-based pipeline)

## Quick start

Requirements:

- NVIDIA driver supporting CUDA 12.8+ (any GPU with compute capability ≥ 8.6)
- A running [`pearld`](https://github.com/pearl-research-labs/pearl) node you can
  reach over JSON-RPC

### Docker (recommended)

```bash
docker build -t pearl-miner-live .
docker run --gpus all --rm \
    -e PEARLD_RPC_URL=http://<pearld-host>:<port> \
    -e PEARLD_MINING_ADDRESS=<your-bech32m-address> \
    pearl-miner-live
```

### From source

```bash
# 1. Build the multi-arch pearl-gemm fatbin (requires nvcc 12.4+).
./csrc/build_fatbin.sh                   # writes /tmp/pearl_gemm.fatbin

# 2. Build the Rust binary.
cargo build --release --bin pearl-miner-live

# 3. Run.
PEARLD_RPC_URL=http://<pearld-host>:<port> \
PEARLD_MINING_ADDRESS=<your-bech32m-address> \
PEARL_FATBIN=/tmp/pearl_gemm.fatbin \
    ./target/release/pearl-miner-live
```

## Configuration

All configuration is via environment variables.

| Variable                 | Default                          | Effect                                                              |
|--------------------------|----------------------------------|---------------------------------------------------------------------|
| `PEARLD_RPC_URL`         | `http://0.0.0.0:44107`           | pearld JSON-RPC endpoint                                            |
| `PEARLD_MINING_ADDRESS`  | (required)                       | Taproot bech32m payout address                                      |
| `PEARL_FATBIN`           | `/opt/fatbin/pearl_gemm.fatbin`  | path to the pearl-gemm fatbin                                       |
| `PEARL_DEVICES`          | all visible                      | comma-separated device ordinals (e.g. `0,2`)                        |
| `PEARL_WORKER_ID`        | random 16 bytes                  | salt mixed into per-job seed                                        |
| `WORKER_NAME`            | random 16 bytes                  | pool convention; salt mixed into per-job seed                       |
| `TEMPLATE_POLL_SECS`     | `1`                              | `get_block_template` poll cadence (single background thread)        |
| `MAX_ITERS`              | `0` (unbounded)                  | per-worker iteration cap (for benchmarking / smoke tests)           |
| `PEARL_SMALL_M`          | unset                            | small-M shape (m=8192 n=32768 k=2048)                               |
| `PEARL_BIG_M`            | unset                            | m=16384 n=32768                                                     |
| `PEARL_BIG_MN`           | unset                            | m=16384 n=65536                                                     |
| `PEARL_GIGA`             | unset                            | m=32768 n=65536                                                     |
| `PEARL_CPP_SEARCH`       | unset                            | legacy C++ search path (m=2048 n=28672 k=4096)                      |

The default shape is `m=32768 n=32768 k=2048 r=128` (huge-M), which gives the
best SM utilization on Ada / Blackwell-class GeForce.

## Multi-GPU

`pearl-miner-live` picks up every CUDA device visible to the process and runs
one worker thread per GPU. The pearld RPC client, the `get_block_template`
poller, and the hit-submit path are all shared, so multi-GPU does not multiply
pearld traffic.

```
[gpu0] iter 12000 ...
[gpu1] iter 11800 ...
[live-miner] AGG  ~170 iter/s
```

`PEARL_DEVICES=0,2` overrides the auto-detected list. `CUDA_VISIBLE_DEVICES`
also works as you'd expect. `MAX_ITERS` is per-worker.

## Layout

```
src/
  lib.rs              # crate root
  error.rs            # MinerError
  driver.rs           # safe wrappers over cudarc::driver::sys
                      # + CapturedGraph with mutable random_int8 node
  fatbin.rs           # load .nv_fatbin blob; stable extern-C symbol names
  miner_bufs.rs       # MinerBufs: ring buffer, kernel handles, shape configs
  kernels/            # one module per kernel
  gateway/            # pearld JSON-RPC client + MiningJob builder
  proof/              # PlainProof builder + signal-header parser
  bin/
    miner.rs              # kernel smoke tests
    triton_smoke.rs       # Triton dispatch smoke test
    live_miner.rs         # end-to-end live miner (default binary)
    bench_breakdown.rs    # per-phase timing for tuning

csrc/
  build_fatbin.sh           # nvcc → multi-arch pearl_gemm.fatbin
  pearl_gemm_kernels_only.cu
  pearl_search_norotl_blackwell.cu
  extern_c_shims.inc
  pearl_gemm/               # vendored CUDA kernel headers

triton_kernels/
  sm86/   noising_kernel.ptx + pearl_search_norotl_kernel.ptx
  sm89/   noising_kernel.ptx + pearl_search_norotl_kernel.ptx
  sm120/  noising_kernel.ptx + pearl_search_norotl_kernel.ptx

deps/
  zk-pow/        # vendored Pearl ZK prover crate
  pearl-blake3/  # vendored Pearl Blake3 helpers
  plonky2/       # vendored plonky2 fork used by zk-pow
```

## Triton kernels

Triton kernels are AOT-compiled once per supported arch and embedded into the
Rust binary via `include_bytes!`. At startup `MinerBufs::new` queries the
device's compute capability and loads the matching PTX (exact match preferred,
falling back to the highest arch ≤ device). PTX is forward-compatible so the
worst case is a JIT step at module load.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

The vendored `deps/plonky2` carries its own MIT/Apache-2.0 license from its
upstream.
