# Multi-stage Dockerfile for pearl-miner-live.
#
# Builds:
#   1. pearl_gemm.fatbin from csrc/ (SASS for sm_86 / sm_89 / sm_120 + sm_86 PTX fallback)
#   2. the Rust miner binary (release, LTO, stripped)
# and packages both on debian:12-slim alongside libcudart.
#
# At runtime the container needs the host NVIDIA driver bind-mounted via
# nvidia-container-runtime (libcuda.so) — pass `--gpus all` to docker.
#
# Build:
#   docker buildx build --platform linux/amd64 -t pearl-miner-live .
#
# Run:
#   docker run --gpus all --rm \
#       -e PEARLD_RPC_URL=http://<host>:<port> \
#       -e PEARLD_MINING_ADDRESS=<bech32m-addr> \
#       pearl-miner-live

# ─── Builder ───────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl pkg-config binutils libssl-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal

COPY . /src
WORKDIR /src

# 1. Compile pearl-gemm kernels to a multi-arch fatbin.
RUN ./csrc/build_fatbin.sh && ls -lh /tmp/pearl_gemm.fatbin

# 2. Build the Rust miner.
RUN cargo build --release --bin pearl-miner-live

# ─── Runtime ───────────────────────────────────────────────────────────────
FROM debian:12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgcc-s1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/cuda-12.9/targets/x86_64-linux/lib/libcudart.so* \
                    /usr/local/cuda/lib64/
COPY --from=builder /src/target/release/pearl-miner-live \
                    /usr/local/bin/pearl-miner-live
COPY --from=builder /tmp/pearl_gemm.fatbin /opt/fatbin/pearl_gemm.fatbin

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    PEARL_FATBIN=/opt/fatbin/pearl_gemm.fatbin

ENTRYPOINT ["/usr/local/bin/pearl-miner-live"]
