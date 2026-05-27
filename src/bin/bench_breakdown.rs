//! Coarse iter timing — eager vs captured-graph paths.

use std::process::ExitCode;
use std::time::Instant;

use cudarc::driver::sys as cu;

use pearl_hashrate_miner::driver::{CudaCtx, Stream};
use pearl_hashrate_miner::error::cu_check;
use pearl_hashrate_miner::fatbin::load_fatbin_file;
use pearl_hashrate_miner::{MinerBufs, MinerBufsConfig};

const N_ITERS: usize = 4000;
const N_WARMUP: u64 = 50;

fn main() -> ExitCode {
    let fatbin_path = std::env::var("PEARL_FATBIN")
        .unwrap_or_else(|_| "/opt/fatbin/pearl_gemm.fatbin".to_string());
    let ctx = CudaCtx::new(0).expect("CudaCtx");
    println!("device: {}", ctx.device_name().unwrap());
    let module = load_fatbin_file(&fatbin_path).expect("fatbin");
    let cfg = MinerBufsConfig::shape_default();
    let mut bufs = MinerBufs::new(&module, cfg).expect("MinerBufs");
    println!("shape m={} n={} k={} r={}", bufs.m, bufs.n, bufs.k, bufs.r);

    let stream = Stream::new().expect("stream");
    let s = stream.handle;
    let fake_header = b"bench-breakdown-fake-header";
    let key: [u8; 32] = std::array::from_fn(|i| (i * 7) as u8);
    let target: [u8; 32] = [0xFF; 32];
    let seed: [u8; 32] = std::array::from_fn(|i| (i * 11 + 3) as u8);
    unsafe {
        bufs.ensure_for_job(fake_header, &key, &target, &seed, 0, s)
            .expect("ensure");
        stream.synchronize().expect("sync");
    }

    // Warm up eager
    for i in 0..N_WARMUP {
        unsafe {
            bufs.mine_one(i, s).expect("warm");
        }
    }
    stream.synchronize().expect("sync");

    let macs_tn = (bufs.m * bufs.n * bufs.k) as f64 / 1e12;

    // A) Pipelined eager — 200 iters, sync once
    let t0 = Instant::now();
    for i in 0..N_ITERS as u64 {
        unsafe {
            bufs.mine_one(i, s).expect("eager");
        }
    }
    stream.synchronize().expect("sync");
    let eager_ms = t0.elapsed().as_secs_f64() * 1000.0 / N_ITERS as f64;
    println!(
        "\nA) eager (pipelined): {:.3} ms/iter = {:.1} TH/s",
        eager_ms,
        macs_tn / (eager_ms / 1000.0)
    );

    // B) Eager batched: 8 iters then sync (mirrors live_miner batch structure)
    let t0 = Instant::now();
    let n_batches = N_ITERS / 8;
    for b in 0..n_batches {
        for j in 0..8u64 {
            unsafe {
                bufs.mine_one(b as u64 * 8 + j, s).expect("eb");
            }
        }
        stream.synchronize().expect("sync");
    }
    let batched_ms = t0.elapsed().as_secs_f64() * 1000.0 / (n_batches * 8) as f64;
    println!(
        "B) eager (batch=8): {:.3} ms/iter = {:.1} TH/s",
        batched_ms,
        macs_tn / (batched_ms / 1000.0)
    );

    // C) Captured graph replay (uses cuGraphExecKernelNodeSetParams for iter_idx)
    let mut graphs = unsafe { bufs.capture_all_slots(&stream).expect("cap") };
    // Warm up
    for i in 0..N_WARMUP {
        unsafe {
            bufs.mine_one_with_graphs(i, &mut graphs, &stream)
                .expect("g_warm");
        }
    }
    stream.synchronize().expect("sync");

    // C1) Pipelined graph: 200 iters, sync once
    let t0 = Instant::now();
    for i in 0..N_ITERS as u64 {
        unsafe {
            bufs.mine_one_with_graphs(i + 100, &mut graphs, &stream)
                .expect("g");
        }
    }
    stream.synchronize().expect("sync");
    let graph_ms = t0.elapsed().as_secs_f64() * 1000.0 / N_ITERS as f64;
    println!(
        "C1) graph (pipelined): {:.3} ms/iter = {:.1} TH/s",
        graph_ms,
        macs_tn / (graph_ms / 1000.0)
    );

    // C2) Graph batched: 8 iters then sync (mirrors live_miner exactly)
    let t0 = Instant::now();
    for b in 0..n_batches {
        for j in 0..8u64 {
            let i = 100 + b as u64 * 8 + j;
            unsafe {
                bufs.mine_one_with_graphs(i, &mut graphs, &stream)
                    .expect("gb");
            }
        }
        stream.synchronize().expect("sync");
    }
    let graph_batched_ms = t0.elapsed().as_secs_f64() * 1000.0 / (n_batches * 8) as f64;
    println!(
        "C2) graph (batch=8):  {:.3} ms/iter = {:.1} TH/s",
        graph_batched_ms,
        macs_tn / (graph_batched_ms / 1000.0)
    );

    println!(
        "\n  eager pipelined: {:.3}  vs  graph pipelined: {:.3}  (Δ {:.3} ms)",
        eager_ms,
        graph_ms,
        graph_ms - eager_ms
    );
    println!(
        "  eager batched=8: {:.3}  vs  graph batched=8: {:.3}  (Δ {:.3} ms)",
        batched_ms,
        graph_batched_ms,
        graph_batched_ms - batched_ms
    );

    drop(graphs);
    drop(bufs);
    ExitCode::SUCCESS
}
