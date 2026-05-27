//! pearl-miner-live — end-to-end Rust miner (no gateway process).
//!
//! Auto-detects N visible GPUs and spawns one worker thread per device,
//! each holding its own CUDA context. Workers share a single
//! [`PearldClient`], a single template-poller thread, and a single
//! hit-poller thread. The default is to use every visible device;
//! `PEARL_DEVICES=0,2` overrides to a specific subset, and
//! `CUDA_VISIBLE_DEVICES` works as it always has.
//!
//! Env config:
//!   PEARLD_RPC_URL          (default http://0.0.0.0:44107)
//!   PEARLD_RPC_USER         (default user)
//!   PEARLD_RPC_PASSWORD     (default pass)
//!   PEARLD_MINING_ADDRESS   (required — Taproot bech32m address)
//!   PEARL_FATBIN            (default /opt/fatbin/pearl_gemm.fatbin)
//!   PEARL_DEVICES           (default all visible — e.g. "0" or "0,2,3")
//!   TEMPLATE_POLL_SECS      (default 1)
//!   MAX_ITERS               (default 0 = unbounded; PER-WORKER cap)
//!
//! Hit detection is synchronous per-batch (sync, scan, snapshot, queue).
//! The async-event polling path is a follow-up; this wiring favors
//! correctness over the last 10% of speed.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use bitcoin::consensus::Decodable;

use pearl_hashrate_miner::driver::{device_count, CapturedGraph, CudaCtx, Module, Stream};
use pearl_hashrate_miner::gateway::{
    build_mining_config_cpp_search, build_mining_config_triton_norotl, create_coinbase_transaction,
    BlockTemplateResponse, MiningJob, PearldClient, PearldConfig,
};
use pearl_hashrate_miner::miner_bufs::{MiningPath, HOST_SIGNAL_HEADER_SIZE};
use pearl_hashrate_miner::proof::{
    build_plain_proof, extract_indices, signal_header::ParsedSignalHeader, submit_plain_proof,
};
use pearl_hashrate_miner::{MinerBufs, MinerBufsConfig, MinerError};

use zk_pow::circuit::pearl_circuit::{PearlRecursion, RecursionCircuit};

// ============================================================================
//   Hit work — owned snapshots for the poller thread.
// ============================================================================

struct HitWork {
    job: Arc<ReadyJob>,
    a_bytes: Vec<u8>,
    a_rows: Vec<usize>,
    b_cols: Vec<usize>,
    src_gpu: i32,
}

struct ReadyJob {
    job: MiningJob,
    b_bytes: Vec<u8>,
    m: usize,
    n: usize,
}

// ============================================================================
//   Hit poller — one thread, drains hits from all workers.
// ============================================================================

fn hit_poller(rx: mpsc::Receiver<HitWork>, client: Arc<PearldClient>) {
    let mut circuit_cache = <PearlRecursion as RecursionCircuit>::CircuitCache::default();
    let mut blocks_accepted: u64 = 0;
    let mut blocks_rejected: u64 = 0;

    while let Ok(work) = rx.recv() {
        let job = &work.job.job;
        let started = Instant::now();
        let plain_proof = match build_plain_proof(
            work.job.m,
            work.job.n,
            job.mining_config.common_dim as usize,
            job.mining_config.rank as usize,
            &work.a_bytes,
            &work.job.b_bytes,
            work.a_rows,
            work.b_cols,
            job.key,
        ) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[poller] gpu{} PlainProof build failed: {e}", work.src_gpu);
                continue;
            }
        };
        match submit_plain_proof(&plain_proof, job, &client, &mut circuit_cache) {
            Ok(()) => {
                blocks_accepted += 1;
                println!(
                    "[poller] gpu{} ACCEPTED (zk+rpc {:.1}s, accepted={} rejected={})",
                    work.src_gpu,
                    started.elapsed().as_secs_f64(),
                    blocks_accepted,
                    blocks_rejected,
                );
            }
            Err(e) => {
                blocks_rejected += 1;
                eprintln!(
                    "[poller] gpu{} REJECTED ({:.1}s): {e} (accepted={} rejected={})",
                    work.src_gpu,
                    started.elapsed().as_secs_f64(),
                    blocks_accepted,
                    blocks_rejected,
                );
            }
        }
    }
}

// ============================================================================
//   Template parse / job build
// ============================================================================

fn parse_template_txs(t: &BlockTemplateResponse) -> Result<Vec<bitcoin::Transaction>, MinerError> {
    t.transactions
        .iter()
        .map(|tx| {
            let raw = hex::decode(&tx.data).map_err(|e| MinerError::Rpc {
                method: "tx_decode".to_string(),
                msg: format!("hex: {e}"),
            })?;
            bitcoin::Transaction::consensus_decode(&mut std::io::Cursor::new(&raw[..])).map_err(
                |e| MinerError::Rpc {
                    method: "tx_decode".to_string(),
                    msg: format!("consensus: {e}"),
                },
            )
        })
        .collect()
}

fn build_job(
    t: BlockTemplateResponse,
    mining_address: &str,
    path: MiningPath,
) -> Result<MiningJob, MinerError> {
    let other_txs = parse_template_txs(&t)?;
    let aux: Option<Vec<u8>> = if t.coinbaseaux.flags.is_empty() {
        None
    } else {
        Some(
            hex::decode(&t.coinbaseaux.flags).map_err(|e| MinerError::Rpc {
                method: "coinbase_aux".to_string(),
                msg: format!("hex: {e}"),
            })?,
        )
    };
    let witness: Option<[u8; 32]> = match &t.default_witness_commitment {
        Some(s) => {
            let b = hex::decode(s).map_err(|e| MinerError::Rpc {
                method: "witness".to_string(),
                msg: format!("hex: {e}"),
            })?;
            if b.len() != 32 {
                return Err(MinerError::Rpc {
                    method: "witness".to_string(),
                    msg: format!("len {}", b.len()),
                });
            }
            let mut a = [0u8; 32];
            a.copy_from_slice(&b);
            Some(a)
        }
        None => None,
    };
    let coinbase_tx = create_coinbase_transaction(
        t.height,
        t.coinbasevalue,
        mining_address,
        aux.as_deref(),
        witness.as_ref(),
    )?;
    let mining_config = match path {
        MiningPath::CppSearch => build_mining_config_cpp_search(4096, 128)?,
        MiningPath::TritonNorotl => build_mining_config_triton_norotl(2048, 128)?,
    };
    MiningJob::build(t, coinbase_tx, other_txs, mining_config)
}

// ============================================================================
//   Config selection
// ============================================================================

fn pick_config() -> MinerBufsConfig {
    if std::env::var("PEARL_CPP_SEARCH").is_ok() {
        println!("[live-miner] path: CppSearch (legacy, ~39 TH/s on 4090)");
        MinerBufsConfig::production()
    } else if std::env::var("PEARL_SMALL_M").is_ok() {
        println!("[live-miner] path: TritonNorotl (m=8192 baseline)");
        MinerBufsConfig::shape_default()
    } else if std::env::var("PEARL_49K").is_ok() {
        println!("[live-miner] path: TritonNorotl 49k (m=49152 n=49152)");
        MinerBufsConfig::shape_49k_n()
    } else if std::env::var("PEARL_49K_M").is_ok() {
        println!("[live-miner] path: TritonNorotl 49k_m (m=49152 n=32768)");
        MinerBufsConfig::shape_49k_m()
    } else if std::env::var("PEARL_GIGA").is_ok() {
        println!("[live-miner] path: TritonNorotl GIGA (m=32768 n=65536)");
        MinerBufsConfig::shape_giga()
    } else if std::env::var("PEARL_BIG_MN").is_ok() {
        println!("[live-miner] path: TritonNorotl big-MN (m=16384 n=65536)");
        MinerBufsConfig::shape_big_mn()
    } else if std::env::var("PEARL_BIG_M").is_ok() {
        println!("[live-miner] path: TritonNorotl big-M (m=16384)");
        MinerBufsConfig::shape_big_m()
    } else {
        println!("[live-miner] path: TritonNorotl huge-M (m=32768 n=32768)");
        MinerBufsConfig::shape_huge_m()
    }
}

fn pick_devices() -> Result<Vec<i32>, MinerError> {
    let avail = device_count()?;
    if avail == 0 {
        return Err(MinerError::Rpc {
            method: "device_count".to_string(),
            msg: "no CUDA devices visible".to_string(),
        });
    }
    let devs: Vec<i32> = match std::env::var("PEARL_DEVICES") {
        Ok(s) => s
            .split(',')
            .filter_map(|t| t.trim().parse::<i32>().ok())
            .collect(),
        Err(_) => (0..avail).collect(),
    };
    for &d in &devs {
        if d < 0 || d >= avail {
            return Err(MinerError::Rpc {
                method: "PEARL_DEVICES".to_string(),
                msg: format!("device {d} out of range (have {avail})"),
            });
        }
    }
    if devs.is_empty() {
        return Err(MinerError::Rpc {
            method: "PEARL_DEVICES".to_string(),
            msg: "empty device list".to_string(),
        });
    }
    Ok(devs)
}

// ============================================================================
//   Worker salt — process-wide unique string mixed into per-job seed
// ============================================================================

/// Resolve a process-wide unique salt that distinguishes this miner
/// process from any other miner running against the same pool/template.
/// Order of preference:
///   1. `PEARL_WORKER_ID` — explicit operator override
///   2. `WORKER_NAME`     — pool convention (set by wrapper entrypoints)
///   3. 16 random bytes from `/dev/urandom`, hex-encoded — last-resort
///      uniqueness guarantee. Hostname is intentionally NOT used as a
///      fallback because containers on the same host (or replicas
///      launched from the same image) routinely share hostnames.
fn resolve_worker_salt() -> String {
    for var in ["PEARL_WORKER_ID", "WORKER_NAME"] {
        if let Ok(v) = std::env::var(var) {
            let v = v.trim();
            if !v.is_empty() {
                return v.to_string();
            }
        }
    }
    use std::io::Read;
    let mut buf = [0u8; 16];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        if f.read_exact(&mut buf).is_ok() {
            let mut s = String::with_capacity(32);
            for b in buf.iter() {
                s.push_str(&format!("{:02x}", b));
            }
            return s;
        }
    }
    // Last-resort fallback if /dev/urandom is unavailable: pid + nanos.
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("pid{pid}-{nanos}")
}

// ============================================================================
//   Per-worker mining loop — owns a CudaCtx + Module + MinerBufs on its thread
// ============================================================================

struct SharedTemplate {
    /// Most recent template the poller has fetched. Workers clone the Arc
    /// and compare prev-hash against their own to decide whether to rebuild.
    latest: Mutex<Option<Arc<BlockTemplateResponse>>>,
}

struct WorkerCtx {
    device_ord: i32,
    fatbin: Arc<Vec<u8>>,
    cfg: MinerBufsConfig,
    mining_address: String,
    template: Arc<SharedTemplate>,
    hit_tx: mpsc::Sender<HitWork>,
    total_iters: Arc<AtomicU64>,
    max_iters: u64,
    /// Process-wide unique salt mixed into the per-job seed so that
    /// multiple miner processes / boxes that see the same template don't
    /// also iterate the same A-space and find the same hits in lockstep
    /// (causing "already have block" duplicate submissions). Resolution
    /// order: PEARL_WORKER_ID > WORKER_NAME > 16 random bytes hex.
    worker_salt: Arc<String>,
}

fn worker(wctx: WorkerCtx) -> Result<(), MinerError> {
    // CudaCtx::new makes this device's new context the calling thread's
    // current context. Every subsequent CUDA call on this thread is on
    // this device.
    let ctx = CudaCtx::new(wctx.device_ord)?;
    let tag = format!("[gpu{}]", wctx.device_ord);
    println!("{tag} device: {}", ctx.device_name()?);

    let module = Module::load_fatbin(&wctx.fatbin)?;
    let mut bufs = MinerBufs::new(&module, wctx.cfg)?;
    println!(
        "{tag} MinerBufs: m={} n={} k={} r={}",
        bufs.m, bufs.n, bufs.k, bufs.r
    );
    let path = wctx.cfg.path;

    let stream = Stream::new()?;
    let mut graphs: Option<Vec<CapturedGraph>> = None;
    let mut current_job: Option<Arc<ReadyJob>> = None;
    let mut current_prev: Option<String> = None;

    let mut iter_idx: u64 = 0;
    let mut last_log = Instant::now();
    let mut iters_at_log = 0u64;
    let mut hits_total = 0u64;

    loop {
        // ---- 1. Check shared template cell for a newer tip ----
        let newest_tpl: Option<Arc<BlockTemplateResponse>> = {
            let guard = wctx.template.latest.lock().unwrap();
            guard.as_ref().map(Arc::clone)
        };
        if let Some(t) = newest_tpl {
            let need = match &current_prev {
                None => true,
                Some(p) => *p != t.previousblockhash,
            };
            if need {
                current_prev = Some(t.previousblockhash.clone());
                // We hold an Arc; build_job needs ownership of the response.
                // Take a clone of the inner only for the one we'll consume.
                let owned: BlockTemplateResponse = (*t).clone();
                let prev_new = owned.previousblockhash.clone();
                let job = build_job(owned, &wctx.mining_address, path)?;
                let header_bytes = job.header_bytes;
                let key = job.key;
                let target_le = job.adjusted_target_le;
                let seed = blake3::hash(
                    &[
                        &header_bytes[..],
                        b"pearl-rust-miner-seed-v1",
                        wctx.worker_salt.as_bytes(),
                        &(wctx.device_ord as u32).to_le_bytes()[..],
                    ]
                    .concat(),
                )
                .as_bytes()
                .to_owned();

                unsafe {
                    bufs.ensure_for_job(&header_bytes, &key, &target_le, &seed, 0, stream.handle)?;
                }
                stream.synchronize()?;

                let b_bytes = bufs.b_pinned.as_slice().to_vec();

                if graphs.is_none() {
                    let g = unsafe { bufs.capture_all_slots(&stream)? };
                    println!("{tag} captured {} per-slot graphs", g.len());
                    graphs = Some(g);
                }

                let height = job.template.height;
                let bits = job.template.bits.clone();
                let target_hex = hex::encode(&job.target_be[..8]);
                current_job = Some(Arc::new(ReadyJob {
                    job,
                    b_bytes,
                    m: bufs.m,
                    n: bufs.n,
                }));
                println!(
                    "{tag} new job: height={} prev={} bits={} target_be[0..8]={}",
                    height,
                    &prev_new[..prev_new.len().min(16)],
                    bits,
                    target_hex,
                );
            }
        }

        let job = match &current_job {
            Some(j) => Arc::clone(j),
            None => {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
        };
        let graphs_ref = graphs.as_mut().expect("graphs captured by job_load");

        // ---- 2. Batch of iters, sync, scan slots for hits ----
        let batch_start = iter_idx;
        for _ in 0..bufs.ring_size as u64 {
            unsafe {
                bufs.mine_one_with_graphs(iter_idx, graphs_ref, &stream)?;
            }
            iter_idx += 1;
        }
        stream.synchronize()?;

        let mut hits_this_batch: Vec<(u64, usize, Vec<u8>)> = Vec::new();
        for batch_iter in batch_start..iter_idx {
            let slot = bufs.slot(batch_iter);
            let header_src = bufs.host_signal_header_pool[slot].as_slice();
            let status = u32::from_le_bytes(header_src[0..4].try_into().unwrap());
            if status == 1 {
                hits_total += 1;
                let header_bytes =
                    header_src[..HOST_SIGNAL_HEADER_SIZE.min(header_src.len())].to_vec();
                unsafe {
                    bufs.snapshot_a_for_hit(slot, stream.handle)?;
                }
                hits_this_batch.push((batch_iter, slot, header_bytes));
            }
        }
        if !hits_this_batch.is_empty() {
            stream.synchronize()?;
            for (_batch_iter, slot, header_bytes) in hits_this_batch {
                let parsed = ParsedSignalHeader::parse(&header_bytes)?;
                let (a_rows, b_cols) = extract_indices(&parsed);
                let a_bytes = bufs.a_snapshot_pool[slot].as_slice().to_vec();
                let _ = wctx.hit_tx.send(HitWork {
                    job: Arc::clone(&job),
                    a_bytes,
                    a_rows,
                    b_cols,
                    src_gpu: wctx.device_ord,
                });
            }
        }

        // Bump the shared aggregate counter once per batch.
        wctx.total_iters
            .fetch_add(bufs.ring_size as u64, Ordering::Relaxed);

        if wctx.max_iters > 0 && iter_idx >= wctx.max_iters {
            let dt = last_log.elapsed().as_secs_f64();
            let d_iters = iter_idx - iters_at_log;
            println!(
                "{tag} FINAL iters={} rate={:.1}/s hits_total={}",
                iter_idx,
                d_iters as f64 / dt,
                hits_total,
            );
            break;
        }

        if last_log.elapsed() >= Duration::from_secs(5) {
            let dt = last_log.elapsed().as_secs_f64();
            let d_iters = iter_idx - iters_at_log;
            println!(
                "{tag} iters={} rate={:.1}/s hits_total={}",
                iter_idx,
                d_iters as f64 / dt,
                hits_total,
            );
            last_log = Instant::now();
            iters_at_log = iter_idx;
        }
    }
    // Locals drop in reverse declaration order on return:
    //   current_job → graphs → stream → bufs → module → ctx.
    // All GPU handles are released before the CUDA context that owns them,
    // and all of it happens on this worker thread (cuCtxDestroy requires
    // the destroying thread to have a current context).
    Ok(())
}

// ============================================================================
//   Main
// ============================================================================

fn main() -> std::process::ExitCode {
    let mining_address = match std::env::var("PEARLD_MINING_ADDRESS") {
        Ok(s) => s,
        Err(_) => {
            eprintln!("PEARLD_MINING_ADDRESS env var is required");
            return std::process::ExitCode::from(2);
        }
    };
    let fatbin_path = std::env::var("PEARL_FATBIN")
        .unwrap_or_else(|_| "/opt/fatbin/pearl_gemm.fatbin".to_string());
    let poll_secs: u64 = std::env::var("TEMPLATE_POLL_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let max_iters: u64 = std::env::var("MAX_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    if let Err(e) = run(&mining_address, &fatbin_path, poll_secs, max_iters) {
        eprintln!("FATAL: {e}");
        return std::process::ExitCode::from(1);
    }
    std::process::ExitCode::SUCCESS
}

fn run(
    mining_address: &str,
    fatbin_path: &str,
    poll_secs: u64,
    max_iters: u64,
) -> Result<(), MinerError> {
    let devs = pick_devices()?;
    println!("[live-miner] devices: {:?} ({} GPU(s))", devs, devs.len());

    let fatbin_bytes = Arc::new(std::fs::read(fatbin_path)?);
    println!(
        "[live-miner] fatbin: {fatbin_path} ({} bytes)",
        fatbin_bytes.len()
    );

    let client = Arc::new(PearldClient::new(PearldConfig::default()));
    println!("[live-miner] pearld: {}", PearldConfig::default().rpc_url);

    let cfg = pick_config();

    // ---- Shared template cell ----
    let template = Arc::new(SharedTemplate {
        latest: Mutex::new(None),
    });
    {
        let template = Arc::clone(&template);
        let client = Arc::clone(&client);
        std::thread::Builder::new()
            .name("template-poller".into())
            .spawn(move || {
                let mut last_prev = String::new();
                loop {
                    match client.get_block_template() {
                        Ok(t) => {
                            if t.previousblockhash != last_prev {
                                last_prev = t.previousblockhash.clone();
                                let mut guard = template.latest.lock().unwrap();
                                *guard = Some(Arc::new(t));
                            }
                        }
                        Err(e) => eprintln!("[tpl-thread] template fetch failed: {e}"),
                    }
                    std::thread::sleep(Duration::from_secs(poll_secs.max(1)));
                }
            })
            .expect("spawn template-poller");
    }

    // ---- Shared hit poller ----
    let (hit_tx, hit_rx) = mpsc::channel::<HitWork>();
    let hit_poller_handle = {
        let client = Arc::clone(&client);
        std::thread::Builder::new()
            .name("hit-poller".into())
            .spawn(move || hit_poller(hit_rx, client))
            .expect("spawn hit-poller")
    };

    // ---- Aggregate logger ----
    let total_iters = Arc::new(AtomicU64::new(0));
    let agg_running = Arc::new(std::sync::atomic::AtomicBool::new(true));
    {
        let total_iters = Arc::clone(&total_iters);
        let agg_running = Arc::clone(&agg_running);
        let n_dev = devs.len();
        std::thread::Builder::new()
            .name("agg-logger".into())
            .spawn(move || {
                let mut last = (Instant::now(), 0u64);
                while agg_running.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_secs(5));
                    let now = Instant::now();
                    let cur = total_iters.load(Ordering::Relaxed);
                    let dt = now.duration_since(last.0).as_secs_f64();
                    let rate = (cur - last.1) as f64 / dt;
                    println!(
                        "[live-miner] AGG iters={} aggregate_rate={:.1}/s ({:.1}/s per GPU avg, N={})",
                        cur, rate, rate / n_dev as f64, n_dev,
                    );
                    last = (now, cur);
                }
            })
            .expect("spawn agg-logger");
    }

    // ---- Per-process worker salt (mixed into per-job seed) ----
    let worker_salt = Arc::new(resolve_worker_salt());
    println!("[live-miner] worker_salt: {}", worker_salt);

    // ---- Workers (one OS thread per GPU) ----
    let mut handles = Vec::new();
    for &dev in &devs {
        let wctx = WorkerCtx {
            device_ord: dev,
            fatbin: Arc::clone(&fatbin_bytes),
            cfg,
            mining_address: mining_address.to_string(),
            template: Arc::clone(&template),
            hit_tx: hit_tx.clone(),
            total_iters: Arc::clone(&total_iters),
            max_iters,
            worker_salt: Arc::clone(&worker_salt),
        };
        let h = std::thread::Builder::new()
            .name(format!("worker-gpu{}", dev))
            .spawn(move || worker(wctx))
            .expect("spawn worker");
        handles.push((dev, h));
    }
    // Drop our own sender so the hit poller exits once all workers do.
    drop(hit_tx);

    let mut first_err: Option<MinerError> = None;
    for (dev, h) in handles {
        match h.join() {
            Ok(Ok(())) => println!("[live-miner] worker gpu{dev} finished"),
            Ok(Err(e)) => {
                eprintln!("[live-miner] worker gpu{dev} error: {e}");
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
            Err(_) => eprintln!("[live-miner] worker gpu{dev} panicked"),
        }
    }

    agg_running.store(false, Ordering::Relaxed);
    println!("[live-miner] draining hit poller (max 60s)…");
    let _ = hit_poller_handle.join();

    match first_err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
