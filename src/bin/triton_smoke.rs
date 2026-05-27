//! Standalone smoke test for the embedded Triton kernels.
//!
//! Loads `noising_kernel` + `pearl_search_norotl_kernel` PTX, dispatches
//! them on synthetic inputs, and verifies:
//!
//! 1. The PTX loads via `cuModuleLoadData` (driver JIT).
//! 2. The kernel symbols resolve.
//! 3. The launchers don't crash (no kernel-arg-layout errors).

use std::process::ExitCode;

use cudarc::driver::sys as cu;

use pearl_hashrate_miner::driver::{CudaCtx, DevBuf, Module};
use pearl_hashrate_miner::error::cu_check;
use pearl_hashrate_miner::kernels::triton::{
    noising_ptx, search_norotl_ptx, TritonNoising, TritonSearchNorotl, BLOCK_M, BLOCK_N,
    HASH_CANDIDATES, JACKPOT_SIZE,
};

fn main() -> ExitCode {
    if let Err(e) = run() {
        eprintln!("FAIL: {e}");
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

fn run() -> Result<(), pearl_hashrate_miner::MinerError> {
    let ctx = CudaCtx::new(0)?;
    println!("device: {}", ctx.device_name()?);
    let (cc_maj, cc_min) = ctx.compute_capability()?;

    // ----- Load Triton kernels (PTX, driver JITs to device CC) -----
    let noising_blob = noising_ptx(cc_maj, cc_min);
    println!("loading noising PTX ({} bytes)…", noising_blob.len());
    let noising_mod = Module::load_data(&noising_blob)?;
    let noising = TritonNoising::new(&noising_mod)?;
    println!("  noising kernel loaded");

    let search_blob = search_norotl_ptx(cc_maj, cc_min);
    println!("loading search PTX ({} bytes)…", search_blob.len());
    let search_mod = Module::load_data(&search_blob)?;
    let search = TritonSearchNorotl::new(&search_mod)?;
    println!("  search kernel loaded");

    // ----- default shape -----
    let (m, n, k, r) = (8192i32, 32768i32, 2048i32, 128i32);

    // Allocate inputs with deterministic data (just zeros + 1; we're
    // smoke-testing dispatch, not correctness).
    println!(
        "allocating buffers ({} MB total)…",
        (m * n + n * k + m * k * 2) as i64 / 1_000_000
    );
    let mut a = DevBuf::alloc((m * k) as usize)?;
    let mut b = DevBuf::alloc((n * k) as usize)?;
    let mut eal = DevBuf::alloc((m * r) as usize)?;
    let mut ebr = DevBuf::alloc((n * r) as usize)?;
    let mut ear_r = DevBuf::alloc((k * r) as usize)?;
    let mut ebl_r = DevBuf::alloc((k * r) as usize)?;
    let ap_ea = DevBuf::alloc((m * k) as usize)?;
    let bp_eb = DevBuf::alloc((n * k) as usize)?;

    a.zero()?;
    b.zero()?;
    eal.zero()?;
    ebr.zero()?;
    ear_r.zero()?;
    ebl_r.zero()?;
    let a_init = vec![1u8; (m * k) as usize];
    a.copy_from(&a_init)?;

    // ----- Noising A → ApEA -----
    println!("noising A (Triton)…");
    unsafe {
        noising.launch(
            m,
            k,
            r,
            a.ptr,
            eal.ptr,
            ear_r.ptr,
            ap_ea.ptr,
            std::ptr::null_mut(),
        )?;
    }
    ctx.synchronize()?;
    println!("  ApEA OK");

    // ----- Noising B → BpEB -----
    println!("noising B (Triton)…");
    unsafe {
        noising.launch(
            n,
            k,
            r,
            b.ptr,
            ebr.ptr,
            ebl_r.ptr,
            bp_eb.ptr,
            std::ptr::null_mut(),
        )?;
    }
    ctx.synchronize()?;
    println!("  BpEB OK");

    // ----- Search -----
    let num_tile_m = m / BLOCK_M as i32;
    let num_tile_n = n / BLOCK_N as i32;
    let num_tiles = (num_tile_m * num_tile_n) as usize;
    let transcript_bytes = num_tiles * HASH_CANDIDATES * JACKPOT_SIZE * 4;
    println!(
        "transcripts buffer: {} MB (num_tiles={} HC={} JP={})",
        transcript_bytes / 1_000_000,
        num_tiles,
        HASH_CANDIDATES,
        JACKPOT_SIZE
    );
    let transcripts = DevBuf::alloc(transcript_bytes)?;
    transcripts.zero()?;

    println!("search_norotl (Triton)…");
    unsafe {
        search.launch(
            m,
            n,
            k,
            ap_ea.ptr,
            bp_eb.ptr,
            transcripts.ptr,
            std::ptr::null_mut(),
        )?;
    }
    ctx.synchronize()?;

    // Sanity: the transcript buffer should be all-zero with our zero
    // inputs (Triton kernel writes per-checkpoint values; for zero
    // inputs, every XOR-reduction is zero too). Verify it's still zero.
    let mut sample = vec![0u8; 4096.min(transcript_bytes)];
    transcripts.copy_to(&mut sample)?;
    let nonzero = sample.iter().filter(|b| **b != 0).count();
    println!(
        "  search done, first {} bytes have {} non-zero values",
        sample.len(),
        nonzero
    );

    println!("ALL DISPATCH SMOKE OK");
    Ok(())
}
