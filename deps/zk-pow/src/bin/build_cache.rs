// Build verifier cache binary file
// Usage: cargo run --release --bin build_cache <output_file>
// Must be run before building the package with the embedded_cache feature.

use std::env;
use std::fs;
use std::path::PathBuf;
use zk_pow::circuit::circuit_utils::CircuitCache;
use zk_pow::circuit::pearl_circuit::{PearlRecursion, RecursionCircuit};

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run --release --bin build_cache <output_file>");
        eprintln!("Example: cargo run --release --bin build_cache src/circuit/cache.bin");
        std::process::exit(1);
    }

    let output_file = PathBuf::from(&args[1]);

    println!("Generating verifier cache...");
    let mut cache = CircuitCache::default();
    PearlRecursion::fill_verifier_cache(&mut cache);

    println!(
        "Cache generated: {} first circuits, {} second circuits",
        cache.verifier_circuits_1.len(),
        cache.verifier_circuits_2.len()
    );

    println!("Serializing to binary format...");
    let binary_data = cache.to_bytes().expect("Failed to serialize cache");

    println!("Writing {} bytes to {:?}", binary_data.len(), output_file);
    fs::write(&output_file, &binary_data).expect("Failed to write cache file");

    println!("\n✅ Cache binary generated successfully!");
    println!(
        "   Size: {} bytes ({:.2} KB)",
        binary_data.len(),
        binary_data.len() as f64 / 1024.0
    );
    println!("   Location: {:?}", output_file);
}
