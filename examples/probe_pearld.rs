//! Standalone probe: fetches one block template from pearld and dumps it.
//! Doesn't need CUDA — useful for testing the JSON-RPC wire format from
//! any host with network access to the pearld endpoint.
//!
//! Usage:
//!   PEARLD_RPC_URL=http://<pearld-host>:<port> \
//!     cargo run --example probe_pearld

use pearl_hashrate_miner::gateway::{PearldClient, PearldConfig};

fn main() {
    let cfg = PearldConfig::default();
    println!("connecting to pearld at {}", cfg.rpc_url);

    let client = PearldClient::new(cfg);
    let template = match client.get_block_template() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("getblocktemplate failed: {e}");
            std::process::exit(1);
        }
    };

    println!("BlockTemplate:");
    println!("  height            {}", template.height);
    println!("  version           {}", template.version);
    println!("  previousblockhash {}", template.previousblockhash);
    println!("  bits              {}", template.bits);
    println!("  target            {}", template.target);
    println!("  curtime           {}", template.curtime);
    println!("  mintime           {}", template.mintime);
    println!("  maxtime           {}", template.maxtime);
    println!("  coinbasevalue     {}", template.coinbasevalue);
    println!("  vsizelimit        {}", template.vsizelimit);
    println!(
        "  transactions      {} entries",
        template.transactions.len()
    );
    println!("  capabilities      {:?}", template.capabilities);
    println!("  mutable           {:?}", template.mutable);
    println!("  noncerange        {}", template.noncerange);
    println!("  longpollid        {}", template.longpollid);
    println!("  coinbaseaux.flags {:?}", template.coinbaseaux.flags);
    println!(
        "  default_witness_commitment {:?}",
        template.default_witness_commitment
    );
}
