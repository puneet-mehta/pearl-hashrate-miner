//! Serde types matching `pearld`'s `getblocktemplate` response shape.
//! Mirrors `miner/pearl-gateway/src/pearl_gateway/rpc_types.py`.

use serde::{Deserialize, Serialize};

/// Regular (non-coinbase) transaction in the `getblocktemplate` response.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BlockTemplateTx {
    pub data: String,
    pub hash: String,
    pub txid: String,
    pub depends: Vec<u64>,
    pub fee: i64,
    pub vsize: u64,
}

/// Coinbase auxiliary blob (free-form bytes the miner stamps into the
/// coinbase scriptSig — used for tagging like "/p2pool/" etc).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct CoinbaseAux {
    #[serde(default)]
    pub flags: String,
}

/// Full `getblocktemplate` response. Strict-ish — extra fields are ignored
/// to be forwards-compatible with pearld version bumps.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BlockTemplateResponse {
    pub bits: String,
    pub curtime: u64,
    pub height: u64,
    pub previousblockhash: String,
    pub vsizelimit: u64,
    #[serde(default)]
    pub transactions: Vec<BlockTemplateTx>,
    pub version: i32,
    #[serde(default)]
    pub longpollid: String,
    pub target: String,
    #[serde(default)]
    pub maxtime: u64,
    #[serde(default)]
    pub mintime: u64,
    #[serde(default)]
    pub mutable: Vec<String>,
    #[serde(default)]
    pub noncerange: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub coinbaseaux: CoinbaseAux,
    pub coinbasevalue: u64,
    #[serde(default)]
    pub default_witness_commitment: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(default)]
    pub data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcResponse<T> {
    #[allow(dead_code)]
    pub jsonrpc: Option<String>,
    pub result: Option<T>,
    pub error: Option<JsonRpcError>,
    #[allow(dead_code)]
    pub id: Option<serde_json::Value>,
}
