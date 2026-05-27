//! Sync HTTP JSON-RPC client for pearld.
//!
//! Uses `ureq` (no tokio). Per-call timeout = 10 s; on transient failure,
//! callers should back off and retry (the polling thread handles this).

use std::time::Duration;

use crate::error::MinerError;
use crate::gateway::rpc_types::{BlockTemplateResponse, JsonRpcResponse};

/// Connection + auth config for the pearld RPC endpoint. Mirrors the
/// gateway's `PearlConfig` (env-driven defaults are in `bin/miner.rs`).
#[derive(Debug, Clone)]
pub struct PearldConfig {
    /// Full URL, e.g. `http://<pearld-host>:<port>`.
    pub rpc_url: String,
    pub rpc_user: String,
    pub rpc_password: String,
    /// HTTP timeout per request.
    pub timeout: Duration,
}

impl Default for PearldConfig {
    fn default() -> Self {
        Self {
            rpc_url: std::env::var("PEARLD_RPC_URL")
                .unwrap_or_else(|_| "http://0.0.0.0:44107".to_string()),
            rpc_user: std::env::var("PEARLD_RPC_USER").unwrap_or_else(|_| "user".to_string()),
            rpc_password: std::env::var("PEARLD_RPC_PASSWORD")
                .unwrap_or_else(|_| "pass".to_string()),
            timeout: Duration::from_secs(10),
        }
    }
}

pub struct PearldClient {
    config: PearldConfig,
    agent: ureq::Agent,
}

impl PearldClient {
    pub fn new(config: PearldConfig) -> Self {
        let agent = ureq::AgentBuilder::new().timeout(config.timeout).build();
        Self { config, agent }
    }

    /// Low-level JSON-RPC call. Inner error returned verbatim from pearld.
    fn rpc<T: serde::de::DeserializeOwned>(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<T, MinerError> {
        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1,
        });

        let basic = format!(
            "Basic {}",
            base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                format!("{}:{}", self.config.rpc_user, self.config.rpc_password),
            ),
        );

        let resp = self
            .agent
            .post(&self.config.rpc_url)
            .set("Authorization", &basic)
            .set("Content-Type", "application/json")
            .send_json(payload)
            .map_err(|e| MinerError::Rpc {
                method: method.to_string(),
                msg: format!("transport: {e}"),
            })?;

        let parsed: JsonRpcResponse<T> = resp.into_json().map_err(|e| MinerError::Rpc {
            method: method.to_string(),
            msg: format!("body decode: {e}"),
        })?;

        if let Some(err) = parsed.error {
            return Err(MinerError::Rpc {
                method: method.to_string(),
                msg: format!("pearld returned error {}: {}", err.code, err.message),
            });
        }
        parsed.result.ok_or_else(|| MinerError::Rpc {
            method: method.to_string(),
            msg: "pearld returned neither result nor error".to_string(),
        })
    }

    /// Fetch the latest block template. Use the standard segwit + coinbase
    /// capabilities the Python gateway sends so pearld returns the same
    /// shape of response.
    pub fn get_block_template(&self) -> Result<BlockTemplateResponse, MinerError> {
        let req = serde_json::json!([{
            "capabilities": ["coinbasevalue", "workid", "coinbase/append"],
            "rules": ["segwit"],
        }]);
        self.rpc("getblocktemplate", req)
    }

    /// Submit a hex-encoded block. pearld returns `null` on accept, or a
    /// reason string on reject. We have to parse the response manually
    /// (not via the generic `rpc()`) so we can distinguish "result: null"
    /// (which serde maps to None for Option<T>) from a missing field.
    pub fn submit_block(&self, block_hex: &str) -> Result<(), MinerError> {
        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "submitblock",
            "params": [block_hex],
            "id": 1,
        });
        let basic = format!(
            "Basic {}",
            base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                format!("{}:{}", self.config.rpc_user, self.config.rpc_password),
            ),
        );
        let resp = self
            .agent
            .post(&self.config.rpc_url)
            .set("Authorization", &basic)
            .set("Content-Type", "application/json")
            .send_json(payload)
            .map_err(|e| MinerError::Rpc {
                method: "submitblock".to_string(),
                msg: format!("transport: {e}"),
            })?;
        let v: serde_json::Value = resp.into_json().map_err(|e| MinerError::Rpc {
            method: "submitblock".to_string(),
            msg: format!("body decode: {e}"),
        })?;
        // pearld returns `result: null` on accept. Treat null OR absent as success.
        if let Some(err) = v.get("error") {
            if !err.is_null() {
                return Err(MinerError::Rpc {
                    method: "submitblock".to_string(),
                    msg: format!("pearld error: {}", err),
                });
            }
        }
        if let Some(r) = v.get("result") {
            if r.is_null() {
                return Ok(());
            }
            return Err(MinerError::Rpc {
                method: "submitblock".to_string(),
                msg: format!("rejected: {}", r),
            });
        }
        Ok(())
    }
}
