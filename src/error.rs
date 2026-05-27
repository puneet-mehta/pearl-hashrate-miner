#[cfg(feature = "cuda")]
use std::ffi::CStr;
#[cfg(feature = "cuda")]
use std::ptr;

#[cfg(feature = "cuda")]
use cudarc::driver::sys as cu;
#[cfg(feature = "cuda")]
use cudarc::driver::sys::CUresult;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MinerError {
    #[error("CUDA driver error in {op}: {err}")]
    Cuda { op: &'static str, err: String },

    #[error("kernel `{name}` not found in fatbin")]
    KernelNotFound { name: String },

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("RPC `{method}`: {msg}")]
    Rpc { method: String, msg: String },

    #[error("{0}")]
    Other(String),
}

/// Convert a `CUresult` to a `Result`, attaching the operation name for context.
#[cfg(feature = "cuda")]
pub fn cu_check(r: CUresult, op: &'static str) -> Result<(), MinerError> {
    if r == CUresult::CUDA_SUCCESS {
        return Ok(());
    }
    let mut name: *const i8 = ptr::null();
    let err = unsafe {
        cu::cuGetErrorName(r, &mut name);
        if name.is_null() {
            format!("err={:?}", r)
        } else {
            CStr::from_ptr(name).to_string_lossy().into_owned()
        }
    };
    Err(MinerError::Cuda { op, err })
}
