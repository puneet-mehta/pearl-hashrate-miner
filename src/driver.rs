//! Thin safe wrappers over `cudarc::driver::sys`.
//!
//! The bits the high-level cudarc API doesn't cover for our use case
//! (`cuModuleLoadFatBinary`, raw `cuLaunchKernel` with arbitrary param packs).
//! Where cudarc's safe API is sufficient we use it directly.

use std::ffi::{c_void, CStr};
use std::ptr;

use cudarc::driver::sys as cu;
use cudarc::driver::sys::{CUcontext, CUdevice, CUdeviceptr, CUfunction, CUmodule};

use crate::error::{cu_check, MinerError};

/// Owns a CUDA primary context + device handle.
///
/// Created once at process start, threaded through every kernel launch.
pub struct CudaCtx {
    pub device: CUdevice,
    pub context: CUcontext,
}

/// Number of CUDA devices visible to the process (honors `CUDA_VISIBLE_DEVICES`).
/// Safe to call before any [`CudaCtx`] exists — calls `cuInit` itself.
pub fn device_count() -> Result<i32, MinerError> {
    unsafe {
        cu_check(cu::cuInit(0), "cuInit")?;
        let mut n: i32 = 0;
        cu_check(cu::cuDeviceGetCount(&mut n), "cuDeviceGetCount")?;
        Ok(n)
    }
}

impl CudaCtx {
    pub fn new(device_ord: i32) -> Result<Self, MinerError> {
        unsafe {
            cu_check(cu::cuInit(0), "cuInit")?;
            let mut device: CUdevice = 0;
            cu_check(cu::cuDeviceGet(&mut device, device_ord), "cuDeviceGet")?;
            let mut context: CUcontext = ptr::null_mut();
            cu_check(cu::cuCtxCreate_v2(&mut context, 0, device), "cuCtxCreate")?;
            Ok(CudaCtx { device, context })
        }
    }

    pub fn device_name(&self) -> Result<String, MinerError> {
        let mut buf = [0i8; 256];
        unsafe {
            cu_check(
                cu::cuDeviceGetName(buf.as_mut_ptr(), 256, self.device),
                "cuDeviceGetName",
            )?;
            Ok(CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned())
        }
    }

    pub fn synchronize(&self) -> Result<(), MinerError> {
        unsafe { cu_check(cu::cuCtxSynchronize(), "cuCtxSynchronize") }
    }

    /// `(major, minor)` compute capability — e.g. (8, 9) for an RTX 4090.
    /// Used to pick the matching Triton PTX blob.
    pub fn compute_capability(&self) -> Result<(i32, i32), MinerError> {
        let mut major: i32 = 0;
        let mut minor: i32 = 0;
        unsafe {
            cu_check(
                cu::cuDeviceGetAttribute(
                    &mut major,
                    cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    self.device,
                ),
                "cuDeviceGetAttribute(CC_MAJOR)",
            )?;
            cu_check(
                cu::cuDeviceGetAttribute(
                    &mut minor,
                    cu::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    self.device,
                ),
                "cuDeviceGetAttribute(CC_MINOR)",
            )?;
        }
        Ok((major, minor))
    }
}

impl Drop for CudaCtx {
    fn drop(&mut self) {
        unsafe {
            let _ = cu::cuCtxDestroy_v2(self.context);
        }
    }
}

/// A loaded CUDA module (cubin / fatbin).
pub struct Module {
    pub handle: CUmodule,
}

impl Module {
    /// Load a fatbin from an in-memory blob.
    pub fn load_fatbin(blob: &[u8]) -> Result<Self, MinerError> {
        let mut handle: CUmodule = ptr::null_mut();
        unsafe {
            cu_check(
                cu::cuModuleLoadFatBinary(&mut handle, blob.as_ptr() as *const c_void),
                "cuModuleLoadFatBinary",
            )?;
        }
        Ok(Module { handle })
    }

    /// Load a cubin OR a NULL-terminated PTX string via `cuModuleLoadData`.
    /// The driver auto-detects the format. For PTX, the driver JIT-
    /// compiles to whichever compute capability is present.
    pub fn load_data(data: &[u8]) -> Result<Self, MinerError> {
        let mut handle: CUmodule = ptr::null_mut();
        unsafe {
            cu_check(
                cu::cuModuleLoadData(&mut handle, data.as_ptr() as *const c_void),
                "cuModuleLoadData",
            )?;
        }
        Ok(Module { handle })
    }

    /// Get a kernel function handle by exact symbol name.
    pub fn get_function(&self, name: &str) -> Result<CUfunction, MinerError> {
        let cname = std::ffi::CString::new(name).expect("kernel name must not contain NUL");
        let mut fn_handle: CUfunction = ptr::null_mut();
        let r = unsafe { cu::cuModuleGetFunction(&mut fn_handle, self.handle, cname.as_ptr()) };
        if r == cu::CUresult::CUDA_ERROR_NOT_FOUND {
            return Err(MinerError::KernelNotFound { name: name.into() });
        }
        cu_check(r, "cuModuleGetFunction")?;
        Ok(fn_handle)
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            let _ = cu::cuModuleUnload(self.handle);
        }
    }
}

/// RAII wrapper for a single linear device allocation.
pub struct DevBuf {
    pub ptr: CUdeviceptr,
    pub size: usize,
}

impl DevBuf {
    pub fn alloc(size: usize) -> Result<Self, MinerError> {
        let mut ptr: CUdeviceptr = 0;
        unsafe {
            cu_check(cu::cuMemAlloc_v2(&mut ptr, size), "cuMemAlloc")?;
        }
        Ok(DevBuf { ptr, size })
    }

    pub fn copy_from(&mut self, src: &[u8]) -> Result<(), MinerError> {
        assert!(src.len() <= self.size);
        unsafe {
            cu_check(
                cu::cuMemcpyHtoD_v2(self.ptr, src.as_ptr() as *const c_void, src.len()),
                "cuMemcpyHtoD",
            )
        }
    }

    pub fn copy_to(&self, dst: &mut [u8]) -> Result<(), MinerError> {
        assert!(dst.len() <= self.size);
        unsafe {
            cu_check(
                cu::cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut c_void, self.ptr, dst.len()),
                "cuMemcpyDtoH",
            )
        }
    }

    /// Synchronously zero the entire allocation.
    pub fn zero(&self) -> Result<(), MinerError> {
        unsafe { cu_check(cu::cuMemsetD8_v2(self.ptr, 0, self.size), "cuMemsetD8") }
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        unsafe {
            let _ = cu::cuMemFree_v2(self.ptr);
        }
    }
}

/// Page-locked host buffer mapped into device address space.
///
/// Allocated via `cuMemHostAlloc(PORTABLE | DEVICEMAP | WRITECOMBINED)`.
/// Both `host_ptr` (for CPU reads/writes) and `device_ptr` (passed to kernels
/// via `cuMemHostGetDevicePointer`) refer to the same physical pages, so a
/// kernel write becomes visible to the host without an explicit D2H copy
/// (after a stream sync / event).
pub struct PinnedHostBuf {
    pub host_ptr: *mut c_void,
    pub device_ptr: CUdeviceptr,
    pub size: usize,
}

// `*mut c_void` is not Send/Sync by default, but we treat PinnedHostBuf as a
// passive resource container — adding manual impls so the struct can live in
// Arc<Mutex<...>>-style sharing if MinerBufs ever needs cross-thread access.
unsafe impl Send for PinnedHostBuf {}
unsafe impl Sync for PinnedHostBuf {}

impl PinnedHostBuf {
    /// Allocate `size` bytes of pinned host memory, mapped to device.
    pub fn alloc(size: usize) -> Result<Self, MinerError> {
        let flags: u32 = cu::CU_MEMHOSTALLOC_PORTABLE | cu::CU_MEMHOSTALLOC_DEVICEMAP;
        let mut host_ptr: *mut c_void = ptr::null_mut();
        unsafe {
            cu_check(
                cu::cuMemHostAlloc(&mut host_ptr, size, flags),
                "cuMemHostAlloc",
            )?;
            let mut device_ptr: CUdeviceptr = 0;
            cu_check(
                cu::cuMemHostGetDevicePointer_v2(&mut device_ptr, host_ptr, 0),
                "cuMemHostGetDevicePointer",
            )?;
            Ok(PinnedHostBuf {
                host_ptr,
                device_ptr,
                size,
            })
        }
    }

    /// CPU-side slice view. Safe because the pages are pinned + portable.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.host_ptr as *const u8, self.size) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.host_ptr as *mut u8, self.size) }
    }

    /// Zero the buffer via CPU writes (fast for small pinned regions like the
    /// 640-byte host-signal header).
    pub fn zero_cpu(&mut self) {
        self.as_mut_slice().fill(0);
    }
}

impl Drop for PinnedHostBuf {
    fn drop(&mut self) {
        unsafe {
            let _ = cu::cuMemFreeHost(self.host_ptr);
        }
    }
}

/// A non-default CUDA stream. CUDA graph capture is only valid on
/// non-default streams (cuStreamBeginCapture rejects the legacy default
/// stream).
pub struct Stream {
    pub handle: cu::CUstream,
}

impl Stream {
    /// Create a fresh non-blocking stream on the current context.
    /// Flag value `1` = `CU_STREAM_NON_BLOCKING` (the constant is feature-
    /// gated in cudarc; using the literal avoids the gate).
    pub fn new() -> Result<Self, MinerError> {
        let mut handle: cu::CUstream = ptr::null_mut();
        unsafe {
            cu_check(cu::cuStreamCreate(&mut handle, 1), "cuStreamCreate")?;
        }
        Ok(Self { handle })
    }

    pub fn synchronize(&self) -> Result<(), MinerError> {
        unsafe { cu_check(cu::cuStreamSynchronize(self.handle), "cuStreamSynchronize") }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            let _ = cu::cuStreamDestroy_v2(self.handle);
        }
    }
}

/// Instantiated CUDA graph (graph + executable handle).
///
/// Produced by recording a stream sequence between
/// `cuStreamBeginCapture` and `cuStreamEndCapture`, then
/// `cuGraphInstantiate`'ing the resulting `CUgraph`. Replay via
/// [`Self::launch`].
pub struct CapturedGraph {
    graph: cu::CUgraph,
    exec: cu::CUgraphExec,
    /// Optional handle to a per-replay mutable kernel node — used to vary
    /// `random_int8`'s `iter_idx` arg across replays without re-capturing.
    /// Populated by [`Self::record_mutable_random_int8`].
    mutable_random: Option<MutableRandomInt8>,
}

/// Per-replay mutable kernel node parameters. Captures everything needed to
/// rebuild a fresh `CUDA_KERNEL_NODE_PARAMS` and call
/// `cuGraphExecKernelNodeSetParams` before each `cuGraphLaunch`. Storage for
/// the four kernel arg values is inlined here so it has a stable address for
/// the duration of the SetParams call.
struct MutableRandomInt8 {
    node: cu::CUgraphNode,
    func: CUfunction,
    grid_x: u32,
    block_x: u32,
    /// Kernel arg slots. Order matches `RandomInt8::launch`:
    ///   [0] total_bytes (i32, lower 4 bytes of a u64 cell)
    ///   [1] seed_ptr     (CUdeviceptr)
    ///   [2] iter_idx     (u64) — the only field that varies per replay
    ///   [3] out_ptr      (CUdeviceptr)
    arg_total_bytes: i32,
    arg_seed: CUdeviceptr,
    arg_iter: u64,
    arg_out: CUdeviceptr,
}

impl CapturedGraph {
    /// Begin capture on `stream`. Callers run their stream-ordered ops
    /// between this call and [`Self::end`].
    ///
    /// # Safety
    /// Stream-ordered API only inside the capture region. Sync ops or
    /// default-stream operations will abort the capture.
    pub unsafe fn begin(stream: &Stream) -> Result<(), MinerError> {
        cu_check(
            cu::cuStreamBeginCapture_v2(
                stream.handle,
                cu::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            ),
            "cuStreamBeginCapture",
        )
    }

    /// End capture and instantiate.
    ///
    /// # Safety
    /// Must pair with a prior [`Self::begin`] on the same stream.
    pub unsafe fn end(stream: &Stream) -> Result<Self, MinerError> {
        let mut graph: cu::CUgraph = ptr::null_mut();
        cu_check(
            cu::cuStreamEndCapture(stream.handle, &mut graph),
            "cuStreamEndCapture",
        )?;
        let mut exec: cu::CUgraphExec = ptr::null_mut();
        // cudarc-cuda12080 exposes `cuGraphInstantiateWithFlags` (the
        // version-stable variant that doesn't require an error-node out-param).
        cu_check(
            cu::cuGraphInstantiateWithFlags(&mut exec, graph, 0),
            "cuGraphInstantiateWithFlags",
        )?;
        Ok(CapturedGraph {
            graph,
            exec,
            mutable_random: None,
        })
    }

    /// Launch the captured graph on `stream`.
    ///
    /// # Safety
    /// Stream must be on the same context as the capture.
    pub unsafe fn launch(&self, stream: &Stream) -> Result<(), MinerError> {
        cu_check(cu::cuGraphLaunch(self.exec, stream.handle), "cuGraphLaunch")
    }

    /// Locate the kernel node inside this captured graph whose function
    /// handle matches `func`, and remember it (along with the other kernel
    /// args we'll need to rebuild a `CUDA_KERNEL_NODE_PARAMS`). After this
    /// call, [`Self::launch_with_iter_idx`] will mutate the node's
    /// `iter_idx` arg per replay via `cuGraphExecKernelNodeSetParams`.
    ///
    /// Used to fold the `random_int8` launch INTO the captured graph (so
    /// it benefits from graph dispatch overhead reduction) without baking
    /// a constant `iter_idx` into the executable graph.
    ///
    /// Fails if zero or multiple matching kernel nodes are found.
    ///
    /// # Safety
    /// `func` must be the same `CUfunction` that was launched inside the
    /// capture region. `total_bytes`, `seed`, `out` must match the args
    /// passed at capture time (CUDA does not let you change `func` or
    /// dims after instantiate; the other args we pass in the SetParams
    /// will be applied).
    pub unsafe fn record_mutable_random_int8(
        &mut self,
        func: CUfunction,
        total_bytes: i32,
        seed: CUdeviceptr,
        out: CUdeviceptr,
        grid_x: u32,
        block_x: u32,
    ) -> Result<(), MinerError> {
        // Walk all nodes in the original (non-exec) graph.
        let mut num_nodes: usize = 0;
        cu_check(
            cu::cuGraphGetNodes(self.graph, ptr::null_mut(), &mut num_nodes),
            "cuGraphGetNodes(count)",
        )?;
        let mut nodes: Vec<cu::CUgraphNode> = vec![ptr::null_mut(); num_nodes];
        cu_check(
            cu::cuGraphGetNodes(self.graph, nodes.as_mut_ptr(), &mut num_nodes),
            "cuGraphGetNodes(fetch)",
        )?;

        let mut matches: Vec<cu::CUgraphNode> = Vec::new();
        for &node in &nodes {
            let mut ty: cu::CUgraphNodeType = cu::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
            cu_check(cu::cuGraphNodeGetType(node, &mut ty), "cuGraphNodeGetType")?;
            if ty != cu::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
                continue;
            }
            let mut params: cu::CUDA_KERNEL_NODE_PARAMS = std::mem::zeroed();
            cu_check(
                cu::cuGraphKernelNodeGetParams_v2(node, &mut params),
                "cuGraphKernelNodeGetParams_v2",
            )?;
            if params.func == func {
                matches.push(node);
            }
        }

        if matches.len() != 1 {
            return Err(MinerError::Other(format!(
                "expected exactly 1 random_int8 kernel node in captured \
                 graph, found {}",
                matches.len()
            )));
        }
        self.mutable_random = Some(MutableRandomInt8 {
            node: matches[0],
            func,
            grid_x,
            block_x,
            arg_total_bytes: total_bytes,
            arg_seed: seed,
            arg_iter: 0,
            arg_out: out,
        });
        Ok(())
    }

    /// Replay the captured graph after updating the random_int8 kernel
    /// node's `iter_idx` arg to `iter_idx`. Falls back to a plain replay
    /// if no mutable node was recorded.
    ///
    /// # Safety
    /// [`Self::record_mutable_random_int8`] must have been called first if
    /// you want the iter_idx mutation to take effect. The recorded kernel
    /// arg storage (`arg_seed`, `arg_out`) must still be valid device
    /// allocations.
    pub unsafe fn launch_with_iter_idx(
        &mut self,
        stream: &Stream,
        iter_idx: u64,
    ) -> Result<(), MinerError> {
        if let Some(m) = self.mutable_random.as_mut() {
            m.arg_iter = iter_idx;
            // kernelParams is an array of pointers to each arg's storage.
            // We point into &mut m fields — stable for the duration of the
            // SetParams call (the driver copies the values internally).
            let mut params: [*mut c_void; 4] = [
                &mut m.arg_total_bytes as *mut _ as *mut c_void,
                &mut m.arg_seed as *mut _ as *mut c_void,
                &mut m.arg_iter as *mut _ as *mut c_void,
                &mut m.arg_out as *mut _ as *mut c_void,
            ];
            let node_params = cu::CUDA_KERNEL_NODE_PARAMS {
                func: m.func,
                gridDimX: m.grid_x,
                gridDimY: 1,
                gridDimZ: 1,
                blockDimX: m.block_x,
                blockDimY: 1,
                blockDimZ: 1,
                sharedMemBytes: 0,
                kernelParams: params.as_mut_ptr(),
                extra: ptr::null_mut(),
                kern: ptr::null_mut(),
                ctx: ptr::null_mut(),
            };
            cu_check(
                cu::cuGraphExecKernelNodeSetParams_v2(self.exec, m.node, &node_params),
                "cuGraphExecKernelNodeSetParams_v2(random_int8.iter_idx)",
            )?;
        }
        cu_check(cu::cuGraphLaunch(self.exec, stream.handle), "cuGraphLaunch")
    }
}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        unsafe {
            let _ = cu::cuGraphExecDestroy(self.exec);
            let _ = cu::cuGraphDestroy(self.graph);
        }
    }
}

/// Launch a kernel with arbitrary parameters.
///
/// Each entry in `params` is a `*mut c_void` pointing at the value to pass.
/// The caller is responsible for keeping the pointed-at values alive for the
/// duration of the call (typically `let mut p = device_ptr;` then `&mut p as
/// *mut _ as *mut c_void`).
///
/// # Safety
///
/// `params` must contain valid pointers to the actual kernel argument values.
/// CUDA does not type-check parameters at launch time — mismatched layouts
/// silently produce wrong results.
pub unsafe fn launch_kernel(
    func: CUfunction,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem_bytes: u32,
    stream: cu::CUstream,
    params: &mut [*mut c_void],
) -> Result<(), MinerError> {
    cu_check(
        cu::cuLaunchKernel(
            func,
            grid.0,
            grid.1,
            grid.2,
            block.0,
            block.1,
            block.2,
            shared_mem_bytes,
            stream,
            params.as_mut_ptr(),
            ptr::null_mut(),
        ),
        "cuLaunchKernel",
    )
}
