//! Compatibility layer for the historical `ocr-rs` MNN API.
//!
//! Native ownership, linking and inference are delegated to `mnn-runtime`.
//! The types in this module intentionally keep their existing names, fields,
//! variants and method signatures so downstream `ocr-rs` users do not need to
//! migrate with the native implementation.

mod gpu;
pub use gpu::GpuTuningMode;

use ndarray::{ArrayD, ArrayViewD, IxDyn};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Condvar, Mutex},
};

/// MNN-related errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MnnError {
    /// Invalid parameter.
    InvalidParameter(String),
    /// Out of memory.
    OutOfMemory,
    /// Runtime error.
    RuntimeError(String),
    /// Unsupported operation.
    Unsupported,
    /// Model loading failed.
    ModelLoadFailed(String),
    /// Requested inference backend is not available in the linked MNN build.
    BackendUnavailable(String),
    /// Null pointer error.
    NullPointer,
    /// Shape mismatch.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
}

impl std::fmt::Display for MnnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidParameter(message) => write!(f, "Invalid parameter: {message}"),
            Self::OutOfMemory => write!(f, "Out of memory"),
            Self::RuntimeError(message) => write!(f, "Runtime error: {message}"),
            Self::Unsupported => write!(f, "Unsupported operation"),
            Self::ModelLoadFailed(message) => write!(f, "Model loading failed: {message}"),
            Self::BackendUnavailable(backend) => write!(f, "Backend unavailable: {backend}"),
            Self::NullPointer => write!(f, "Null pointer"),
            Self::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {expected:?}, got {got:?}")
            }
        }
    }
}

impl std::error::Error for MnnError {}

/// Result type used by the compatibility API.
pub type Result<T> = std::result::Result<T, MnnError>;

/// Precision mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum PrecisionMode {
    /// Normal precision.
    #[default]
    Normal = 0,
    /// Low precision (faster).
    Low = 1,
    /// High precision (more accurate).
    High = 2,
}

/// Data format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum DataFormat {
    /// NCHW format (Caffe/PyTorch/ONNX).
    #[default]
    NCHW = 0,
    /// NHWC format (TensorFlow).
    NHWC = 1,
    /// Auto detect.
    Auto = 2,
}

/// OpenCL memory representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum GpuMemoryMode {
    /// Let MNN select the OpenCL memory representation.
    #[default]
    Auto = 0,
    /// Store OpenCL tensors in buffers, avoiding 2D image dimension limits.
    Buffer = 1 << 6,
    /// Store OpenCL tensors in images.
    Image = 1 << 7,
}

/// Inference backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// CPU backend.
    #[default]
    CPU,
    /// Metal GPU (macOS/iOS).
    Metal,
    /// OpenCL GPU.
    OpenCL,
    /// OpenGL GPU.
    OpenGL,
    /// Vulkan GPU.
    Vulkan,
    /// CUDA GPU (NVIDIA).
    CUDA,
    /// CoreML (macOS/iOS).
    CoreML,
}

impl Backend {
    /// Stable backend name used in errors, logs, and command-line options.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CPU => "cpu",
            Self::Metal => "metal",
            Self::OpenCL => "opencl",
            Self::OpenGL => "opengl",
            Self::Vulkan => "vulkan",
            Self::CUDA => "cuda",
            Self::CoreML => "coreml",
        }
    }

    /// Return whether this backend was compiled and registered by linked MNN.
    pub fn is_available(self) -> bool {
        mnn_runtime::Runtime::new(mnn_runtime::RuntimeConfig::new().with_backend(self.to_runtime()))
            .is_ok()
    }

    const fn to_runtime(self) -> mnn_runtime::Backend {
        match self {
            Self::CPU => mnn_runtime::Backend::Cpu,
            Self::Metal => mnn_runtime::Backend::Metal,
            Self::OpenCL => mnn_runtime::Backend::OpenCl,
            Self::OpenGL => mnn_runtime::Backend::OpenGl,
            Self::Vulkan => mnn_runtime::Backend::Vulkan,
            Self::CUDA => mnn_runtime::Backend::Cuda,
            Self::CoreML => mnn_runtime::Backend::CoreMl,
        }
    }
}

/// Inference configuration.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Thread count (0 means auto, default is 4).
    pub thread_count: i32,
    /// Precision mode.
    pub precision_mode: PrecisionMode,
    /// Whether to use cache.
    pub use_cache: bool,
    /// Data format.
    pub data_format: DataFormat,
    /// Inference backend.
    pub backend: Backend,
    /// OpenCL tensor memory representation; ignored by other backends.
    pub gpu_memory_mode: GpuMemoryMode,
    /// GPU tuning effort.
    pub gpu_tuning_mode: GpuTuningMode,
    /// Optional directory for model/config-specific GPU kernel caches.
    pub gpu_cache_dir: Option<PathBuf>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            thread_count: 4,
            precision_mode: PrecisionMode::Normal,
            use_cache: false,
            data_format: DataFormat::NCHW,
            backend: Backend::CPU,
            gpu_memory_mode: GpuMemoryMode::Auto,
            gpu_tuning_mode: GpuTuningMode::Auto,
            gpu_cache_dir: None,
        }
    }
}

impl InferenceConfig {
    /// Create a new inference configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set thread count.
    pub fn with_threads(mut self, threads: i32) -> Self {
        self.thread_count = threads;
        self
    }

    /// Set precision mode.
    pub fn with_precision(mut self, precision: PrecisionMode) -> Self {
        self.precision_mode = precision;
        self
    }

    /// Set backend.
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Set the OpenCL tensor memory representation.
    pub fn with_gpu_memory_mode(mut self, mode: GpuMemoryMode) -> Self {
        self.gpu_memory_mode = mode;
        self
    }

    /// Set GPU kernel tuning effort.
    pub fn with_gpu_tuning(mut self, mode: GpuTuningMode) -> Self {
        self.gpu_tuning_mode = mode;
        self
    }

    /// Enable a persistent GPU cache isolated by model and configuration.
    pub fn with_gpu_cache_dir(mut self, directory: impl Into<PathBuf>) -> Self {
        self.gpu_cache_dir = Some(directory.into());
        self.use_cache = true;
        self
    }

    /// Set data format.
    pub fn with_data_format(mut self, format: DataFormat) -> Self {
        self.data_format = format;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.backend == Backend::Vulkan && self.gpu_tuning_mode.bits(true).is_none() {
            return Err(MnnError::InvalidParameter(
                "Fast/Normal tuning are OpenCL-only; use Auto/None/Wide/Heavy for Vulkan"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    fn to_runtime(&self) -> Result<mnn_runtime::RuntimeConfig> {
        self.validate()?;
        let threads = if self.thread_count <= 0 {
            4
        } else {
            usize::try_from(self.thread_count).map_err(|_| {
                MnnError::InvalidParameter("thread count does not fit in usize".to_owned())
            })?
        };
        let precision = match self.precision_mode {
            PrecisionMode::Normal => mnn_runtime::PrecisionMode::Normal,
            PrecisionMode::Low => mnn_runtime::PrecisionMode::Low,
            PrecisionMode::High => mnn_runtime::PrecisionMode::High,
        };
        let tuning = match self.gpu_tuning_mode {
            GpuTuningMode::Auto => mnn_runtime::GpuTuning::Auto,
            GpuTuningMode::None => mnn_runtime::GpuTuning::None,
            GpuTuningMode::Fast => mnn_runtime::GpuTuning::Fast,
            GpuTuningMode::Normal => mnn_runtime::GpuTuning::Normal,
            GpuTuningMode::Wide => mnn_runtime::GpuTuning::Wide,
            GpuTuningMode::Heavy => mnn_runtime::GpuTuning::Heavy,
        };
        let gpu_memory = match self.gpu_memory_mode {
            GpuMemoryMode::Auto => mnn_runtime::GpuMemoryMode::Auto,
            GpuMemoryMode::Buffer => mnn_runtime::GpuMemoryMode::Buffer,
            GpuMemoryMode::Image => mnn_runtime::GpuMemoryMode::Image,
        };

        let mut config = mnn_runtime::RuntimeConfig::new()
            .with_backend(self.backend.to_runtime())
            .with_threads(threads)
            .with_precision(precision)
            .with_gpu_tuning(tuning)
            .with_gpu_memory(gpu_memory);
        if let Some(directory) = &self.gpu_cache_dir {
            config = config.with_gpu_cache_dir(directory);
        }
        Ok(config)
    }

    fn runtime(&self) -> Result<mnn_runtime::Runtime> {
        let config = self.to_runtime()?;
        mnn_runtime::Runtime::new(config).map_err(convert_runtime_error)
    }
}

/// Shared runtime for sharing execution policy and MNN resources among engines.
pub struct SharedRuntime {
    runtime: mnn_runtime::Runtime,
}

impl SharedRuntime {
    /// Create a new shared runtime.
    pub fn new(config: &InferenceConfig) -> Result<Self> {
        Ok(Self {
            runtime: config.runtime()?,
        })
    }
}

/// MNN inference engine.
pub struct InferenceEngine {
    model: mnn_runtime::Model,
    model_bytes: Arc<[u8]>,
    input_name: String,
    output_name: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl InferenceEngine {
    /// Create an inference engine from model byte data.
    pub fn from_buffer(model_buffer: &[u8], config: Option<InferenceConfig>) -> Result<Self> {
        if model_buffer.is_empty() {
            return Err(MnnError::InvalidParameter("Model data is empty".to_owned()));
        }
        let config = config.unwrap_or_default();
        let runtime = SharedRuntime::new(&config)?;
        Self::load(model_buffer, &runtime)
    }

    /// Save compiled GPU kernels/tuning results to the configured cache.
    pub fn save_cache(&self) -> Result<()> {
        self.model.save_cache().map_err(convert_runtime_error)
    }

    /// Create an inference engine from a model file.
    pub fn from_file(
        model_path: impl AsRef<Path>,
        config: Option<InferenceConfig>,
    ) -> Result<Self> {
        let model_buffer = std::fs::read(model_path.as_ref()).map_err(|error| {
            MnnError::ModelLoadFailed(format!("Failed to read model file: {error}"))
        })?;
        Self::from_buffer(&model_buffer, config)
    }

    /// Create an inference engine from model bytes using a shared runtime.
    pub fn from_buffer_with_runtime(model_buffer: &[u8], runtime: &SharedRuntime) -> Result<Self> {
        if model_buffer.is_empty() {
            return Err(MnnError::InvalidParameter("Model data is empty".to_owned()));
        }
        Self::load(model_buffer, runtime)
    }

    fn load(model_buffer: &[u8], runtime: &SharedRuntime) -> Result<Self> {
        let model = runtime
            .runtime
            .load_bytes(model_buffer.to_vec())
            .map_err(convert_model_load_error)?;
        let info = model.info();
        if info.inputs().len() != 1 || info.outputs().len() != 1 {
            return Err(MnnError::Unsupported);
        }
        let input = &info.inputs()[0];
        let output = &info.outputs()[0];
        let input_name = input.name().to_owned();
        let output_name = output.name().to_owned();
        let input_shape = legacy_shape(input.shape());
        let output_shape = legacy_shape(output.shape());

        Ok(Self {
            model,
            model_bytes: Arc::from(model_buffer),
            input_name,
            output_name,
            input_shape,
            output_shape,
        })
    }

    /// Get input tensor shape.
    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    /// Get output tensor shape.
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// Execute fixed-shape inference.
    pub fn run(&self, input_data: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
        if input_data.shape() != self.input_shape.as_slice() {
            return Err(MnnError::ShapeMismatch {
                expected: self.input_shape.clone(),
                got: input_data.shape().to_vec(),
            });
        }
        let values = contiguous(&input_data)?;
        let output = run_model(
            &self.model,
            &self.input_name,
            values,
            input_data.shape(),
            false,
        )?;
        tensor_to_array(output)
    }

    /// Execute fixed-shape inference using raw slices.
    pub fn run_raw(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let expected_input = checked_product(&self.input_shape).unwrap_or(usize::MAX);
        let expected_output = checked_product(&self.output_shape).unwrap_or(usize::MAX);
        if input.len() != expected_input {
            return Err(MnnError::ShapeMismatch {
                expected: vec![expected_input],
                got: vec![input.len()],
            });
        }
        if output.len() != expected_output {
            return Err(MnnError::ShapeMismatch {
                expected: vec![expected_output],
                got: vec![output.len()],
            });
        }
        let result = run_model(
            &self.model,
            &self.input_name,
            input,
            &self.input_shape,
            false,
        )?;
        if result.name() != self.output_name || result.data().len() != output.len() {
            return Err(MnnError::InvalidParameter(
                "output length mismatch".to_owned(),
            ));
        }
        output.copy_from_slice(result.data());
        Ok(())
    }

    /// Check whether the graph has unresolved input or output dimensions.
    pub fn has_dynamic_shape(&self) -> bool {
        self.model.info().inputs()[0].is_dynamic() || self.model.info().outputs()[0].is_dynamic()
    }

    /// Execute dynamic-shape inference.
    pub fn run_dynamic(&self, input_data: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
        let values = contiguous(&input_data)?;
        let output = run_model(
            &self.model,
            &self.input_name,
            values,
            input_data.shape(),
            true,
        )?;
        tensor_to_array(output)
    }

    /// Execute dynamic-shape inference using raw slices.
    pub fn run_dynamic_raw(
        &self,
        input: &[f32],
        input_shape: &[usize],
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        validate_dynamic_input(input.len(), input_shape)?;
        let output = run_model(&self.model, &self.input_name, input, input_shape, true)?;
        let shape = output.shape().to_vec();
        Ok((output.into_data(), shape))
    }
}

/// Session pool for high-concurrency inference scenarios.
pub struct SessionPool {
    model: mnn_runtime::Model,
    input_name: String,
    output_name: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    permits: PoolPermits,
}

impl SessionPool {
    /// Create a session pool.
    pub fn new(
        engine: &InferenceEngine,
        pool_size: usize,
        config: Option<InferenceConfig>,
    ) -> Result<Self> {
        if pool_size == 0 {
            return Err(MnnError::InvalidParameter(
                "Pool size cannot be 0".to_owned(),
            ));
        }

        // Historically the pool's optional configuration selected the sessions
        // independently from the engine's default session. Reloading one worker
        // preserves that observable configuration while permits preserve the
        // old admission/available contract. MNN execution remains serialized,
        // as it was by the old bridge's process-wide mutex.
        let config = config.unwrap_or_default();
        let runtime = config.runtime()?;
        let model = runtime
            .load_bytes(engine.model_bytes.as_ref().to_vec())
            .map_err(convert_model_load_error)?;
        if model.info().inputs().len() != 1 || model.info().outputs().len() != 1 {
            return Err(MnnError::Unsupported);
        }

        Ok(Self {
            input_name: model.info().inputs()[0].name().to_owned(),
            output_name: model.info().outputs()[0].name().to_owned(),
            model,
            input_shape: engine.input_shape.clone(),
            output_shape: engine.output_shape.clone(),
            permits: PoolPermits::new(pool_size),
        })
    }

    /// Execute fixed-shape inference, waiting for an available pool slot.
    pub fn run(&self, input_data: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
        if input_data.shape() != self.input_shape.as_slice() {
            return Err(MnnError::ShapeMismatch {
                expected: self.input_shape.clone(),
                got: input_data.shape().to_vec(),
            });
        }
        let values = contiguous(&input_data)?;
        let _permit = self.permits.acquire();
        let output = run_model(
            &self.model,
            &self.input_name,
            values,
            input_data.shape(),
            false,
        )?;
        if output.name() != self.output_name || output.shape() != self.output_shape {
            return Err(MnnError::RuntimeError(
                "Session pool inference produced unexpected output metadata".to_owned(),
            ));
        }
        tensor_to_array(output)
    }

    /// Get the number of currently available pool slots.
    pub fn available(&self) -> usize {
        self.permits.available()
    }
}

struct PoolPermits {
    available: Mutex<usize>,
    ready: Condvar,
}

impl PoolPermits {
    fn new(size: usize) -> Self {
        Self {
            available: Mutex::new(size),
            ready: Condvar::new(),
        }
    }

    fn available(&self) -> usize {
        *self
            .available
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn acquire(&self) -> PoolPermit<'_> {
        let mut available = self
            .available
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while *available == 0 {
            available = self
                .ready
                .wait(available)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        *available -= 1;
        PoolPermit { pool: self }
    }
}

struct PoolPermit<'a> {
    pool: &'a PoolPermits,
}

impl Drop for PoolPermit<'_> {
    fn drop(&mut self) {
        let mut available = self
            .pool
            .available
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *available += 1;
        self.pool.ready.notify_one();
    }
}

fn run_model(
    model: &mnn_runtime::Model,
    input_name: &str,
    values: &[f32],
    shape: &[usize],
    dynamic: bool,
) -> Result<mnn_runtime::Tensor> {
    let input = mnn_runtime::Tensor::new(input_name, shape.to_vec(), values.to_vec())
        .map_err(convert_runtime_error)?;
    let mut outputs = if dynamic {
        model.run_dynamic_owned(vec![input])
    } else {
        model.run_owned(vec![input])
    }
    .map_err(convert_runtime_error)?;
    if outputs.len() != 1 {
        return Err(MnnError::Unsupported);
    }
    Ok(outputs.remove(0))
}

fn tensor_to_array(output: mnn_runtime::Tensor) -> Result<ArrayD<f32>> {
    let shape = output.shape().to_vec();
    ArrayD::from_shape_vec(IxDyn(&shape), output.into_data())
        .map_err(|error| MnnError::RuntimeError(format!("Failed to create output array: {error}")))
}

fn contiguous<'view, 'data>(input: &'view ArrayViewD<'data, f32>) -> Result<&'view [f32]> {
    input
        .as_slice()
        .ok_or_else(|| MnnError::InvalidParameter("Input data must be contiguous".to_owned()))
}

fn legacy_shape(shape: &[i32]) -> Vec<usize> {
    shape.iter().map(|&dimension| dimension as usize).collect()
}

fn checked_product(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1_usize, |size, &dimension| size.checked_mul(dimension))
}

fn validate_dynamic_input(length: usize, shape: &[usize]) -> Result<()> {
    if shape.is_empty()
        || shape.len() > 8
        || shape
            .iter()
            .any(|&dimension| dimension == 0 || dimension > i32::MAX as usize)
    {
        return Err(MnnError::InvalidParameter(
            "dynamic input needs 1..=8 positive i32 dimensions".to_owned(),
        ));
    }
    if checked_product(shape) != Some(length)
        || length > i32::MAX as usize / std::mem::size_of::<f32>()
    {
        return Err(MnnError::InvalidParameter(
            "dynamic input shape does not match buffer length or exceeds native limits".to_owned(),
        ));
    }
    Ok(())
}

fn convert_model_load_error(error: mnn_runtime::Error) -> MnnError {
    match error {
        mnn_runtime::Error::BackendNotCompiled { backend, .. }
        | mnn_runtime::Error::BackendUnavailable(backend) => {
            MnnError::BackendUnavailable(backend.to_owned())
        }
        mnn_runtime::Error::InvalidConfig(message) => MnnError::InvalidParameter(message),
        error => match native_status(&error) {
            Some(2) => MnnError::OutOfMemory,
            Some(4) => MnnError::Unsupported,
            _ => MnnError::ModelLoadFailed(error.to_string()),
        },
    }
}

fn convert_runtime_error(error: mnn_runtime::Error) -> MnnError {
    use mnn_runtime::Error;
    match error {
        Error::BackendNotCompiled { backend, .. } | Error::BackendUnavailable(backend) => {
            MnnError::BackendUnavailable(backend.to_owned())
        }
        Error::InvalidConfig(message) => MnnError::InvalidParameter(message),
        Error::ShapeMismatch {
            expected, actual, ..
        } => MnnError::ShapeMismatch {
            expected: legacy_shape(&expected),
            got: actual,
        },
        Error::DataLengthMismatch {
            expected, actual, ..
        } => MnnError::ShapeMismatch {
            expected: vec![expected],
            got: vec![actual],
        },
        Error::InvalidShape { .. }
        | Error::ShapeOverflow { .. }
        | Error::UnresolvedShape { .. }
        | Error::DuplicateTensor { .. }
        | Error::MissingInput(_)
        | Error::UnknownTensor { .. } => MnnError::InvalidParameter(error.to_string()),
        error => match native_status(&error) {
            Some(1) => MnnError::InvalidParameter(error.to_string()),
            Some(2) => MnnError::OutOfMemory,
            Some(4) => MnnError::Unsupported,
            Some(5) => MnnError::ModelLoadFailed(error.to_string()),
            _ => MnnError::RuntimeError(error.to_string()),
        },
    }
}

// mnn-runtime 0.1 keeps the sys status in NativeError's Display text. Parse
// only the stable numeric suffix until the safe error exposes it structurally.
fn native_status(error: &mnn_runtime::Error) -> Option<i32> {
    let message = match error {
        mnn_runtime::Error::Native { message, .. } => message,
        _ => return None,
    };
    let suffix = message.strip_suffix(')')?;
    suffix.rsplit_once("(status ")?.1.parse().ok()
}

/// Get the linked MNN version number.
pub fn get_version() -> String {
    mnn_runtime::Runtime::native_version()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_and_builders_remain_compatible() {
        let default = InferenceConfig::default();
        assert_eq!(default.thread_count, 4);
        assert_eq!(default.precision_mode, PrecisionMode::Normal);
        assert_eq!(default.data_format, DataFormat::NCHW);
        assert_eq!(default.gpu_memory_mode, GpuMemoryMode::Auto);

        let config = InferenceConfig::new()
            .with_threads(8)
            .with_precision(PrecisionMode::High)
            .with_backend(Backend::OpenCL)
            .with_gpu_memory_mode(GpuMemoryMode::Buffer)
            .with_data_format(DataFormat::Auto);
        assert_eq!(config.thread_count, 8);
        assert_eq!(config.precision_mode, PrecisionMode::High);
        assert_eq!(config.backend, Backend::OpenCL);
        assert_eq!(config.gpu_memory_mode, GpuMemoryMode::Buffer);
        assert_eq!(config.data_format, DataFormat::Auto);
    }

    #[test]
    fn tuning_and_dynamic_shape_validation_remain_compatible() {
        assert!(InferenceConfig::new()
            .with_backend(Backend::Vulkan)
            .with_gpu_tuning(GpuTuningMode::Fast)
            .validate()
            .is_err());
        assert!(validate_dynamic_input(6, &[1, 3, 1, 2]).is_ok());
        assert!(validate_dynamic_input(5, &[1, 3, 1, 2]).is_err());
        assert!(validate_dynamic_input(0, &[1, 0]).is_err());
        assert!(validate_dynamic_input(0, &[]).is_err());
    }

    #[test]
    fn pool_permits_report_and_restore_availability() {
        let permits = PoolPermits::new(2);
        assert_eq!(permits.available(), 2);
        let first = permits.acquire();
        assert_eq!(permits.available(), 1);
        let second = permits.acquire();
        assert_eq!(permits.available(), 0);
        drop(first);
        assert_eq!(permits.available(), 1);
        drop(second);
        assert_eq!(permits.available(), 2);
    }

    #[test]
    fn cpu_backend_and_version_are_available() {
        assert!(Backend::CPU.is_available());
        assert!(!get_version().is_empty());
        assert_ne!(get_version(), "unknown");
    }
}
