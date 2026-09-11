//! Compile-time coverage for the public MNN API exposed since ocr-rs 2.0.

use ndarray::{ArrayD, ArrayViewD};
use ocr_rs::mnn::{DataFormat, MnnError, Result, SessionPool, SharedRuntime};
use ocr_rs::{
    Backend, GpuMemoryMode, GpuTuningMode, InferenceConfig, InferenceEngine, PrecisionMode,
};
use std::path::PathBuf;

type DynamicRawFn = fn(&InferenceEngine, &[f32], &[usize]) -> Result<(Vec<f32>, Vec<usize>)>;

fn assert_send_sync<T: Send + Sync>() {}

fn from_file_compat(path: PathBuf, config: Option<InferenceConfig>) -> Result<InferenceEngine> {
    InferenceEngine::from_file(path, config)
}

#[test]
fn historical_types_and_signatures_compile() {
    assert_send_sync::<SharedRuntime>();
    assert_send_sync::<InferenceEngine>();
    assert_send_sync::<SessionPool>();

    let _: fn(&InferenceConfig) -> Result<SharedRuntime> = SharedRuntime::new;
    let _: fn(&[u8], Option<InferenceConfig>) -> Result<InferenceEngine> =
        InferenceEngine::from_buffer;
    let _: fn(PathBuf, Option<InferenceConfig>) -> Result<InferenceEngine> = from_file_compat;
    let _: fn(&[u8], &SharedRuntime) -> Result<InferenceEngine> =
        InferenceEngine::from_buffer_with_runtime;
    let _: for<'a> fn(&InferenceEngine, ArrayViewD<'a, f32>) -> Result<ArrayD<f32>> =
        InferenceEngine::run;
    let _: fn(&InferenceEngine, &[f32], &mut [f32]) -> Result<()> = InferenceEngine::run_raw;
    let _: for<'a> fn(&InferenceEngine, ArrayViewD<'a, f32>) -> Result<ArrayD<f32>> =
        InferenceEngine::run_dynamic;
    let _: DynamicRawFn = InferenceEngine::run_dynamic_raw;
    let _: fn(&InferenceEngine) -> &[usize] = InferenceEngine::input_shape;
    let _: fn(&InferenceEngine) -> &[usize] = InferenceEngine::output_shape;
    let _: fn(&InferenceEngine) -> bool = InferenceEngine::has_dynamic_shape;
    let _: fn(&InferenceEngine) -> Result<()> = InferenceEngine::save_cache;

    let _: fn(&InferenceEngine, usize, Option<InferenceConfig>) -> Result<SessionPool> =
        SessionPool::new;
    let _: for<'a> fn(&SessionPool, ArrayViewD<'a, f32>) -> Result<ArrayD<f32>> = SessionPool::run;
    let _: fn(&SessionPool) -> usize = SessionPool::available;
}

#[test]
fn historical_enum_variants_fields_and_discriminants_are_stable() {
    let config = InferenceConfig {
        thread_count: 4,
        precision_mode: PrecisionMode::Normal,
        use_cache: false,
        data_format: DataFormat::NCHW,
        backend: Backend::CPU,
        gpu_memory_mode: GpuMemoryMode::Auto,
        gpu_tuning_mode: GpuTuningMode::Auto,
        gpu_cache_dir: None,
    };
    assert_eq!(config.thread_count, 4);
    assert_eq!(PrecisionMode::Normal as i32, 0);
    assert_eq!(PrecisionMode::Low as i32, 1);
    assert_eq!(PrecisionMode::High as i32, 2);
    assert_eq!(DataFormat::NCHW as i32, 0);
    assert_eq!(DataFormat::NHWC as i32, 1);
    assert_eq!(DataFormat::Auto as i32, 2);
    assert_eq!(GpuMemoryMode::Auto as i32, 0);
    assert_eq!(GpuMemoryMode::Buffer as i32, 1 << 6);
    assert_eq!(GpuMemoryMode::Image as i32, 1 << 7);

    let errors = [
        MnnError::InvalidParameter(String::new()),
        MnnError::OutOfMemory,
        MnnError::RuntimeError(String::new()),
        MnnError::Unsupported,
        MnnError::ModelLoadFailed(String::new()),
        MnnError::BackendUnavailable(String::new()),
        MnnError::NullPointer,
        MnnError::ShapeMismatch {
            expected: vec![],
            got: vec![],
        },
    ];
    assert_eq!(errors.len(), 8);
}
