//! Native dynamic-shape regressions and optional device validation for issue #49.
#![cfg(not(feature = "docsrs"))]
use ocr_rs::mnn::SharedRuntime;
use ocr_rs::{Backend, GpuTuningMode, InferenceConfig, InferenceEngine, PrecisionMode};

const MODEL: &[u8] = include_bytes!("../src/mnn/fixtures/layout_identity.mnn");

fn check_dynamic(engine: &InferenceEngine) {
    // Repeated and changing widths exercise resize plan/host-buffer reuse.
    for width in [2, 2, 61, 1024, 61, 2] {
        let values: Vec<f32> = (0..3 * width)
            .map(|v| ((v % 23) as f32 - 7.0) / 8.0)
            .collect();
        let (actual, shape) = engine.run_dynamic_raw(&values, &[1, 3, 1, width]).unwrap();
        assert_eq!(shape, [1, 3, 1, width]);
        assert_eq!(
            actual,
            values.iter().map(|v| v.max(0.0)).collect::<Vec<_>>()
        );
    }
    assert!(engine.run_dynamic_raw(&[1.0], &[1, 3, 1, 2]).is_err());
    assert!(engine.run_dynamic_raw(&[], &[1, 0]).is_err());
}

#[test]
fn dynamic_layout_resize_and_buffer_lengths() {
    let engine = InferenceEngine::from_buffer(MODEL, None).unwrap();
    check_dynamic(&engine);
    let runtime = SharedRuntime::new(&InferenceConfig::new()).unwrap();
    let engine = InferenceEngine::from_buffer_with_runtime(MODEL, &runtime).unwrap();
    check_dynamic(&engine);
}

#[test]
#[ignore = "requires MNN_GPU_TEST_BACKEND and the corresponding GPU-enabled MNN library/device"]
fn gpu_dynamic_layout_matches_known_cpu_result() {
    let backend = match std::env::var("MNN_GPU_TEST_BACKEND").as_deref() {
        Ok("opencl") => Backend::OpenCL,
        Ok("vulkan") => Backend::Vulkan,
        Ok("metal") => Backend::Metal,
        _ => panic!("set MNN_GPU_TEST_BACKEND=opencl|vulkan|metal"),
    };
    assert!(backend.is_available(), "requested backend is unavailable");
    let directory = std::env::temp_dir().join(format!(
        "ocr-gpu-cache-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let config = InferenceConfig::new()
        .with_backend(backend)
        .with_precision(PrecisionMode::High)
        .with_gpu_tuning(GpuTuningMode::Auto)
        .with_gpu_cache_dir(&directory);
    let runtime = SharedRuntime::new(&config).unwrap();
    let engine = InferenceEngine::from_buffer_with_runtime(MODEL, &runtime).unwrap();
    check_dynamic(&engine);
    engine.save_cache().unwrap();
    let files: Vec<_> = std::fs::read_dir(&directory).unwrap().collect();
    assert_eq!(files.len(), 1);
    assert!(files[0].as_ref().unwrap().metadata().unwrap().len() > 0);
    drop(engine);
    drop(runtime);
    // A new interpreter must produce the same result after loading the cache.
    let engine = InferenceEngine::from_buffer(MODEL, Some(config)).unwrap();
    check_dynamic(&engine);
    drop(engine);
    std::fs::remove_dir_all(directory).unwrap();
}
