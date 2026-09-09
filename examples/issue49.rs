//! Compare CPU/GPU OCR output and cold/warm latency for issue #49.
//! Run from the repository: cargo run --example issue49 -- IMAGE [opencl|vulkan|metal] [ITERATIONS] [CACHE_DIR]
use ocr_rs::{Backend, GpuTuningMode, OcrEngine, OcrEngineConfig};
use std::{
    error::Error,
    time::{Duration, Instant},
};

fn texts(engine: &OcrEngine, image: &image::DynamicImage) -> Result<Vec<String>, Box<dyn Error>> {
    Ok(engine
        .recognize(image)?
        .into_iter()
        .map(|r| r.text)
        .collect())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .ok_or("usage: issue49 IMAGE [opencl|vulkan|metal] [ITERATIONS] [CACHE_DIR]")?;
    let backend = match args.get(2).map(String::as_str).unwrap_or("opencl") {
        "opencl" => Backend::OpenCL,
        "vulkan" => Backend::Vulkan,
        "metal" => Backend::Metal,
        _ => return Err("backend must be opencl, vulkan or metal".into()),
    };
    let iterations: usize = args.get(3).map(|v| v.parse()).transpose()?.unwrap_or(10);
    if iterations == 0 {
        return Err("iterations must be positive".into());
    }
    let image = image::open(path)?;
    let load = |config| {
        OcrEngine::new(
            "models/PP-OCRv5_mobile_det.mnn",
            "models/PP-OCRv5_mobile_rec.mnn",
            "models/ppocr_keys_v5.txt",
            Some(config),
        )
    };
    let cpu = load(OcrEngineConfig::new())?;
    let start = Instant::now();
    let expected = texts(&cpu, &image)?;
    println!(
        "CPU reference {:?}, {} regions",
        start.elapsed(),
        expected.len()
    );
    drop(cpu);
    let mut config = OcrEngineConfig::new()
        .with_backend(backend)
        .with_gpu_tuning(GpuTuningMode::Auto);
    if let Some(directory) = args.get(4) {
        config = config.with_gpu_cache_dir(directory);
    }
    let start = Instant::now();
    let gpu = load(config)?;
    println!("GPU load {:?}", start.elapsed());
    let start = Instant::now();
    let first = texts(&gpu, &image)?;
    println!(
        "GPU cold {:?}; exact CPU text match: {}",
        start.elapsed(),
        first == expected
    );
    if first != expected {
        println!("CPU: {expected:#?}\nGPU: {first:#?}");
    }
    let mut durations: Vec<Duration> = Vec::with_capacity(iterations);
    for i in 0..iterations {
        let start = Instant::now();
        let actual = texts(&gpu, &image)?;
        durations.push(start.elapsed());
        println!(
            "warm {} {:?}; stable text: {}",
            i + 1,
            durations[i],
            actual == first
        );
        if actual != first {
            return Err("GPU text changed between identical inputs".into());
        }
    }
    durations.sort_unstable();
    println!(
        "warm P50/P95/P99: {:?}/{:?}/{:?}",
        durations[iterations / 2],
        durations[iterations * 95 / 100],
        durations[iterations * 99 / 100]
    );
    drop(gpu); // flush persistent GPU caches before a second process run
    if first != expected {
        return Err(
            "GPU text differs from CPU; retain output and compare backend/precision".into(),
        );
    }
    Ok(())
}
