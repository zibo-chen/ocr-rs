# GPU OCR: cold starts, wide tensors and numerical validation

Issue: https://github.com/zibo-chen/rust-paddle-ocr/issues/49

The existing OpenCL Buffer default fixes GPU image-width allocation failures.
The follow-up reported a very slow first OpenCL inference and incorrect Vulkan
text. The Rust wrapper previously selected WIDE tuning for every OpenCL/Vulkan
session, including sessions resized to many different recognition widths.

## Defaults and tradeoffs

- `GpuTuningMode::Auto` now selects FAST for OpenCL and NONE for Vulkan. This
  reduces tuning work during cold/dynamic inference. It does not eliminate
  shader compilation and can trade steady-state throughput for startup latency.
- `with_gpu_tuning(Wide)` and `Heavy` remain explicit options for fixed-shape,
  long-running workloads. Vulkan rejects the OpenCL-only Fast/Normal modes.
- OpenCL retains Buffer memory. Its memory flags are never sent to Vulkan.
- `OcrEngineConfig` maps Normal precision to High for Vulkan. MNN 3.6.1's Vulkan
  buffer backend enables fp16 unless High is requested; High is a conservative
  OCR default for long recognition sequences. Explicit Low is preserved. The
  lower-level `InferenceConfig` continues to pass its precision through unchanged.
- Non-finite recognition scores now produce a diagnostic instead of entering
  CTC decoding. This cannot detect every possible incorrect but finite result.

High precision mitigates a plausible numerical cause of the Vulkan report;
Windows/NVIDIA results must still be compared on the reporter's hardware. It is
not evidence that every Vulkan kernel or model combination has been repaired.

## Persistent kernel caches

```rust,no_run
use ocr_rs::{Backend, OcrEngineConfig};
let config = OcrEngineConfig::new()
    .with_backend(Backend::OpenCL)
    .with_gpu_cache_dir(".cache/ocr-gpu");
```

Cache file names include a SHA-256 of model bytes, linked MNN version, backend,
precision, tuning and memory configuration. Detection and recognition caches
therefore remain separate even in the same directory. MNN loads the file before
session creation. `InferenceEngine::save_cache()` saves after warmup; normal
engine destruction also attempts a save. Destruction cannot report cache write
failures, so use `save_cache()` when persistence must be confirmed. Keep caches
local to one device/driver environment and clear them after driver upgrades.
Cache writes should not be shared by concurrent processes.

Dynamic inference skips resize for an unchanged shape and reuses host buffers.
Changed shapes must pass MNN's resize/allocation status check before copying or
running. Input lengths, rank and multiplication limits are checked before FFI;
output copy failures leave no partially returned allocation.

## Reproduce and compare

Run from the repository with the appropriate GPU-enabled native MNN library:

```bash
cargo run --features mnn-dynamic --example issue49 -- paper.png opencl 10 .cache/ocr-gpu
cargo run --features mnn-dynamic --example issue49 -- paper.png vulkan 10 .cache/ocr-gpu
```

For external libraries, set `MNN_INCLUDE_DIR`, `MNN_LIB_DIR` and the platform's
runtime library search path as described in the main README. Repeat each command
in a new process to measure persistent-cache reuse. Use both the original large
page and the small line. The example reports CPU reference text, GPU load/cold
latency, warm P50/P95/P99 and exact text consistency. It returns a nonzero status
when GPU text differs from CPU or changes across identical inputs.

The small native fixture also validates changing widths, packed output layout,
and malformed input sizes. Run it against an available device:

```bash
MNN_GPU_TEST_BACKEND=opencl cargo test --features mnn-dynamic --test gpu_regressions -- --ignored --nocapture
```

Set the backend to `vulkan` or `metal` for those devices. The default CPU
regression is always run; device tests are opt-in and require the actual backend.
A passing synthetic fixture does not replace the real OCR text comparison.
