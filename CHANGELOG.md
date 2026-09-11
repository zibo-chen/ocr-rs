
## Unreleased

- Delegate native MNN ownership to `mnn-runtime` while preserving the existing
  `ocr_rs::mnn` API, feature names, dynamic inference, GPU caches, and session-pool surface.
- Reduce default OpenCL cold-start tuning; expose explicit GPU tuning modes and
  persistent model/configuration-specific kernel caches.
- Use High precision for Vulkan OCR's Normal preset; preserve explicit Low.
- Validate dynamic resize/allocation and copy results, reject invalid buffer
  lengths before FFI, reuse same-shape host tensors, and reject non-finite CTC scores.
- Add issue #49 CPU/GPU comparison tooling and dynamic-layout regression tests.
