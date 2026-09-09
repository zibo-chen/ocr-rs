# Test fixtures

- `issue_45_mixed_orientation.png` is the original image attached to
  [Issue #45](https://github.com/zibo-chen/rust-paddle-ocr/issues/45). It covers
  horizontal and 90°-rotated Chinese text in one image. SHA-256:
  `3b2f90b09e2b2a1eb6fcd4ef0e5f642aca547f4bbc6d8b5aa3dfb7d10f5e7fe3`.

`src/mnn/fixtures/layout_identity.mnn` is a project-owned, weight-free Input → ConvertTensor → ReLU graph (392 bytes), generated from the MNN 3.6.0 schema. It exercises contiguous NCHW data crossing an NC4HW4 output and supports varying input widths. It lives under `src` so packaged library unit tests can also include it.
