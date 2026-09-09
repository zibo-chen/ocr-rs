//! GPU tuning policy shared by native and documentation builds.

/// Kernel tuning effort. More tuning trades first-run latency for steady-state speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuTuningMode {
    /// Fast OpenCL tuning; no Vulkan tuning. Suitable for variable-width OCR.
    #[default]
    Auto,
    /// Disable tuning on OpenCL/Vulkan.
    None,
    /// Small OpenCL tuning search (unsupported by Vulkan).
    Fast,
    /// Medium OpenCL tuning search (unsupported by Vulkan).
    Normal,
    /// Wide OpenCL/Vulkan tuning search; can be very slow for many input widths.
    Wide,
    /// Exhaustive OpenCL/Vulkan tuning search.
    Heavy,
}

impl GpuTuningMode {
    #[cfg(not(feature = "docsrs"))]
    pub(crate) fn bits(self, vulkan: bool) -> Option<i32> {
        match self {
            Self::Auto => Some(if vulkan { 1 } else { 1 << 4 }),
            Self::None => Some(1),
            Self::Fast if !vulkan => Some(1 << 4),
            Self::Normal if !vulkan => Some(1 << 3),
            Self::Wide => Some(1 << 2),
            Self::Heavy => Some(1 << 1),
            _ => None,
        }
    }
}
