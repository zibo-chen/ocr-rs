//! Pure compatibility checks shared by `build.rs` and integration tests.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildConfigError {
    ConflictingLinkModes,
}

impl fmt::Display for BuildConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConflictingLinkModes => formatter
                .write_str("features `mnn-dynamic` and `mnn-static` are mutually exclusive"),
        }
    }
}

impl std::error::Error for BuildConfigError {}

pub fn validate_link_features(dynamic: bool, static_link: bool) -> Result<(), BuildConfigError> {
    if dynamic && static_link {
        Err(BuildConfigError::ConflictingLinkModes)
    } else {
        Ok(())
    }
}

pub const fn requires_external_installation(dynamic: bool, static_link: bool) -> bool {
    dynamic || static_link
}
