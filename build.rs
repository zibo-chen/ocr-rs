//! Compatibility checks for the historical MNN link-mode features.
//!
//! Native compilation and linking are owned by `mnn-runtime-sys`. This small
//! script only preserves the validation users of `ocr-rs` already rely on.

use std::{env, path::Path};

mod build_support;

fn main() {
    for variable in ["MNN_INCLUDE_DIR", "MNN_LIB_DIR", "DOCS_RS"] {
        println!("cargo:rerun-if-env-changed={variable}");
    }

    if env::var_os("DOCS_RS").is_some() || env::var_os("CARGO_FEATURE_DOCSRS").is_some() {
        return;
    }

    let dynamic = env::var_os("CARGO_FEATURE_MNN_DYNAMIC").is_some();
    let static_link = env::var_os("CARGO_FEATURE_MNN_STATIC").is_some();
    build_support::validate_link_features(dynamic, static_link)
        .unwrap_or_else(|error| panic!("{error}"));

    if build_support::requires_external_installation(dynamic, static_link) {
        validate_external_directory("MNN_INCLUDE_DIR");
        validate_external_directory("MNN_LIB_DIR");
    }
}

fn validate_external_directory(variable: &str) {
    let value = env::var_os(variable).unwrap_or_else(|| {
        panic!("{variable} is required when using `mnn-dynamic` or `mnn-static`")
    });
    let path = Path::new(&value);
    assert!(
        path.is_dir(),
        "{variable}='{}' does not exist or is not a directory",
        path.display()
    );
}
