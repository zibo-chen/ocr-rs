#[path = "../build_support.rs"]
mod build_support;

use build_support::{requires_external_installation, validate_link_features, BuildConfigError};

#[test]
fn legacy_link_features_remain_mutually_exclusive() {
    assert_eq!(
        validate_link_features(true, true),
        Err(BuildConfigError::ConflictingLinkModes)
    );
    assert_eq!(validate_link_features(true, false), Ok(()));
    assert_eq!(validate_link_features(false, true), Ok(()));
    assert_eq!(validate_link_features(false, false), Ok(()));
}

#[test]
fn legacy_explicit_link_modes_still_require_an_external_installation() {
    assert!(requires_external_installation(true, false));
    assert!(requires_external_installation(false, true));
    assert!(!requires_external_installation(false, false));
}

#[test]
fn historical_features_forward_to_mnn_runtime() {
    let manifest = include_str!("../Cargo.toml");
    for mapping in [
        r#"metal = ["mnn-runtime/metal"]"#,
        r#"opencl = ["mnn-runtime/opencl"]"#,
        r#"opengl = ["mnn-runtime/opengl"]"#,
        r#"vulkan = ["mnn-runtime/vulkan"]"#,
        r#"cuda = ["mnn-runtime/cuda"]"#,
        r#"coreml = ["mnn-runtime/coreml"]"#,
        r#"build-mnn-from-source = ["mnn-runtime/build-from-source"]"#,
        r#"mnn-dynamic = ["mnn-runtime/dynamic"]"#,
        r#"mnn-static = ["mnn-runtime/static"]"#,
        r#"static-cpp-runtime = ["mnn-runtime/static-cpp-runtime"]"#,
    ] {
        assert!(
            manifest.contains(mapping),
            "missing feature mapping: {mapping}"
        );
    }
    assert!(manifest.contains("default = []"));
    assert!(manifest.contains("docsrs = []"));
}
