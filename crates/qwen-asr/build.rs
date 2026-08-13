fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let blas_backend = std::env::var("QWEN_BLAS_BACKEND").unwrap_or_default();
    println!("cargo:rerun-if-env-changed=QWEN_BLAS_BACKEND");
    println!("cargo:rerun-if-env-changed=OPENBLAS_LIB_DIR");
    println!("cargo:rustc-check-cfg=cfg(qwen_openblas)");

    // Check if blas feature is enabled via CARGO_FEATURE_BLAS env var
    if std::env::var("CARGO_FEATURE_BLAS").is_ok() {
        if blas_backend.eq_ignore_ascii_case("openblas") {
            let dir = std::env::var("OPENBLAS_LIB_DIR")
                .unwrap_or_else(|_| panic!("QWEN_BLAS_BACKEND=openblas requires OPENBLAS_LIB_DIR"));
            println!("cargo:rustc-cfg=qwen_openblas");
            println!("cargo:rustc-link-search=native={dir}");
            println!("cargo:rustc-link-lib=openblas");
            if target_os == "macos" {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
            }
            return;
        }
        match target_os.as_str() {
            "macos" => {
                println!("cargo:rustc-link-lib=framework=Accelerate");
            }
            "linux" => {
                println!("cargo:rustc-link-lib=openblas");
            }
            "android" => {
                if let Ok(dir) = std::env::var("OPENBLAS_ANDROID_LIB_DIR") {
                    println!("cargo:rustc-link-search=native={dir}");
                }
                if std::env::var("OPENBLAS_ANDROID_STATIC").ok().as_deref() == Some("1") {
                    println!("cargo:rustc-link-lib=static=openblas");
                } else {
                    println!("cargo:rustc-link-lib=openblas");
                }
            }
            _ => {
                // No BLAS available, will use fallback matmul
            }
        }
    }
}
