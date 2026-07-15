fn main() {
    // Prefer an explicit CUDA root from the environment, then fall back to the
    // usual install locations on x86-64 and Grace (sbsa) systems.
    let mut cuda_lib_candidates: Vec<String> = Vec::new();
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        println!("cargo:rerun-if-env-changed={var}");
        if let Ok(root) = std::env::var(var) {
            cuda_lib_candidates.push(format!("{root}/lib64"));
            cuda_lib_candidates.push(format!("{root}/targets/sbsa-linux/lib"));
        }
    }
    cuda_lib_candidates.push("/usr/local/cuda/lib64".to_string());
    cuda_lib_candidates.push("/usr/local/cuda/targets/sbsa-linux/lib".to_string());

    // Watch the whole native tree so new .cu/.h files trigger rebuilds.
    println!("cargo:rerun-if-changed=native");
    println!("cargo:rerun-if-env-changed=CMAKE_CUDA_ARCHITECTURES");

    let mut config = cmake::Config::new("native");
    config.define("CMAKE_BUILD_TYPE", "Release");
    let arch = std::env::var("CMAKE_CUDA_ARCHITECTURES").unwrap_or_else(|_| "native".to_string());
    config.define("CMAKE_CUDA_ARCHITECTURES", arch);
    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    for candidate in cuda_lib_candidates {
        if std::path::Path::new(&candidate).exists() {
            println!("cargo:rustc-link-search=native={candidate}");
            // Note: rustc-link-arg only applies to this package's own binaries
            // (solver_diag, force_check); downstream binaries such as
            // sim-server rely on the system loader configuration knowing the
            // CUDA library directory.
            println!("cargo:rustc-link-arg=-Wl,-rpath,{candidate}");
        }
    }
    println!("cargo:rustc-link-lib=static=sim_cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cufft");
    println!("cargo:rustc-link-lib=stdc++");
}
