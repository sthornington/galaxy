fn main() {
    let cuda_lib_candidates = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/sbsa-linux/lib",
    ];

    println!("cargo:rerun-if-changed=native/CMakeLists.txt");
    println!("cargo:rerun-if-changed=native/include/sim_cuda.h");
    println!("cargo:rerun-if-changed=native/src/sim_cuda.cu");

    let dst = cmake::Config::new("native")
        .define("CMAKE_BUILD_TYPE", "Release")
        .build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    for candidate in cuda_lib_candidates {
        if std::path::Path::new(candidate).exists() {
            println!("cargo:rustc-link-search=native={candidate}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{candidate}");
        }
    }
    println!("cargo:rustc-link-lib=static=sim_cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cufft");
    println!("cargo:rustc-link-lib=stdc++");
}
