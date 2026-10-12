fn main() {
    println!("cargo:rerun-if-changed=engine/src/model.cpp");
    println!("cargo:rerun-if-changed=engine/include/clodex/model.h");
    cc::Build::new()
        .cpp(true)
        .file("engine/src/model.cpp")
        .include("engine/include")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-O3")
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra")
        .compile("clodex_engine");
}
