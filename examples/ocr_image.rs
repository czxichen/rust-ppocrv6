//! Minimal example: run PP-OCRv6 over one image from the command line.
//!
//! Shares the exact implementation used by the `rust_ppocrv6` binary — see
//! [`ppocr::cli`] — so this example and `src/main.rs` can never diverge.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(ppocr::cli::main_with_args(&args));
}
