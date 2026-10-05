//! `rust_ppocrv6` command-line entry point.
//!
//! All parsing, model resolution and reporting live in [`ppocr::cli`] so this file and
//! `examples/ocr_image.rs` cannot drift apart.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(ppocr::cli::main_with_args(&args));
}
