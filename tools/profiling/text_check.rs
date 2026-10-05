//! 文本正确性对照：量化 `rec_max_w` 放宽到底修好了什么、代价多大。
//!
//! `rec_max_w` 由 960 放宽到 3200 会让宽行的 rec 输入宽度显著变大（本机文档图上
//! 实测像素量 +61%），这是纯粹的性能代价。要判断这笔代价是否值得，必须看到它
//! 在文本层面的收益：到底有没有修掉行尾丢字、修掉了多少。
//!
//! 运行：cargo run --release --example text_check -- <图片> [真值行号前缀]

use image::{DynamicImage, ImageReader};
use ppocr::{OcrEngine, OcrEngineConfig};

const DET: &str = "models/PP-OCRv6_small_det.safetensors";
const REC: &str = "models/PP-OCRv6_small_rec.safetensors";
const DICT: &str = "models/ppocrv6_keys.txt";

fn load(path: &str) -> DynamicImage {
    ImageReader::open(path).expect("open").decode().expect("decode")
}

fn run(path: &str, max_w: u32, threads: usize) -> Vec<String> {
    let img = load(path);
    let cfg = OcrEngineConfig {
        rec_max_w: max_w,
        rec_batch_threads: Some(threads),
        ..Default::default()
    };
    let engine = OcrEngine::with_config(DET, REC, DICT, cfg).expect("engine");
    engine
        .run(&img)
        .expect("ocr")
        .into_iter()
        .map(|r| r.text)
        .collect()
}

fn main() {
    let img = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/data/bench_doc.png".to_string());
    let threads: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    let a = run(&img, 960, threads);
    let b = run(&img, 3200, threads);

    println!("图片: {}   (线程数 {})\n", img, threads);
    println!("识别行数: max_w=960 -> {} | max_w=3200 -> {}\n", a.len(), b.len());

    // 按文本匹配行（两版顺序应一致，但框数可能不同，做长度对齐比较）
    let n = a.len().max(b.len());
    let mut diff = 0usize;
    // 行数少时输出完整文本，便于逐字核对尾部是否丢字
    let full = n <= 4;
    if full {
        println!("(行数少，输出完整文本)\n");
    }
    println!("{:<4} {:<46} {:<46} {}", "序号", "max_w=960", "max_w=3200", "判定");
    println!("{}", "-".repeat(126));
    for i in 0..n {
        let sa = a.get(i).map(|s| s.as_str()).unwrap_or("<缺失>");
        let sb = b.get(i).map(|s| s.as_str()).unwrap_or("<缺失>");
        let same = sa == sb;
        if !same {
            diff += 1;
        }
        let fmt = |s: &str| -> String {
            if full {
                return format!("{} ({}字)", s, s.chars().count());
            }
            let c: Vec<char> = s.chars().collect();
            if c.len() > 44 {
                format!("{}…({}字)", c[..43].iter().collect::<String>(), c.len())
            } else {
                s.to_string()
            }
        };
        if full {
            println!("\n[{}] max_w=960 : {}\n      max_w=3200: {}\n      判定: {}", i, fmt(sa), fmt(sb), if same { "相同" } else { "不同" });
        } else {
            println!(
                "{:<4} {:<46} {:<46} {}",
                i,
                fmt(sa),
                fmt(sb),
                if same { "" } else { "不同" }
            );
        }
    }
    println!("{}", "-".repeat(126));
    println!("共 {} 行，其中 {} 行识别结果不同\n", n, diff);

    // 字符数对比：丢字的直接证据
    let ca: usize = a.iter().map(|s| s.chars().count()).sum();
    let cb: usize = b.iter().map(|s| s.chars().count()).sum();
    println!("总识别字符数: max_w=960 -> {} | max_w=3200 -> {}  (差 {} 字)", ca, cb, cb as i64 - ca as i64);
    println!("\n提示：若 max_w=960 的个别行明显短于 3200 版本，即为行尾丢字。");
}
