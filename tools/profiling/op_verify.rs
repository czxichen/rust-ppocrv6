//! 算子数值等价性校验：把优化后的内核输出与标量 f64 参考实现逐元素对比。
//!
//! 目的：本轮优化改动了两个高风险点
//!   1. `Tensor::zeros` -> `Tensor::empty`（若某条写入路径漏写一个输出像素，
//!      该像素就是未初始化内存里的垃圾值）；
//!   2. ConvTranspose2d 2x2/s2 快路径重写为 AVX2（16 宽 / 8 宽）+ 标量尾部，
//!      交错与 permute 索引一旦错位，结果会静默错位而不是崩溃。
//!
//! 端到端测试只做字符串子串断言，抓不住这类漂移。这里用随机张量 + 标量参考
//! 实现做逐元素比对，并且刻意覆盖奇数宽度/小尺寸，逼出所有尾部与边界分支。
//!
//! 运行：cargo run --release --example op_verify

use ppocr::ops::conv2d::{conv2d, Conv2dParams};
use ppocr::ops::conv_transpose2d::{conv_transpose2d, ConvTranspose2dParams};
use ppocr::ops::ConvActivation;
use ppocr::Tensor;

/// 确定性伪随机数（避免依赖 rand crate）
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = (x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as u32;
        // 映射到 [-1.5, 1.5)
        (v as f32 / (1u32 << 23) as f32) - 1.5
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

struct Report {
    case: String,
    max_abs: f64,
    max_rel: f64,
    n: usize,
    worst_idx: usize,
}
impl Report {
    fn verdict(&self) -> &'static str {
        if self.max_abs < 1e-4 {
            "PASS"
        } else if self.max_abs < 1e-2 {
            "WARN"
        } else {
            "FAIL"
        }
    }
}

fn compare(case: String, got: &[f32], want: &[f64]) -> Report {
    assert_eq!(got.len(), want.len(), "长度不一致: {}", case);
    let mut max_abs = 0f64;
    let mut max_rel = 0f64;
    let mut worst = 0usize;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let d = (*g as f64 - *w).abs();
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
        let denom = w.abs().max(1e-3);
        max_rel = max_rel.max(d / denom);
    }
    Report {
        case,
        max_abs,
        max_rel,
        n: got.len(),
        worst_idx: worst,
    }
}

// ---------------------------------------------------------------- ConvTranspose

#[allow(clippy::too_many_arguments)]
fn ref_conv_transpose(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    cin: usize,
    cout: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    padt: usize,
    padl: usize,
    dil_h: usize,
    dil_w: usize,
    group: usize,
    act: ConvActivation,
) -> Vec<f64> {
    let eff_kh = dil_h * (kh - 1) + 1;
    let eff_kw = dil_w * (kw - 1) + 1;
    let oh = (ih - 1) * sh + eff_kh - padt - padt;
    let ow = (iw - 1) * sw + eff_kw - padl - padl;
    let cpg_in = cin / group;
    let cpg_out = cout / group;

    let mut out = vec![0f64; batch * cout * oh * ow];
    for b in 0..batch {
        for g in 0..group {
            for ocg in 0..cpg_out {
                let oc = g * cpg_out + ocg;
                let base = bias.map(|bs| bs[oc] as f64).unwrap_or(0.0);
                for i in 0..oh * ow {
                    out[b * cout * oh * ow + oc * oh * ow + i] = base;
                }
                for icg in 0..cpg_in {
                    let ic = g * cpg_in + icg;
                    for y in 0..ih {
                        for x in 0..iw {
                            let iv = input[((b * cin + ic) * ih + y) * iw + x] as f64;
                            for ky in 0..kh {
                                let oy = (y * sh) as isize - padt as isize + (ky * dil_h) as isize;
                                if oy < 0 || oy >= oh as isize {
                                    continue;
                                }
                                for kx in 0..kw {
                                    let ox =
                                        (x * sw) as isize - padl as isize + (kx * dil_w) as isize;
                                    if ox < 0 || ox >= ow as isize {
                                        continue;
                                    }
                                    // ONNX ConvTranspose weight layout: [C_in, C_out/group, kh, kw]
                                    let wv = weight[((ic * cpg_out + ocg) * kh + ky) * kw + kx]
                                        as f64;
                                    out[((b * cout + oc) * oh + oy as usize) * ow + ox as usize] +=
                                        iv * wv;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if act != ConvActivation::None {
        for v in out.iter_mut() {
            *v = match act {
                ConvActivation::Relu => v.max(0.0),
                _ => *v,
            };
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_ct_case(
    rng: &mut Rng,
    label: &str,
    batch: usize,
    cin: usize,
    cout: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
    out: &mut Vec<Report>,
) {
    let params = ConvTranspose2dParams {
        strides,
        pads,
        dilations,
        group,
        activation: ConvActivation::None,
    };
    let in_data = rng.vec(batch * cin * ih * iw);
    let w_data = rng.vec(cin * (cout / group) * kh * kw);
    let b_data = rng.vec(cout);

    let input = Tensor::from_vec(&[batch, cin, ih, iw], in_data.clone()).unwrap();
    let weight = Tensor::from_vec(&[cin, cout / group, kh, kw], w_data.clone()).unwrap();
    let bias = Tensor::from_vec(&[cout], b_data.clone()).unwrap();

    let got = conv_transpose2d(&input, &weight, Some(&bias), &params).unwrap();
    let want = ref_conv_transpose(
        &in_data,
        &w_data,
        Some(&b_data),
        batch,
        cin,
        cout,
        ih,
        iw,
        kh,
        kw,
        strides[0],
        strides[1],
        pads[0],
        pads[1],
        dilations[0],
        dilations[1],
        group,
        ConvActivation::None,
    );
    let mut r = compare(
        format!(
            "{}  in={}x{} k={}x{} s={:?} p={:?} d={:?} g={} ci={} co={} b={}",
            label, ih, iw, kh, kw, strides, pads, dilations, group, cin, cout, batch
        ),
        got.as_slice(),
        &want,
    );
    r.n = got.as_slice().len();
    out.push(r);
}

// ---------------------------------------------------------------- Conv2d

#[allow(clippy::too_many_arguments)]
fn ref_conv2d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    cin: usize,
    cout: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    padt: usize,
    padl: usize,
    dil_h: usize,
    dil_w: usize,
    group: usize,
) -> Vec<f64> {
    let eff_kh = dil_h * (kh - 1) + 1;
    let eff_kw = dil_w * (kw - 1) + 1;
    let oh = (ih + padt + padt - eff_kh) / sh + 1;
    let ow = (iw + padl + padl - eff_kw) / sw + 1;
    let cpg_in = cin / group;
    let cpg_out = cout / group;

    let mut out = vec![0f64; batch * cout * oh * ow];
    for b in 0..batch {
        for oc in 0..cout {
            let g = oc / cpg_out;
            let base = bias.map(|bs| bs[oc] as f64).unwrap_or(0.0);
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = base;
                    for icg in 0..cpg_in {
                        let ic = g * cpg_in + icg;
                        for ky in 0..kh {
                            let iy =
                                (oy * sh + ky * dil_h) as isize - padt as isize;
                            if iy < 0 || iy >= ih as isize {
                                continue;
                            }
                            for kx in 0..kw {
                                let ix = (ox * sw + kx * dil_w) as isize - padl as isize;
                                if ix < 0 || ix >= iw as isize {
                                    continue;
                                }
                                let iv =
                                    input[((b * cin + ic) * ih + iy as usize) * iw + ix as usize]
                                        as f64;
                                let wv =
                                    weight[((oc * cpg_in + icg) * kh + ky) * kw + kx] as f64;
                                acc += iv * wv;
                            }
                        }
                    }
                    out[((b * cout + oc) * oh + oy) * ow + ox] = acc;
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_conv_case(
    rng: &mut Rng,
    label: &str,
    batch: usize,
    cin: usize,
    cout: usize,
    ih: usize,
    iw: usize,
    strides: [usize; 2],
    pads: [usize; 4],
    dilations: [usize; 2],
    group: usize,
    out: &mut Vec<Report>,
) {
    let params = Conv2dParams {
        strides,
        pads,
        dilations,
        group,
        activation: ConvActivation::None,
    };
    let kh = 3;
    let kw = 3;
    let in_data = rng.vec(batch * cin * ih * iw);
    let w_data = rng.vec(cout * (cin / group) * kh * kw);
    let b_data = rng.vec(cout);

    let input = Tensor::from_vec(&[batch, cin, ih, iw], in_data.clone()).unwrap();
    let weight = Tensor::from_vec(&[cout, cin / group, kh, kw], w_data.clone()).unwrap();
    let bias = Tensor::from_vec(&[cout], b_data.clone()).unwrap();

    let got = conv2d(&input, &weight, Some(&bias), &params).unwrap();
    let want = ref_conv2d(
        &in_data,
        &w_data,
        Some(&b_data),
        batch,
        cin,
        cout,
        ih,
        iw,
        kh,
        kw,
        strides[0],
        strides[1],
        pads[0],
        pads[1],
        dilations[0],
        dilations[1],
        group,
    );
    let mut r = compare(
        format!(
            "{}  in={}x{} k=3x3 s={:?} p={:?} d={:?} g={} ci={} co={} b={}",
            label, ih, iw, strides, pads, dilations, group, cin, cout, batch
        ),
        got.as_slice(),
        &want,
    );
    r.n = got.as_slice().len();
    out.push(r);
}

fn main() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut reports: Vec<Report> = Vec::new();

    // ---------- ConvTranspose2d ----------
    // A. 2x2/s2 快路径（AVX2 16 宽 + 8 宽 + 标量尾部）：刻意覆盖奇数宽度
    for &(ih, iw) in &[
        (1, 1),
        (3, 7),
        (8, 15),
        (8, 16),
        (8, 17),
        (16, 33),
        (13, 41),
        (40, 100),
    ] {
        run_ct_case(
            &mut rng,
            "CT/2x2s2",
            1,
            3,
            8,
            ih,
            iw,
            2,
            2,
            [2, 2],
            [0, 0, 0, 0],
            [1, 1],
            1,
            &mut reports,
        );
    }
    // A2. 快路径 + 多 batch + 较大通道
    run_ct_case(
        &mut rng,
        "CT/2x2s2/b2",
        2,
        4,
        64,
        24,
        37,
        2,
        2,
        [2, 2],
        [0, 0, 0, 0],
        [1, 1],
        1,
        &mut reports,
    );
    // B. 一般路径 k=3 s=2 pad=1（`empty` + bias 初始化是否漏写）
    run_ct_case(
        &mut rng,
        "CT/general",
        1,
        4,
        3,
        7,
        9,
        3,
        3,
        [2, 2],
        [1, 1, 1, 1],
        [1, 1],
        1,
        &mut reports,
    );
    run_ct_case(
        &mut rng,
        "CT/general",
        2,
        8,
        6,
        19,
        23,
        3,
        3,
        [2, 2],
        [1, 1, 1, 1],
        [1, 1],
        1,
        &mut reports,
    );
    // C. 空洞卷积
    run_ct_case(
        &mut rng,
        "CT/dilated",
        1,
        2,
        2,
        6,
        6,
        3,
        3,
        [1, 1],
        [1, 1, 1, 1],
        [2, 2],
        1,
        &mut reports,
    );
    // D. 分组反卷积
    run_ct_case(
        &mut rng,
        "CT/group",
        1,
        4,
        6,
        9,
        11,
        2,
        2,
        [2, 2],
        [0, 0, 0, 0],
        [1, 1],
        2,
        &mut reports,
    );
    // E. 裁剪（pad > kernel 导致输出收缩），确认不发生越界写
    run_ct_case(
        &mut rng,
        "CT/crop",
        1,
        3,
        5,
        12,
        14,
        4,
        4,
        [2, 2],
        [2, 2, 2, 2],
        [1, 1],
        1,
        &mut reports,
    );

    // ---------- Conv2d ----------
    // 首层 RGB 3x3/s2 专用快路径，覆盖奇数宽度
    for &(ih, iw) in &[(64, 63), (64, 64), (65, 65), (128, 127), (32, 33)] {
        run_conv_case(
            &mut rng,
            "Conv/cin3k3s2",
            1,
            3,
            16,
            ih,
            iw,
            [2, 2],
            [0, 1, 0, 1],
            [1, 1],
            1,
            &mut reports,
        );
    }
    // 一般卷积（im2col / gemm 路径）
    run_conv_case(
        &mut rng,
        "Conv/general",
        1,
        8,
        16,
        17,
        19,
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        1,
        &mut reports,
    );
    run_conv_case(
        &mut rng,
        "Conv/stride2",
        2,
        16,
        32,
        33,
        35,
        [2, 2],
        [1, 1, 1, 1],
        [1, 1],
        1,
        &mut reports,
    );
    // 分组卷积
    run_conv_case(
        &mut rng,
        "Conv/group",
        1,
        8,
        8,
        15,
        15,
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        4,
        &mut reports,
    );
    // 深度可分离（depthwise）快路径
    run_conv_case(
        &mut rng,
        "Conv/depthwise",
        1,
        16,
        16,
        64,
        64,
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        16,
        &mut reports,
    );

    // ---------- 输出 ----------
    println!("{:<78} {:>10} {:>6}", "case", "max_abs", "verdict");
    println!("{}", "-".repeat(98));
    let mut fails = 0;
    let mut warns = 0;
    for r in &reports {
        let v = r.verdict();
        if v == "FAIL" {
            fails += 1;
        }
        if v == "WARN" {
            warns += 1;
        }
        let flag = if v == "PASS" { " " } else { "!" };
        println!(
            "{} {:<76} {:>10.3e} {:>6}",
            flag, r.case, r.max_abs, v
        );
        if v != "PASS" {
            println!(
                "       └─ 元素数 {}，最差索引 {}（相对误差 {:.3e}）",
                r.n, r.worst_idx, r.max_rel
            );
        }
    }
    println!("{}", "-".repeat(98));
    println!(
        "合计 {} 项：PASS {} / WARN {} / FAIL {}",
        reports.len(),
        reports.len() - fails - warns,
        warns,
        fails
    );
    if fails > 0 {
        eprintln!("\n存在数值不等价，请检查对应内核实现。");
        std::process::exit(1);
    }
}
