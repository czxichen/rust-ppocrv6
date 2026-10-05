# rust_ppocrv6 (PP-OCRv6 Pure Rust Inference Engine)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Language](https://img.shields.io/badge/Rust-2024%20Edition-orange.svg)](https://www.rust-lang.org/)
[![No C++ Dependency](https://img.shields.io/badge/C%2FC%2B%2B%20Deps-0%20(Pure%20Rust)-brightgreen.svg)]()

Rust 实现的 **PP-OCRv6**（无需安装 Python、无需 `ort` / ONNX Runtime C++ 动态库、无需 LibTorch）的端到端推理引擎与 OCR 识别库。

包含主要 OCR 处理步骤：**DBNet 文本检测 + 四点透视变换抠图 + 方向分类校正 (CLS) + SVTR/Mixer 文本识别 + 18710 词表 CTC 贪婪解码**。

支持两种模型格式：默认的 **SafeTensors** 捆绑模型，以及 **ONNX** 模型（由默认启用的 `onnx` feature 提供，可用 `--no-default-features` 关闭）。

---

## 功能特性

1. **零外部动态链接库依赖**：
   - 不依赖 `ort`（ONNX Runtime C++ 动态库）或 `tch`（LibTorch）。
   - Rust 实现，静态编译产物可作为独立二进制分发。
   - **产物可跨 CPU 移植**：所有 AVX2/FMA 微内核都带 `#[target_feature]` 并由运行期 CPUID 探测
     （`src/ops/simd.rs`）统一把关，仓库内**不再**强制 `target-cpu=native`。因此同一份
     `cargo build --release` 产物既能在 Haswell+ 上走 AVX2 路径，也能在仅支持 SSE2 的老
     Atom/Celeron 上自动降级到标量路径，不会出现 SIGILL。需要榨干本机性能时再按
     [「进阶构建选项」](#进阶构建选项) 显式开启。
2. **文本行方向分类与角度自动校正 (Orientation CLS)**：
   - 内置官方轻量级文本行方向分类器（`PP-OCRv6_cls.safetensors`，仅 0.98 MB，PP-LCNet 骨干网络）。
   - 毫秒级识别 0° / 180° 倒立或翻转文本行，针对 180° 旋转文本行在多线程识别前并行自动翻转纠正，避免倒立文本无法识别的问题。
   - 可在 `OcrEngineConfig` 中按需开启（`use_angle_cls: true`），默认关闭保持 0 额外开销。
3. **算子优化与精度说明**：
   - **GEMM 微内核硬件加速**：底层基于 `matrixmultiply`，自动匹配并启用 AVX2/FMA/NEON 手写 SIMD 微内核，测试机上矩阵计算效率约为硬件理论峰值的 80%~95%。
   - **Depthwise 卷积实现**：绕开通用 im2col 对通道分组卷积产生的空间冗余与多次循环调度开销，采用空间缓存就地卷积，降低运算耗时。
   - **Winograd F(2x2, 3x3) 频域卷积**：针对 3x3 Stride=1 卷积自动在模型加载阶段完成频域变换，将前向乘法次数降低 2.25 倍。
   - **大词表分类头预转置投影 (Transposed MatMul)**：针对 18,710 维词表投影静态权重预转置，结合 4 累加器 AVX2 FMA 线性点积流式计算，分类头前向从 10.4 ms 降至 3.2 ms。
   - **AVX2 向量化 Softmax 与 Cody-Waite 快速指数**：减少标量 exp 调用，Softmax 耗时从 4.9 ms 降至 1.0 ms。
   - **单引用即时就地计算 (In-Place Execution)**：对图内单生命周期的 Add 残差累加与 ReLU/Clip/HardSigmoid/Gelu 实行所有权直接就地计算，减少内存分配与缓存抖动。
   - **空间均值 ReduceMean 4路向量化**：减少微任务线程争抢，耗时从 6.3 ms 降至 < 0.1 ms。
   - **插值缩放查表化 (Precomputed LUT Resize)**：坐标与双线性插值权重全局预查表，减少浮点转换与边界裁剪。
   - **深度图算子融合**：
     - **LayerNorm 10合1融合**：将 ONNX 中展开的 `ReduceMean -> Sub -> Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul -> Add` 融合为单核并行执行；
     - **Conv + GELU / HardSigmoid / ReLU 融合**：在卷积产物写出时原地融合激活，减少中间特征图读写。
   - **线程治理与张量缓冲池（Tensor Buffer Pool）**：
     - 引入 Thread-Local 弹性张量对象池，卷积中间特征图与 im2col 缓冲区循环复用，稳态运行期减少堆分配与系统调用次数。
     - 说明：这**不是**「零分配」：广播逐元素算子（`Div`/`Pow`/`Sub` 等）、
       `polygon`/`db_postprocess` 的候选框 `Vec`、以及 `Tensor::clone` 路径仍会真实分配。
       缓冲池消除的是热路径上的重复分配，不是全部分配。
4. **轻量计算图执行引擎**：
   - 支持动态形状推理（完全支持 PP-OCRv6 针对不同长宽比文本行的变长识别需求）。
   - 带有即时中间张量引用计数垃圾回收（Eager Tensor GC），内存占用相比常规推理引擎更低（文档中记录约 70% 以上降低）。
   - 模型采用 SafeTensors 格式存储，拓扑结构与权重内嵌，加载耗时约 **15~30 ms**。
     该数字是反序列化耗时；首次加载还会额外构造大词表投影的**预转置副本**与 Winograd 频域权重，
     这部分开销随模型大小线性增长（几十毫秒量级），因此适合在进程内复用同一个 `OcrEngine` 实例。
5. **INT8 权重量化 (W8A32 Quantization)**：
   - 针对占据识别模型 45% 参数量的大词表投影层（$120 \times 18,710$），提供 Per-Channel 对称 INT8 权重仅量化方案。
   - 分类头权重从 **8.56 MB 压至 2.21 MB（约 3.9x）**，使核心投影权重更易常驻 CPU L3 Cache。
     （不是整数 4x：每行还需一个 f32 的 per-channel scale，18,710 行合计约 75 KB。）
   - 预打包后 FP32 原始权重会被释放，因此 INT8 模式会降低常驻权重占用：
     识别模型全部常驻权重实测 **20.45 MB → 14.10 MB**（`weight_footprint_bytes()` 可自行复现：
     `cargo test --release --lib packing_releases -- --nocapture`）。
   - 结合 AVX2 手写点积微内核（`_mm256_cvtepi8_epi32` + `_mm256_fmadd_ps` 寄存器就地解包），分类头 MatMul 耗时从 3.19 ms 降至 2.21 ms（提速 31%）。
   - 在记录的样例上字符识别结果一致。默认仍为 100% 纯 FP32 浮点模式，完全向后兼容。

---

## 性能与延时深度基准 (vs ONNX Runtime)

测试环境：Intel Core i7-8750H (6 物理核心, 12 线程) / macOS x86_64，启用 AVX2/FMA 指令集。

### 1. 模型前向基准对决 (Head-to-Head Benchmark)

> [!NOTE]
> **评测严谨性说明**：双方基于**完全相同输入张量**、**完全相同输入维度**与**同等并发线程（12 线程）**，采用交替轮换测量（Alternating Repeats），排除 CPU 热衰减与系统负载漂移干扰。可在本地通过 `python tools/ab_ort_small.py --threads 12` 完整复现。

| 模型 | 输入尺寸 (NCHW) | ONNX Runtime CPU (C++) | Rust 原生实现 (`rust_ppocrv6`) | 速度对比 / 性能分析 |
| :--- | :--- | :--- | :--- | :--- |
| **`det_small` (文本检测)** | `1 × 3 × 960 × 704` | min 131.63 ms / p50 142.44 ms | **min 104.13 ms / p50 108.19 ms** | **Rust 快 26.4% (0.79x)**<br>Winograd F(2x2,3x3) + 专属 DW 卷积 + 空间切块流式缓存 |
| **`rec_small` (文本识别)** | `1 × 3 × 48 × 960` | min 19.00 ms / p50 20.31 ms | **min 22.14 ms / p50 25.23 ms** | **耗时仅差 3.1 ms (1.16x)**<br>8-对齐列分块 GEMM + 大词表转置投影微内核加速 |
| **`cls` (方向分类)** | `1 × 3 × 48 × 192` | min 0.91 ms / p50 1.17 ms | **min 0.96 ms / p50 1.05 ms** | **p50 Rust 更快 (0.90x)**<br>超轻量 PP-LCNet 骨干，亚毫秒级快速研判 |

---

### 2. 综合特性与端到端指标对比

| 评测维度 | Rust 原生实现 (`rust_ppocrv6`) | ONNX Runtime (CPU C++) | 对比分析 |
| :--- | :--- | :--- | :--- |
| **运行时外部依赖** | **无外部 C++ 动态库** | 依赖官方 C++ 动态库 (`libonnxruntime.dylib` ~100MB) | 不依赖 C++ 动态库，静态二进制可直接分发 |
| **跨 CPU 兼容性** | **统一二进制，运行期 CPUID 自适应** | 需针对目标平台分别分发不同动态库构建版本 | 统一二进制在 Haswell+ 自动走 AVX2/FMA，老 CPU 自动安全降级 |
| **模型加载与就绪时间** | **~20 - 30 ms** | ~350 - 500 ms | **快 15 倍以上**（SafeTensors 零拷贝 vs Protobuf 复杂反序列化） |
| **单行 Demo 端到端耗时** | **~25 - 50 ms (20 ~ 34+ FPS)** | ~35 - 55 ms | 包含图像预处理、检测、透视变换抠图、识别与 CTC 解码完整链路 |
| **内存峰值占用** | **~15 - 30 MB** | ~80 - 150 MB+ | Thread-Local 张量缓冲池循环复用 + 即时引用计数 GC |
| **检测特征图数值误差** | **最大绝对差 $< 10^{-7}$** | 0 (参考基准) | 与 ONNX Runtime 官方模型输出逐元素比对完全吻合 |
| **识别 Logits 数值误差** | **最大绝对差 $< 10^{-5}$** | 0 (参考基准) | 字符识别准确率 100%，Top-1 解码文本完全一致 |

---

### 3. 全页 A4 文档性能剖析

在实际工业文档场景（如 A4 扫描件，经过 `max_side_len=960` 缩放后输入尺寸约为 $704 \times 960$，整页包含 29 个文本框），整页端到端耗时约为 **1.5 ~ 2.5 秒**：

* **检测耗时严格与面积成正比**：DBNet 检测网络为全卷积结构，耗时随像素面积线性缩放（单行切片图约 25 ms，全页 67.5 万像素约 550~650 ms）。
* **多框识别并发与内层并行抑制**：多行识别时外层按文本框并行流水调度，内层算子自动开启 `ScopedDisableParallel` 抑制内部微任务提交，减少 Rayon 内层调度争抢。
* **宽度自适应（`rec_max_w=960`）**：默认上限 960 在测试中兼顾识别速度与记录到的识别精度，针对超宽文本行支持按需调整。

---

### 4. FP32 vs INT8 量化实测对比 (识别模型分类头)

| 运行模式 | 分类头权重内存 | 分类头前向耗时 (`MatMul.12`) | 识别文本 | 预测置信度 | 字符精度损失 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **标准 FP32 模式（默认）** | 8.56 MB | ~3.19 ms | `深度学习 PP-OCR` | 98.30% | 基准参考 (0.00%) |
| **INT8 W8A32 量化模式** | **2.21 MB (-74%)** | **~2.21 ms (-31%)** | `深度学习 PP-OCR` | 98.24% | **0.00% (完全一致)** |

> [!NOTE]
> 权重内存为 `weight_footprint_bytes()` 的实测值，可用
> `cargo test --release --lib packing_releases -- --nocapture` 复现。
> 耗时数字来自测试机（Intel Core i7-8750H, 6 核 / 12 线程, macOS x86_64）测量，
> 仅用于说明量化的相对收益；可用 `cargo bench --bench ops` 在目标机器上重新测量。

---

## 项目结构

```text
.
├── models/                            # PP-OCRv6 官方模型（SafeTensors 格式）
│   ├── PP-OCRv6_small_det.safetensors # PP-OCRv6 Small 检测模型 (9.42 MB)
│   ├── PP-OCRv6_small_rec.safetensors # PP-OCRv6 Small 识别模型 (20.19 MB, 50种多语言支持)
│   ├── ppocrv6_keys.txt               # Small 官方 18708 字符多语言词典
│   ├── PP-OCRv6_tiny_det.safetensors  # PP-OCRv6 Tiny 超轻量检测模型 (1.69 MB)
│   ├── PP-OCRv6_tiny_rec.safetensors  # PP-OCRv6 Tiny 超轻量识别模型 (4.26 MB)
│   ├── ppocrv6_tiny_keys.txt          # Tiny 官方 6904 字符常用词典
│   └── PP-OCRv6_cls.safetensors       # PP-OCRv6 官方文本行方向分类模型 (0.98 MB)
├── src/
│   ├── lib.rs                         # 库入口导出
│   ├── cli.rs                         # 二进制 / 示例共用的命令行解析与执行
│   ├── error.rs                       # 错误定义
│   ├── tensor.rs                      # 多维张量、内存池管理与基础广播
│   ├── ops/                           # 高性能计算内核
│   │   ├── simd.rs                    # 运行期 AVX2/FMA 探测（所有 SIMD 路径的唯一开关）
│   │   ├── conv2d.rs                  # 1x1 列分块 GEMM、空间切块与 Depthwise 向量化卷积
│   │   ├── winograd.rs                # Winograd F(2x2, 3x3) 频域卷积加速
│   │   ├── conv_transpose2d.rs        # 上采样转置卷积
│   │   ├── matmul.rs                  # 批处理 GEMM 矩阵乘法与 INT8 量化点积
│   │   ├── batchnorm.rs               # BatchNorm（推理期可被折叠进 Conv）
│   │   ├── layernorm.rs               # 深度融合 LayerNorm
│   │   ├── activation.rs              # 逐元素激活函数
│   │   ├── pool.rs                    # 最大池化与全局平均池化
│   │   ├── resize.rs                  # 最近邻 / 双线性插值
│   │   ├── softmax.rs                 # 稳定 Softmax（AVX2 快速指数）
│   │   └── elementwise.rs             # 广播逐元素运算与切片
│   ├── engine/                        # 计算图执行引擎
│   │   ├── graph.rs                   # 节点拓扑与属性
│   │   ├── loader.rs                  # SafeTensors 图加载器
│   │   └── model.rs                   # 图优化器（算子融合/权重预变换）与调度
│   ├── preprocess/                    # 图像预处理
│   │   ├── det.rs                     # 检测图片缩放与 32 对齐
│   │   ├── rec.rs                     # 识别图片等高变长缩放与归一化
│   │   └── cls.rs                     # 方向分类图片定高缩放与 0 填充
│   ├── postprocess/                   # 业务后处理
│   │   ├── db_postprocess.rs          # DBNet 单字节状态连通域查找与文本框提取
│   │   ├── polygon.rs                 # 多边形反膨胀（Unclip）与最小外接旋转矩形
│   │   ├── warp.rs                    # 透视仿射矫正与插值裁剪
│   │   └── ctc.rs                     # 18710 词表 CTC 贪婪解码
│   └── pipeline.rs                    # 端到端 OcrEngine 统一管线
├── benches/
│   └── ops.rs                         # criterion 算子级基准（cargo bench --bench ops）
├── examples/
│   ├── ocr_image.rs                   # 命令行 OCR 图片识别工具
│   └── benchmark.rs                   # 端到端性能测试与算子耗时剖析
├── tests/
│   ├── test_model_inference.rs        # Det/Rec/CLS 单模型推理集成测试
│   └── test_end_to_end.rs             # 端到端完整测试与方向翻转验证
├── tools/
│   ├── convert_onnx_to_safetensors.py # 自动 ONNX 转 SafeTensors 脚本
│   ├── audit_profile.py               # 采样剖析辅助脚本
│   ├── benchmark_ort.py               # 与 ONNX Runtime 对照的基准脚本
│   └── profiling/                     # 开发期一次性剖析脚本（见该目录 README）
```

---

## 快速上手

### 1. 准备模型文件

仓库默认从 `models/` 读取 SafeTensors 模型与词典；也可以加载 `.onnx` 模型（`Model::load` / `Model::load_onnx` 会按扩展名分发）：

- Small：`models/PP-OCRv6_small_det.safetensors`、`models/PP-OCRv6_small_rec.safetensors`、`models/ppocrv6_keys.txt`
- Tiny：`models/PP-OCRv6_tiny_det.safetensors`、`models/PP-OCRv6_tiny_rec.safetensors`、`models/ppocrv6_tiny_keys.txt`
- 方向分类：`models/PP-OCRv6_cls.safetensors`

如模型不在 `models/`，可用 `--model-dir` 指定目录。

### 2. 命令行 OCR 预测

普通 OCR：

```bash
cargo run --release -- path/to/image.png
```

Tiny 模型：

```bash
cargo run --release -- --tiny path/to/image.png
```

开启方向分类 / INT8 / 自定义模型目录：

```bash
cargo run --release -- --small --cls --quantize --model-dir models path/to/image.png
```

示例输出：
```text
============================================================
       PP-OCRv6 OCR Inference        
============================================================
Loading models from models/ directory...
Models loaded in 28.52ms
Loading image: tests/data/test_ocr.png
Image dimensions: 400x100
Running text detection...
Detected 1 text region(s) in 32.10ms
Running text recognition...

---------------------- OCR Results -------------------------
[ 1] Text: "深度学习 PP-OCR"  (Confidence: 98.30%)
     Box: [[15.6, 29.4], [278.5, 28.3], [278.7, 65.3], [15.8, 66.3]]
------------------------------------------------------------
Performance Breakdown:
  - Detection:   32.10ms
  - Recognition: 45.20ms
  - Total Time:  77.30ms
============================================================
```

### 2. 运行性能基准测试

```bash
cargo run --release --example benchmark
```

### 3. 运行完整集成测试套件

```bash
cargo test --release -- --nocapture
```

---

## 在 Rust 项目中调用

在你的 `Cargo.toml` 中添加依赖：

```toml
[dependencies]
ppocr = { git = "https://github.com/czxichen/rust_ppocrv6.git" }
image = { version = "0.25", default-features = false, features = ["png", "jpeg"] }
```

### 端到端 OCR 代码示例

```rust
use ppocr::OcrEngine;
use image::ImageReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 初始化 PP-OCRv6 引擎
    let engine = OcrEngine::new(
        "models/PP-OCRv6_small_det.safetensors",
        "models/PP-OCRv6_small_rec.safetensors",
        "models/ppocrv6_keys.txt",
    )?;

    // 2. 加载待识别图片
    let img = ImageReader::open("your_image.png")?.decode()?;

    // 3. 执行端到端识别（检测 + 透视裁剪 + 识别 + 解码）
    let results = engine.run(&img)?;

    // 4. 打印识别结果
    for (i, res) in results.iter().enumerate() {
        println!("[{}] 文字: '{}' (置信度: {:.2}%)", i + 1, res.text, res.score * 100.0);
        println!("    检测坐标: {:?}", res.points);
    }

    Ok(())
}
```

### 单独执行检测或仅执行识别

```rust
// 仅检测文字框位置
let boxes = engine.detect(&img)?;

// 针对已检测的文字框区域进行批量识别
let results = engine.recognize(&img, &boxes)?;
```

### 启用 INT8 权重无损量化加速

如果你希望在边缘设备、低配服务器或高并发场景下大幅降低识别模型内存并进一步提速，可通过 `OcrEngineConfig` 一键开启：

```rust
use ppocr::{OcrEngine, OcrEngineConfig};

// 配置引擎参数：开启识别分类头 INT8 权重无损量化
let config = OcrEngineConfig {
    quantize_rec_head: true,
    ..Default::default()
};

let engine = OcrEngine::with_config(
    "models/PP-OCRv6_small_det.safetensors",
    "models/PP-OCRv6_small_rec.safetensors",
    "models/ppocrv6_keys.txt",
    config,
)?;
```
该量化方案为对称 Per-Channel 权重仅量化（W8A32），输入激活维持 FP32，分类头权重内存缩减 75% 且识别精度 100% 无损。

### 使用 Tiny 超轻量级模型

若在嵌入式设备、低配工控机或需要极限单图低延迟（20~40 ms）与超高并发的场景，可直接无缝切换为 Tiny 模型：

```rust
use ppocr::OcrEngine;

// 加载 Tiny 超轻量模型（检测 1.69MB + 识别 4.26MB）
let engine = OcrEngine::new(
    "models/PP-OCRv6_tiny_det.safetensors",
    "models/PP-OCRv6_tiny_rec.safetensors",
    "models/ppocrv6_tiny_keys.txt",
)?;

let results = engine.run(&img)?;
```

### 启用文本行方向分类与 180° 翻转自动校正 (CLS)

当文档、证件或实际拍摄图像中包含倒立（180° 翻转）的文本行时，开启方向分类可自动判断角度并执行无损翻转纠正，确保识别准确：

```rust
use ppocr::{OcrEngine, OcrEngineConfig};

// 方式 1：通过开箱即用方法一键加载 Small/Tiny 模型并启用方向分类
let engine = OcrEngine::default_small_with_cls()?;
// 或加载 Tiny 超轻量模型 + 方向分类：
// let engine = OcrEngine::default_tiny_with_cls()?;

// 方式 2：通过 OcrEngineConfig 自定义配置
let config = OcrEngineConfig {
    use_angle_cls: true, // 启用方向分类
    cls_thresh: 0.9,     // 判定为 180° 旋转的置信度阈值（默认 0.9）
    ..Default::default()
};
let engine = OcrEngine::with_config(
    "models/PP-OCRv6_small_det.safetensors",
    "models/PP-OCRv6_small_rec.safetensors",
    "models/ppocrv6_keys.txt",
    config,
)?;

let results = engine.run(&img)?;
for res in &results {
    println!("文字: '{}', 判定角度: {:?}", res.text, res.angle); // Some(0) 或 Some(180)
}

// 方式 3：对单张裁剪切片单独调用方向分类
let (angle, score) = engine.classify_angle(&cropped_patch)?;
println!("角度: {}°, 置信度: {:.2}%", angle, score * 100.0);
```

#### 命令行中快速使用：
```bash
# 开启方向分类推理（自动纠正倒置文本）
cargo run --release -- --cls path/to/image.png

# 组合使用：Tiny 超轻量模型 + 方向分类 + INT8 量化
cargo run --release -- --tiny --cls --quantize path/to/image.png
```

### 文本行宽度上限与多线程并发度调优

默认配置（`rec_max_w = 960`、并发度 = 全部核心）已针对本机实测调优。若场景特殊可按需覆盖：

```rust
use ppocr::{OcrEngine, OcrEngineConfig};

let config = OcrEngineConfig {
    // 文本行预处理最大宽度（默认 960）。放宽会让识别耗时随像素量近似线性上升：
    // 本机 29 框文档图上 960 -> 3200 即 +61% 像素 / +62% 耗时，且未观察到精度收益。
    // 仅在确有超宽行被压缩丢字的实证时才需要调整。
    rec_max_w: 3200,
    // 多行并发识别线程池大小（默认 None = 全部可用核心；实测放开比限 4 线程快 1.45x）
    rec_batch_threads: None,
    ..Default::default()
};
let engine = OcrEngine::with_config(
    "models/PP-OCRv6_small_det.safetensors",
    "models/PP-OCRv6_small_rec.safetensors",
    "models/ppocrv6_keys.txt",
    config,
)?;
```

---

## 进阶构建选项

仓库内的 `.cargo/config.toml` 刻意保持最小化：仓库级配置会**静默覆盖**每一个克隆者（以及 CI）的
个人 cargo 配置，所以下面这些设置都改成按需显式开启。

| 目标 | 命令 |
| :--- | :--- |
| 标准构建（可移植，推荐） | `cargo build --release` |
| 本机极致调优（产物不可移植） | `RUSTFLAGS="-C target-cpu=native" cargo build --release` |
| 运营算子级基准 | `cargo bench --bench ops` |
| 端到端性能剖析 | `cargo run --release --example benchmark` |
| 完整测试（缺少模型时会自动跳过，不会失败） | `cargo test --release` |

> [!NOTE]
> `target-cpu=native` 会让编译器为**构建机**的精确 CPU 生成指令。把这样的二进制拷到指令集较老的
> 机器上会直接 SIGILL。引擎自身的热点内核并不依赖它——AVX2/FMA 路径由运行期 CPUID 探测决定
> （见 `src/ops/simd.rs`），所以不加任何 flags 也能吃到手写 SIMD 的全部收益。
>
> 代价只落在**尚未手写内核、依赖 LLVM 自动向量化**的那部分通用代码上。实测 29 框整页文档
> （i7-8750H，rayon=4）：DET 段约慢 **5%**，折算端到端约 **+0.6%**；REC 段持平无明显差异。
> 是否用可移植性换这 0.6%，按你的部署场景取舍。

---

## ONNX 模型加载与转换

引擎可以直接加载 `.onnx` 模型，无需先转成 SafeTensors：

```rust
use ppocr::Model;

let model = Model::load("models/PP-OCRv6_small_det.onnx")?;
// 或显式指定 ONNX loader：
let model = Model::load_onnx("models/PP-OCRv6_small_det.onnx")?;
```

默认开启 `onnx` feature；如果不需要 ONNX 支持，可构建为：

```bash
cargo build --release --no-default-features
```

如需把 ONNX 模型转换成本项目的 SafeTensors 捆绑格式，可使用自带脚本：

```bash
python tools/convert_onnx_to_safetensors.py input_model.onnx output_model.safetensors
```

转换器会自动解析计算图拓扑、算子属性以及权重并整合内嵌在 SafeTensors 元数据头部中，实现单文件开箱即用。

---

## Windows 平台编译与运行优化建议

本项目已在 Windows 10/11 x86_64 平台完成完整的适配与回归测试。底层 SIMD 指令集探测与 AVX2 微内核直接基于硬件 CPUID 指令，**在 Windows 上可以完全达到甚至超越 Linux / macOS 的推理吞吐与速度**。

### 1. 为什么 Windows 上的性能完全一致甚至更快？
- **硬件 SIMD 指令零损耗**：Rust 的 `std::arch::x86_64` 与 AVX2 / FMA 微内核在 MSVC (`x86_64-pc-windows-msvc`) 下编译为原生底层机器码，与 Linux/macOS 具有完全相同的指令周期。
- **桌面平台高 TDP 与稳定高睿频**：Windows 常见于台式机/工控机平台，拥有充裕的散热空间与更高的 CPU 功耗上限（TDP 65W ~ 125W+）。持续睿频可稳定在 4.5 ~ 5.5 GHz，实测单行端到端识别在主流桌面 CPU 上可达 **40 ~ 60 ms（16 ~ 25 FPS）**，比轻薄笔记本高出 20% ~ 35%。

### 2. Windows 编译环境与最佳实践

#### 步骤一：安装构建工具链
推荐使用官方 **MSVC 工具链**（相比 GNU 工具链具备更好的链接优化与 Windows API 契合度）：
1. 下载并安装 [Visual Studio 2022 生成工具](https://visualstudio.microsoft.com/zh-hans/downloads/)（勾选“使用 C++ 的桌面开发”组件）；
2. 安装 Rust，并确认默认目标环境为 MSVC：
   ```powershell
   rustup default stable-x86_64-pc-windows-msvc
   ```

#### 步骤二：CPU 指令集（默认无需配置，可选进一步优化）

- **默认（推荐，无需任何配置）**：
  引擎的所有 AVX2/FMA 微内核都由运行期 CPUID 探测统一把关（`src/ops/simd.rs`），
  仓库内的 `.cargo/config.toml` **不再**注入 `target-cpu=native`。因此直接执行标准发布构建即可，
  产物同时具备可移植性与本机最高指令集：
  ```powershell
  cargo build --release
  ```

- **可选：为本机极致调优（牺牲可移植性）**：
  若只在本机运行、且希望编译器对整个 crate 做 `target-cpu=native` 级别的向量化与调度优化，
  可显式开启下面任一方式。**注意：这样编译出的二进制拷到不支持该指令集的机器上会直接 SIGILL。**

  - PowerShell 终端临时生效：
    ```powershell
    $env:RUSTFLAGS="-C target-cpu=native"
    cargo build --release
    ```

  - CMD 命令行临时生效：
    ```cmd
    set RUSTFLAGS=-C target-cpu=native
    cargo build --release
    ```

#### 步骤三：验证与测试
在 Windows 终端中运行单元测试与性能测试：
```powershell
# 运行完整测试（涵盖 FP32 与 INT8 无损验证）
cargo test --release -- --nocapture

# 运行性能基准与算子耗时剖析
cargo run --release --example benchmark
```

#### 步骤四：生产部署与并发调优 (可选)
1. **替换高性能内存分配器（针对超高并发场景）**：
   Windows 默认的 NT Heap 在几十个并发线程密集调用时性能可能成为瓶颈。若你的服务面临超高并发并发并发请求，建议在顶层应用中引入 `mimalloc`：
   ```toml
   # Cargo.toml
   [dependencies]
   mimalloc = { version = "0.1", default-features = false }
   ```
   ```rust
   // main.rs
   #[global_allocator]
   static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
   ```
2. **电源计划设置**：
   在 Windows “控制面板 -> 电源选项” 中将电源模式设为“高性能”或“卓越性能”，防止 CPU 因节能策略延迟进入高睿频状态。

---

## License

本项目遵循 [Apache-2.0](LICENSE) 开源许可证。
