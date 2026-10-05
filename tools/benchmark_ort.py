import time
import numpy as np
import onnxruntime as ort

def benchmark_model(name, session, input_shape, iterations=20, warmup=5):
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input})
        
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        
    avg_t = np.mean(times)
    min_t = np.min(times)
    p50_t = np.median(times)
    p95_t = np.percentile(times, 95)
    print(f"[{name}] Input: {input_shape}")
    print(f"  Avg: {avg_t:.2f} ms | Min: {min_t:.2f} ms | Median: {p50_t:.2f} ms | P95: {p95_t:.2f} ms")
    return avg_t, min_t

def main():
    print("=" * 60)
    print("      ONNX Runtime (CPU) PP-OCRv6 Benchmark       ")
    print(f"      ONNX Runtime Version: {ort.__version__}")
    print("=" * 60)
    
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 0 # default all cores
    
    print("\nLoading ONNX models...")
    t0 = time.perf_counter()
    det_sess = ort.InferenceSession("models_v6/det.onnx", sess_options=opts, providers=["CPUExecutionProvider"])
    rec_sess = ort.InferenceSession("models_v6/rec.onnx", sess_options=opts, providers=["CPUExecutionProvider"])
    print(f"Models loaded in {(time.perf_counter() - t0)*1000.0:.2f} ms")
    
    print("\n--- Detection Model Forward ---")
    det_avg, det_min = benchmark_model("PP-OCRv6 Det", det_sess, [1, 3, 128, 416])
    
    print("\n--- Recognition Model Forward ---")
    rec_avg, rec_min = benchmark_model("PP-OCRv6 Rec (48x320)", rec_sess, [1, 3, 48, 320])
    rec_crop_avg, rec_crop_min = benchmark_model("PP-OCRv6 Rec (48x342 - test image)", rec_sess, [1, 3, 48, 342])
    
    print("\n" + "=" * 60)
    print(f"ONNX Runtime Pure Model Forward Sum (Det + Rec): {det_avg + rec_crop_avg:.2f} ms")
    print("=" * 60)

if __name__ == "__main__":
    main()
