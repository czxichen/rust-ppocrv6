# Profiling & one-off measurement scripts

These are the ad-hoc scripts that produced the performance numbers discussed during
development. They are kept for provenance, not as a supported interface, and they are
deliberately **outside `examples/`** so that a plain `cargo test --all-targets` no longer
builds a dozen one-off binaries.

The supported, maintained benchmark is [`examples/benchmark.rs`](../../examples/benchmark.rs)
plus the criterion harness in [`benches/`](../../benches).

## Running one of these

They are ordinary cargo examples; copy the one you need back into `examples/` and run it:

```bash
cp tools/profiling/op_probe.rs examples/
cargo run --release --example op_probe
rm examples/op_probe.rs
```

Requirements vary: most need the `models/` directory populated (see the main README) and
`profile_*.rs` expect the benchmark images produced by `gen_bench_images.py` /
`gen_cjk_stress.py`.

| Script | Purpose |
| :--- | :--- |
| `fwd_ab.rs` | Forward-only A/B driver for **cross-revision** comparisons. Uses only the API slice that has stayed stable (`Model::load` / `Model::graph` / `Model::forward` / `Tensor::from_vec`), so the same source compiles against an older `git worktree` and the current tree — that is what makes the two numbers an A/B instead of two measurements of two programs. Takes a raw `f32` blob plus its shape, so preprocessing cannot leak into the comparison. `fwd_ab <model.safetensors> <input.bin> <N,C,H,W> [--iters N] [--label L] [--out out.bin]` |
| `ab_latency.rs` | Minimal A/B probe: one model pair, one image, fixed iters, plus a recognised-text checksum so two builds can be proven to do identical work. Designed to run under `/usr/bin/time -l`. |
| `int8_probe.rs` | FP32 vs INT8 (W8A32) recognition measured in the same loop, and reports the resident weight footprint of each. |
| `bench_speed.rs` | Older latency sweep across the det/rec/cls models. |
| `concurrency_probe.rs` | Measures how `rec_batch_threads` affects a 29-box page. |
| `op_probe.rs` | Times individual graph operators via `forward_with_profile`. |
| `op_verify.rs` | Numerical cross-checks of the fused operators against reference values. |
| `profile_all.rs` | Whole-model profiler across all bundled models. |
| `profile_breakdown.rs` | Latency breakdown for the end-to-end pipeline. |
| `profile_convs.rs` | Per-convolution timings, used to pick kernel strategies. |
| `text_check.rs` | Prints recognised text for a set of stress images. |
| `batch_poc.rs` | Proof-of-concept for batched recognition throughput. |
| `gen_bench_images.py` | Generates the synthetic benchmark images. |
| `gen_cjk_stress.py` | Generates the wide CJK/ASCII stress images. |

## Measuring on a shared machine

Wall-clock latency is only trustworthy on an otherwise idle host. On a loaded machine the
same binary can vary by **>50% between runs** — that is contention, not a regression. Two
habits make the numbers usable:

1. **Use CPU time (`user+sys`) as the primary metric.** Contention stretches wall-clock but
   does not change how much CPU the process actually burns, so
   `cargo run --release --example <probe>` under `/usr/bin/time -l` stays comparable across runs.
2. **Pin rayon below the physical core count** (`RAYON_NUM_THREADS=4` on a 6C/12T part) to stop
   SMT oversubscription from amplifying the noise, and alternate the order of the builds
   between rounds so any thermal drift cancels out.

Also verify the candidates do *the same work* — `ab_latency.rs` prints a box/character
checksum for exactly this reason. A comparison between different outputs is meaningless.

## Three traps that silently invalidate an A/B

**1. Compile flags must be verified, not assumed.** Cargo resolves extra rustc flags from
four *mutually exclusive* sources and uses only the first one that is set:

1. the `RUSTFLAGS` environment variable,
2. all matching `target.<triple>.rustflags` / `target.<cfg>.rustflags` config entries,
3. the `build.rustflags` config value,
4. (see the cargo book for the fourth)

Anything deeper in that list is **ignored**, not merged. So the repository's
`[build] rustflags = ["-C","target-cpu=native"]` does *nothing* for a developer whose global
`~/.cargo/config.toml` sets `[target.x86_64-pc-windows-msvc] rustflags = [...]`. Print the
real command line before believing a comparison:

```bash
touch examples/<probe>.rs && cargo build --release --example <probe> -v 2>&1 \
  | grep -- "--crate-name <probe>" | tr ' ' '\n' | grep -E "target-cpu|target-feature"
```

**2. `time` in Git Bash under-reports CPU on Windows.** It only covers the main thread, so an
8-thread rayon workload that really burns ~2.6 s of CPU is reported as `user 0.015s`. Either
measure **single-threaded wall clock** (which then equals CPU time) or use a process-wide
timer. A "CPU time unchanged" conclusion drawn from that `time` is not evidence of anything.

**3. Isolate the input.** Feed both sides a raw `f32` blob captured from the real pipeline
(`fwd_ab.rs`) rather than each side running its own preprocessing — otherwise a preprocessing
change is silently folded into the "kernel" number.

