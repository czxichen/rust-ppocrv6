//! Runtime CPU feature detection.
//!
//! Every wider-than-baseline SIMD kernel in this crate must be reachable only
//! through one of the helpers here. On x86_64 the *baseline* is SSE2, so an
//! intrinsic such as `_mm256_add_ps` compiled into a function that is not
//! annotated with `#[target_feature(enable = "avx2")]` becomes an illegal
//! instruction (SIGILL) on older Atom / Celeron / pre-Haswell AMD parts.
//!
//! Two rules keep this sound:
//!   1. the function containing the intrinsics carries
//!      `#[target_feature(enable = "avx2,fma")]` and is `unsafe fn`;
//!   2. its only call sites are guarded by [`avx2_fma_enabled`].

#[cfg(target_arch = "x86_64")]
use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(target_arch = "x86_64")]
const UNKNOWN: u8 = 0;
#[cfg(target_arch = "x86_64")]
const AVAILABLE: u8 = 1;
#[cfg(target_arch = "x86_64")]
const UNAVAILABLE: u8 = 2;

#[cfg(target_arch = "x86_64")]
static AVX2_FMA_STATE: AtomicU8 = AtomicU8::new(UNKNOWN);

/// Test-only override: `0` = use the real CPU probe, `1` = force "enabled",
/// `2` = force "disabled".
///
/// This exists so the AVX2 kernels and their scalar fallbacks can be cross-checked against
/// each other on a machine that has AVX2. It is a **process-global** setting rather than a
/// thread-local one on purpose: the ops fan out to the rayon pool, and a thread-local flag
/// would not reach the workers. Compiled out entirely outside `cfg(test)`.
#[cfg(all(test, target_arch = "x86_64"))]
static AVX2_FMA_TEST_OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// See [`AVX2_FMA_TEST_OVERRIDE`]. `None` restores the real CPU probe.
#[cfg(all(test, target_arch = "x86_64"))]
pub fn set_avx2_fma_override(value: Option<bool>) {
    AVX2_FMA_TEST_OVERRIDE.store(
        match value {
            None => 0,
            Some(true) => 1,
            Some(false) => 2,
        },
        Ordering::Relaxed,
    );
}

/// Whether the host CPU itself implements AVX2+FMA, ignoring any test override.
#[cfg(all(test, target_arch = "x86_64"))]
pub fn cpu_has_avx2_fma() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

/// Run `f` with the AVX2/FMA decision forced to `enabled`, holding a process-wide lock so
/// that concurrently running tests cannot observe (or clobber) each other's override.
///
/// Forcing `true` on a CPU without AVX2 would execute illegal instructions, so in that case
/// the override degrades to `false` and `f` simply runs on the scalar path.
#[cfg(all(test, target_arch = "x86_64"))]
pub static SIMD_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(all(test, target_arch = "x86_64"))]
pub fn with_avx2_override<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    let _guard = SIMD_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    set_avx2_fma_override(Some(enabled && cpu_has_avx2_fma()));
    let out = f();
    set_avx2_fma_override(None);
    out
}

/// Whether the current CPU implements both AVX2 and FMA.
///
/// The answer is cached after the first probe (a single relaxed atomic load on
/// the hot path). Always `false` on non-x86_64 targets.
#[inline]
pub fn avx2_fma_enabled() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(test)]
        match AVX2_FMA_TEST_OVERRIDE.load(Ordering::Relaxed) {
            1 => return true,
            2 => return false,
            _ => {}
        }
        match AVX2_FMA_STATE.load(Ordering::Relaxed) {
            AVAILABLE => true,
            UNAVAILABLE => false,
            _ => {
                let available = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
                AVX2_FMA_STATE.store(
                    if available { AVAILABLE } else { UNAVAILABLE },
                    Ordering::Relaxed,
                );
                available
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_is_stable_across_calls() {
        #[cfg(target_arch = "x86_64")]
        let _guard = SIMD_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let first = avx2_fma_enabled();
        for _ in 0..1000 {
            assert_eq!(avx2_fma_enabled(), first);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn override_is_observable_and_restored() {
        let _guard = SIMD_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let real = avx2_fma_enabled();
        set_avx2_fma_override(Some(false));
        assert!(!avx2_fma_enabled());
        if cpu_has_avx2_fma() {
            set_avx2_fma_override(Some(true));
            assert!(avx2_fma_enabled());
        }
        set_avx2_fma_override(None);
        assert_eq!(avx2_fma_enabled(), real, "the override must not leak out");
    }
}
