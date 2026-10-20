//! CPU feature selection for the hot kernels.
//!
//! Kernels are written once, generic over `FMA`, and instantiated twice on
//! x86-64: a baseline build and one compiled with AVX2 + FMA that is picked
//! at run time. On AArch64 fused multiply-add is always available.

use std::sync::atomic::{AtomicU8, Ordering};

/// 0 = not decided, 1 = no FMA, 2 = FMA.
static FMA_STATE: AtomicU8 = AtomicU8::new(0);

/// Whether the AVX2 + FMA kernels may run. `MOBILECLIP_NO_FMA=1` forces
/// the baseline kernels.
#[inline]
pub fn use_fma() -> bool {
    match FMA_STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = detect() && std::env::var_os("MOBILECLIP_NO_FMA").is_none();
            FMA_STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn detect() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[cfg(target_arch = "aarch64")]
fn detect() -> bool {
    true
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect() -> bool {
    false
}

/// Name of the kernel set in use, for reports.
pub fn kernel_set() -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "aarch64 NEON/FMA"
    } else if use_fma() {
        "x86-64 AVX2+FMA"
    } else {
        "portable"
    }
}

/// `acc + a * b`, fused when `FMA`.
#[inline(always)]
pub fn madd<const FMA: bool>(acc: f32, a: f32, b: f32) -> f32 {
    if FMA {
        a.mul_add(b, acc)
    } else {
        acc + a * b
    }
}
