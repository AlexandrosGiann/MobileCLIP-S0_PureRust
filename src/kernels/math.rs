//! Scalar math: erf, the fused GELU, sigmoid.

// The Cephes coefficients are kept exactly as published.
#![allow(clippy::excessive_precision)]

fn polevl(x: f64, coef: &[f64]) -> f64 {
    coef.iter().skip(1).fold(coef[0], |a, &c| a * x + c)
}

fn p1evl(x: f64, coef: &[f64]) -> f64 {
    coef.iter().skip(1).fold(x + coef[0], |a, &c| a * x + c)
}

const ERF_T: [f64; 5] = [
    9.604_973_739_870_516_387_49E0,
    9.002_601_972_038_426_892_17E1,
    2.232_005_345_946_843_192_26E3,
    7.003_325_141_128_050_754_73E3,
    5.559_230_130_103_949_627_68E4,
];

const ERF_U: [f64; 5] = [
    3.356_171_416_475_030_996_47E1,
    5.213_579_497_801_426_797_95E2,
    4.594_323_829_709_801_279_87E3,
    2.262_900_006_138_909_342_46E4,
    4.926_739_426_086_359_210_86E4,
];

const ERFC_P: [f64; 9] = [
    2.461_969_814_735_305_125_24E-10,
    5.641_895_648_310_688_219_77E-1,
    7.463_210_564_422_699_126_87E0,
    4.863_719_709_856_813_666_14E1,
    1.965_208_329_560_770_982_42E2,
    5.264_451_949_954_773_586_31E2,
    9.345_285_271_719_576_075_40E2,
    1.027_551_886_895_157_102_72E3,
    5.575_353_353_693_993_275_26E2,
];

const ERFC_Q: [f64; 8] = [
    1.322_819_511_547_449_925_08E1,
    8.670_721_408_859_897_423_29E1,
    3.549_377_788_878_198_910_62E2,
    9.757_085_017_432_054_897_53E2,
    1.823_909_166_879_097_362_89E3,
    2.246_337_608_187_109_817_92E3,
    1.656_663_091_941_613_501_82E3,
    5.575_353_408_177_276_755_46E2,
];

/// erf, Cephes double-precision rational approximation.
pub fn erf(x: f64) -> f64 {
    let a = x.abs();
    let r = if a <= 1.0 {
        let z = a * a;
        a * polevl(z, &ERF_T) / p1evl(z, &ERF_U)
    } else if a >= 6.0 {
        1.0
    } else {
        1.0 - (-a * a).exp() * polevl(a, &ERFC_P) / p1evl(a, &ERFC_Q)
    };
    if x < 0.0 {
        -r
    } else {
        r
    }
}

/// float32 erf as a rational function of x on [-4, 4] (outside it erf is
/// +-1 in float32): odd degree-13 numerator over even degree-8
/// denominator, the approximation used by Eigen and TensorFlow. Branch
/// free, so loops over it vectorize.
#[inline(always)]
pub fn erf_f32<const FMA: bool>(x: f32) -> f32 {
    let x = x.clamp(-4.0, 4.0);
    let x2 = x * x;
    // One Horner step: p * x2 + c.
    let h = |p: f32, c: f32| super::isa::madd::<FMA>(c, p, x2);
    let mut p = h(-2.726_142_3e-10, 2.770_681_4e-8);
    p = h(p, -2.101_024e-6);
    p = h(p, -5.692_506_4e-5);
    p = h(p, -7.349_906_3e-4);
    p = h(p, -2.954_6e-3);
    p = h(p, -1.609_603_3e-2);
    let mut q = h(-1.456_607_2e-5, -2.133_740_6e-4);
    q = h(q, -1.682_827e-3);
    q = h(q, -7.373_329e-3);
    q = h(q, -1.426_474e-2);
    x * p / q
}

/// The graph's GELU chain, `x * (erf(x / div) + add) * mul`, in place.
pub fn gelu_in_place(x: &mut [f32], div: f32, add: f32, mul: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if super::isa::use_fma() {
            // SAFETY: AVX2 and FMA were detected at run time.
            unsafe { gelu_fma(x, div, add, mul) };
            return;
        }
    }
    gelu_impl::<{ cfg!(target_arch = "aarch64") }>(x, div, add, mul)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_fma(x: &mut [f32], div: f32, add: f32, mul: f32) {
    gelu_impl::<true>(x, div, add, mul)
}

#[inline(always)]
fn gelu_impl<const FMA: bool>(x: &mut [f32], div: f32, add: f32, mul: f32) {
    let inv = 1.0 / div;
    for v in x.iter_mut() {
        let xv = *v;
        *v = xv * (erf_f32::<FMA>(xv * inv) + add) * mul;
    }
}

pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_known_values() {
        let cases = [
            (0.0, 0.0),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (-1.5, -0.966_105_146_475_310_7),
        ];
        for (x, want) in cases {
            assert!((erf(x) - want).abs() < 1e-14, "erf({})", x);
        }
    }

    #[test]
    fn erf_f32_matches_exact() {
        let mut worst = 0.0f64;
        for i in -60000..60000 {
            let x = i as f32 * 1e-4;
            for e in [erf_f32::<false>(x), erf_f32::<true>(x)] {
                worst = worst.max((e as f64 - erf(x as f64)).abs());
            }
        }
        assert!(worst < 1e-6, "max error {}", worst);
    }

    #[test]
    fn gelu_matches_exact() {
        let xs: Vec<f32> = (-4000..4000).map(|i| i as f32 * 0.003).collect();
        let mut g = xs.clone();
        let s2 = std::f32::consts::SQRT_2;
        gelu_in_place(&mut g, s2, 1.0, 0.5);
        for (x, y) in xs.iter().zip(&g) {
            let want = 0.5 * *x as f64 * (1.0 + erf(*x as f64 / std::f64::consts::SQRT_2));
            assert!((*y as f64 - want).abs() < 1e-6, "gelu({}) = {} want {}", x, y, want);
        }
    }
}
