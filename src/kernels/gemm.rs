//! Packed GEMM shared by pointwise Conv, im2col Conv and MatMul.
//!
//! ```text
//! out[i * n + j] = bias[i] + sum_k w[i * k + kk] * x[kk * n + j]
//! ```
//!
//! `w` is row-major `[m][k]` (Conv weights, or the rows of MatMul's A) and
//! `x` is channel-major `[k][n]` (an NCHW activation, or MatMul's B). The
//! columns of `x` are packed into `NR`-wide panels so the micro-kernel
//! reads two contiguous streams and keeps an `MR x NR` block of
//! accumulators in registers. Threads split the columns, so no thread
//! packs data another thread also packs.

use super::isa::{madd, use_fma};
use crate::par::{Par, SendPtr};

const MR: usize = 4;
const NR: usize = 16;
/// Bytes of packed `x` per panel: stays in L2 while every row block of
/// `w` streams over it.
const PANEL_BYTES: usize = 128 * 1024;

#[derive(Clone, Copy)]
pub struct Gemm<'a> {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    /// `[m][k]`
    pub w: &'a [f32],
    /// `[k][n]`
    pub x: &'a [f32],
    /// `[m]`, or zero.
    pub bias: Option<&'a [f32]>,
}

/// Computes the full `[m][n]` product into `out`.
pub fn gemm(par: &Par, g: &Gemm, out: &mut [f32]) {
    assert!(g.w.len() >= g.m * g.k && g.x.len() >= g.k * g.n);
    assert!(out.len() >= g.m * g.n);
    if let Some(b) = g.bias {
        assert!(b.len() >= g.m);
    }

    let tiles = g.n.div_ceil(NR);
    let parts = par.parts(tiles, g.m * g.k * g.n);
    let ptr = SendPtr(out.as_mut_ptr());
    let n = g.n;

    par.for_ranges(tiles, 1, parts, |t0, t1| {
        let n0 = t0 * NR;
        let n1 = (t1 * NR).min(n);
        // SAFETY: column ranges of different calls are disjoint and
        // inside `out`, checked above.
        unsafe { gemm_cols(g, ptr.get(), n0, n1) }
    });
}

unsafe fn gemm_cols(g: &Gemm, out: *mut f32, n0: usize, n1: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if use_fma() {
            return gemm_cols_fma(g, out, n0, n1);
        }
    }
    gemm_cols_impl::<{ cfg!(target_arch = "aarch64") }>(g, out, n0, n1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_cols_fma(g: &Gemm, out: *mut f32, n0: usize, n1: usize) {
    gemm_cols_impl::<true>(g, out, n0, n1)
}

#[inline(always)]
unsafe fn gemm_cols_impl<const FMA: bool>(g: &Gemm, out: *mut f32, n0: usize, n1: usize) {
    let (m, k, n) = (g.m, g.k, g.n);
    if k == 0 {
        for i in 0..m {
            let b = g.bias.map_or(0.0, |b| b[i]);
            for j in n0..n1 {
                *out.add(i * n + j) = b;
            }
        }
        return;
    }

    let panel_cols = ((PANEL_BYTES / 4 / k) / NR).max(1) * NR;
    let mut packed = vec![0.0f32; k * panel_cols.min((n1 - n0).div_ceil(NR) * NR)];

    let mut j0 = n0;
    while j0 < n1 {
        let nc = (n1 - j0).min(panel_cols);
        let tiles = nc.div_ceil(NR);

        for t in 0..tiles {
            let c0 = j0 + t * NR;
            let width = NR.min(j0 + nc - c0);
            let dst = &mut packed[t * k * NR..(t + 1) * k * NR];
            for (kk, d) in dst.chunks_exact_mut(NR).enumerate() {
                let src = &g.x[kk * n + c0..kk * n + c0 + width];
                d[..width].copy_from_slice(src);
                d[width..].fill(0.0);
            }
        }

        let mut i0 = 0;
        while i0 < m {
            let mr = MR.min(m - i0);
            for t in 0..tiles {
                let p = &packed[t * k * NR..(t + 1) * k * NR];
                let c0 = j0 + t * NR;
                let width = NR.min(j0 + nc - c0);
                let mut acc = [[0.0f32; NR]; MR];
                if let Some(b) = g.bias {
                    for r in 0..mr {
                        acc[r] = [b[i0 + r]; NR];
                    }
                }
                if mr == MR {
                    micro4::<FMA>(&mut acc, p, &g.w[i0 * k..(i0 + MR) * k], k);
                } else {
                    for r in 0..mr {
                        micro1::<FMA>(&mut acc[r], p, &g.w[(i0 + r) * k..(i0 + r + 1) * k]);
                    }
                }
                for (r, row) in acc.iter().enumerate().take(mr) {
                    let o = out.add((i0 + r) * n + c0);
                    std::ptr::copy_nonoverlapping(row.as_ptr(), o, width);
                }
            }
            i0 += MR;
        }

        j0 += nc;
    }
}

#[inline(always)]
fn micro4<const FMA: bool>(acc: &mut [[f32; NR]; MR], p: &[f32], w: &[f32], k: usize) {
    let (w0, rest) = w.split_at(k);
    let (w1, rest) = rest.split_at(k);
    let (w2, w3) = rest.split_at(k);
    let [mut a0, mut a1, mut a2, mut a3] = *acc;

    for ((((xv, &s0), &s1), &s2), &s3) in p
        .chunks_exact(NR)
        .zip(w0)
        .zip(w1)
        .zip(w2)
        .zip(&w3[..k])
    {
        let xv: &[f32; NR] = xv.try_into().unwrap();
        for j in 0..NR {
            a0[j] = madd::<FMA>(a0[j], s0, xv[j]);
            a1[j] = madd::<FMA>(a1[j], s1, xv[j]);
            a2[j] = madd::<FMA>(a2[j], s2, xv[j]);
            a3[j] = madd::<FMA>(a3[j], s3, xv[j]);
        }
    }

    *acc = [a0, a1, a2, a3];
}

#[inline(always)]
fn micro1<const FMA: bool>(acc: &mut [f32; NR], p: &[f32], w: &[f32]) {
    let mut a = *acc;
    for (xv, &s) in p.chunks_exact(NR).zip(w) {
        let xv: &[f32; NR] = xv.try_into().unwrap();
        for j in 0..NR {
            a[j] = madd::<FMA>(a[j], s, xv[j]);
        }
    }
    *acc = a;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(g: &Gemm) -> Vec<f32> {
        let mut out = vec![0.0; g.m * g.n];
        for i in 0..g.m {
            for j in 0..g.n {
                let mut s = g.bias.map_or(0.0, |b| b[i]) as f64;
                for kk in 0..g.k {
                    s += g.w[i * g.k + kk] as f64 * g.x[kk * g.n + j] as f64;
                }
                out[i * g.n + j] = s as f32;
            }
        }
        out
    }

    #[test]
    fn matches_reference_on_odd_shapes() {
        let mut seed = 1u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
        };
        for &(m, k, n) in &[(1, 1, 1), (3, 5, 7), (4, 16, 16), (9, 33, 70), (64, 300, 129)] {
            let w: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
            let x: Vec<f32> = (0..k * n).map(|_| rnd()).collect();
            let b: Vec<f32> = (0..m).map(|_| rnd()).collect();
            for bias in [None, Some(&b[..])] {
                let g = Gemm { m, k, n, w: &w, x: &x, bias };
                let want = reference(&g);
                for threads in [1, 3] {
                    let mut out = vec![f32::NAN; m * n];
                    let par = Par::new(threads);
                    gemm(&par, &g, &mut out);
                    for (a, b) in out.iter().zip(&want) {
                        assert!((a - b).abs() < 1e-4, "{} vs {} ({},{},{})", a, b, m, k, n);
                    }
                }
            }
        }
    }
}
