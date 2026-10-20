//! 2-D convolution, NCHW, batch 1 per call.
//!
//! * 1x1 stride-1 unpadded, group 1: one GEMM on the activation as is.
//! * one input channel per group (depthwise, and the channel-multiplier
//!   layers with `M = 2 * C`): per-plane kernel on a zero-padded copy.
//! * everything else: im2col per group, then GEMM.

use super::gemm::{gemm, Gemm};
use super::isa::{madd, use_fma};
use crate::error::{bail, Result};
use crate::par::Par;

#[derive(Clone, Debug)]
pub struct ConvParams {
    pub group: usize,
    /// top, left, bottom, right
    pub pads: [usize; 4],
    pub strides: [usize; 2],
    pub dilations: [usize; 2],
}

impl ConvParams {
    fn out_size(&self, h: usize, w: usize, kh: usize, kw: usize) -> Result<(usize, usize)> {
        let eh = self.dilations[0] * (kh - 1) + 1;
        let ew = self.dilations[1] * (kw - 1) + 1;
        let hp = h + self.pads[0] + self.pads[2];
        let wp = w + self.pads[1] + self.pads[3];
        if hp < eh || wp < ew {
            bail!("Conv kernel larger than padded input");
        }
        Ok(((hp - eh) / self.strides[0] + 1, (wp - ew) / self.strides[1] + 1))
    }
}

/// `x`: `[N, C, H, W]`, `w`: `[M, C / group, KH, KW]`, `b`: `[M]`.
/// Returns the output and its shape `[N, M, OH, OW]`.
pub fn conv2d(
    par: &Par,
    x: &[f32],
    xs: &[usize],
    w: &[f32],
    ws: &[usize],
    b: Option<&[f32]>,
    p: &ConvParams,
) -> Result<(Vec<f32>, Vec<usize>)> {
    if xs.len() != 4 || ws.len() != 4 {
        bail!("Conv expects 4-D input and weight, got {:?} and {:?}", xs, ws);
    }
    let (n, c, h, wd) = (xs[0], xs[1], xs[2], xs[3]);
    let (m, cg, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
    let g = p.group.max(1);
    if c != cg * g || m % g != 0 {
        bail!("Conv group mismatch: C={} Cg={} M={} group={}", c, cg, m, g);
    }
    if let Some(b) = b {
        if b.len() != m {
            bail!("Conv bias has {} values for {} outputs", b.len(), m);
        }
    }

    let (oh, ow) = p.out_size(h, wd, kh, kw)?;
    let image = c * h * wd;
    let out_image = m * oh * ow;
    let mut out = vec![0.0f32; n * out_image];

    for bi in 0..n {
        let xb = &x[bi * image..(bi + 1) * image];
        let ob = &mut out[bi * out_image..(bi + 1) * out_image];

        let pointwise = kh == 1
            && kw == 1
            && g == 1
            && p.strides == [1, 1]
            && p.pads == [0, 0, 0, 0];

        if pointwise {
            let gm = Gemm { m, k: c, n: h * wd, w, x: xb, bias: b };
            gemm(par, &gm, ob);
        } else if cg == 1 && g == c && p.dilations == [1, 1] {
            depthwise(par, xb, c, h, wd, w, m, kh, kw, b, p, oh, ow, ob);
        } else {
            im2col_conv(par, xb, c, h, wd, w, m, cg, kh, kw, b, p, oh, ow, ob);
        }
    }

    Ok((out, vec![n, m, oh, ow]))
}

#[allow(clippy::too_many_arguments)]
fn im2col_conv(
    par: &Par,
    x: &[f32],
    c: usize,
    h: usize,
    wd: usize,
    w: &[f32],
    m: usize,
    cg: usize,
    kh: usize,
    kw: usize,
    b: Option<&[f32]>,
    p: &ConvParams,
    oh: usize,
    ow: usize,
    out: &mut [f32],
) {
    let g = c / cg;
    let mg = m / g;
    let q = cg * kh * kw;
    let pix = oh * ow;
    let mut cols = vec![0.0f32; q * pix];

    for gi in 0..g {
        // cols[(ic * KH + ky) * KW + kx][oy * OW + ox]: one weight row's order.
        let cols_ptr = crate::par::SendPtr(cols.as_mut_ptr());
        let fill = |r0: usize, r1: usize| {
            for row in r0..r1 {
                let ic = row / (kh * kw);
                let ky = (row / kw) % kh;
                let kx = row % kw;
                let plane = &x[(gi * cg + ic) * h * wd..(gi * cg + ic + 1) * h * wd];
                // SAFETY: each row index is written by one range only.
                let dst = unsafe { std::slice::from_raw_parts_mut(cols_ptr.get().add(row * pix), pix) };
                for oy in 0..oh {
                    let iy = (oy * p.strides[0] + ky * p.dilations[0]) as isize - p.pads[0] as isize;
                    let drow = &mut dst[oy * ow..(oy + 1) * ow];
                    if iy < 0 || iy >= h as isize {
                        drow.fill(0.0);
                        continue;
                    }
                    let srow = &plane[iy as usize * wd..(iy as usize + 1) * wd];
                    for (ox, d) in drow.iter_mut().enumerate() {
                        let ix = (ox * p.strides[1] + kx * p.dilations[1]) as isize - p.pads[1] as isize;
                        *d = if ix < 0 || ix >= wd as isize { 0.0 } else { srow[ix as usize] };
                    }
                }
            }
        };
        par.for_ranges(q, 1, par.parts(q, q * pix * 8), fill);

        let gm = Gemm {
            m: mg,
            k: q,
            n: pix,
            w: &w[gi * mg * q..(gi + 1) * mg * q],
            x: &cols,
            bias: b.map(|b| &b[gi * mg..(gi + 1) * mg]),
        };
        gemm(par, &gm, &mut out[gi * mg * pix..(gi + 1) * mg * pix]);
    }
}

#[allow(clippy::too_many_arguments)]
fn depthwise(
    par: &Par,
    x: &[f32],
    c: usize,
    h: usize,
    wd: usize,
    w: &[f32],
    m: usize,
    kh: usize,
    kw: usize,
    b: Option<&[f32]>,
    p: &ConvParams,
    oh: usize,
    ow: usize,
    out: &mut [f32],
) {
    let mult = m / c;
    let plane = oh * ow;
    let work = m * plane * kh * kw;
    let d = Depthwise { x, h, wd, w, mult, kh, kw, b, p, oh, ow };

    par.for_row_chunks(out, plane, mult, work, |oc0, chunk| {
        #[cfg(target_arch = "x86_64")]
        {
            if use_fma() {
                // SAFETY: AVX2 and FMA were detected at run time.
                unsafe { d.run_fma(oc0, chunk) };
                return;
            }
        }
        d.run::<{ cfg!(target_arch = "aarch64") }>(oc0, chunk);
    });
}

struct Depthwise<'a> {
    x: &'a [f32],
    h: usize,
    wd: usize,
    w: &'a [f32],
    mult: usize,
    kh: usize,
    kw: usize,
    b: Option<&'a [f32]>,
    p: &'a ConvParams,
    oh: usize,
    ow: usize,
}

impl Depthwise<'_> {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn run_fma(&self, oc0: usize, out: &mut [f32]) {
        self.run::<true>(oc0, out)
    }

    /// Output channels `oc0..` filling `out`.
    #[inline(always)]
    fn run<const FMA: bool>(&self, oc0: usize, out: &mut [f32]) {
        let (h, wd, kh, kw, oh, ow) = (self.h, self.wd, self.kh, self.kw, self.oh, self.ow);
        let [pt, pl, pb, pr] = self.p.pads;
        let [sh, sw] = self.p.strides;
        let hp = h + pt + pb;
        let wp = wd + pl + pr;
        // The border stays zero; only the interior is rewritten per plane.
        let mut pad = vec![0.0f32; hp * wp];
        let mut padded_c = usize::MAX;

        for (i, oplane) in out.chunks_exact_mut(oh * ow).enumerate() {
            let oc = oc0 + i;
            let c = oc / self.mult;
            if c != padded_c {
                let src = &self.x[c * h * wd..(c + 1) * h * wd];
                for y in 0..h {
                    pad[(y + pt) * wp + pl..(y + pt) * wp + pl + wd]
                        .copy_from_slice(&src[y * wd..(y + 1) * wd]);
                }
                padded_c = c;
            }

            let bias = self.b.map_or(0.0, |b| b[oc]);
            let wk = &self.w[oc * kh * kw..(oc + 1) * kh * kw];

            for (oy, orow) in oplane.chunks_exact_mut(ow).enumerate() {
                orow.fill(bias);
                for ky in 0..kh {
                    let prow = &pad[(oy * sh + ky) * wp..(oy * sh + ky + 1) * wp];
                    for kx in 0..kw {
                        let wv = wk[ky * kw + kx];
                        if sw == 1 {
                            for (o, &s) in orow.iter_mut().zip(&prow[kx..kx + ow]) {
                                *o = madd::<FMA>(*o, wv, s);
                            }
                        } else {
                            for (ox, o) in orow.iter_mut().enumerate() {
                                *o = madd::<FMA>(*o, wv, prow[ox * sw + kx]);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct definition of a grouped, padded, strided, dilated Conv.
    fn reference(
        x: &[f32], xs: &[usize], w: &[f32], ws: &[usize], b: &[f32], p: &ConvParams,
    ) -> Vec<f32> {
        let (c, h, wd) = (xs[1], xs[2], xs[3]);
        let (m, cg, kh, kw) = (ws[0], ws[1], ws[2], ws[3]);
        let (oh, ow) = p.out_size(h, wd, kh, kw).unwrap();
        let mg = m / p.group;
        let mut out = vec![0.0; m * oh * ow];
        for oc in 0..m {
            let g = oc / mg;
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut s = b[oc] as f64;
                    for ic in 0..cg {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = (oy * p.strides[0] + ky * p.dilations[0]) as isize - p.pads[0] as isize;
                                let ix = (ox * p.strides[1] + kx * p.dilations[1]) as isize - p.pads[1] as isize;
                                if iy < 0 || ix < 0 || iy >= h as isize || ix >= wd as isize {
                                    continue;
                                }
                                let xv = x[((g * cg + ic) * h + iy as usize) * wd + ix as usize];
                                let wv = w[((oc * cg + ic) * kh + ky) * kw + kx];
                                s += xv as f64 * wv as f64;
                            }
                        }
                    }
                    out[(oc * oh + oy) * ow + ox] = s as f32;
                }
            }
        }
        let _ = c;
        out
    }

    #[test]
    fn all_paths_match_reference() {
        let mut seed = 7u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1 << 24) as f32 - 0.5
        };
        // (C, H, W, M, group, K, stride, pad, dilation)
        let cases = [
            (8, 9, 11, 12, 1, 1, 1, 0, 1),  // pointwise
            (6, 10, 9, 6, 6, 3, 1, 1, 1),   // depthwise 3x3
            (6, 13, 12, 6, 6, 7, 1, 3, 1),  // depthwise 7x7
            (4, 12, 12, 8, 4, 3, 2, 1, 1),  // multiplier 2, stride 2
            (3, 16, 15, 8, 1, 3, 2, 1, 1),  // dense stem
            (8, 9, 9, 4, 2, 3, 1, 2, 2),    // grouped, dilated
            (5, 7, 7, 3, 1, 1, 2, 1, 1),    // strided 1x1 with padding
        ];
        for &(c, h, wd, m, g, k, s, pd, dl) in &cases {
            let xs = [1, c, h, wd];
            let ws = [m, c / g, k, k];
            let x: Vec<f32> = (0..c * h * wd).map(|_| rnd()).collect();
            let w: Vec<f32> = (0..ws.iter().product::<usize>()).map(|_| rnd()).collect();
            let b: Vec<f32> = (0..m).map(|_| rnd()).collect();
            let p = ConvParams { group: g, pads: [pd; 4], strides: [s, s], dilations: [dl, dl] };
            let want = reference(&x, &xs, &w, &ws, &b, &p);
            let (got, _) = conv2d(&Par::new(2), &x, &xs, &w, &ws, Some(&b), &p).unwrap();
            assert_eq!(got.len(), want.len());
            for (a, e) in got.iter().zip(&want) {
                assert!((a - e).abs() < 1e-4, "case {:?}: {} vs {}", (c, h, wd, m, g, k, s, pd, dl), a, e);
            }
        }
    }
}
