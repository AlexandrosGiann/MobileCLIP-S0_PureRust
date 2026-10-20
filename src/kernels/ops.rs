//! Elementwise, reduction and shape operators on [`Tensor`]s.
//!
//! Operators that take their input by value overwrite it in place when
//! they hold the only reference to its buffer.

use std::sync::Arc;

use super::gemm::{gemm, Gemm};
use super::math;
use crate::error::{bail, Result};
use crate::par::Par;
use crate::tensor::{broadcast_shape, normalize_axis, numel, pad_shape, strides, Data, Tensor};

/// Applies a generic expression to the buffer of either dtype.
macro_rules! per_dtype {
    ($t:expr, |$v:ident| $body:expr) => {
        match &$t.data {
            Data::F32($v) => Data::F32(Arc::new($body)),
            Data::I32($v) => Data::I32(Arc::new($body)),
        }
    };
}

// ---------------------------------------------------------------- binary

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinOp {
    #[inline(always)]
    fn f32(self, a: f32, b: f32) -> f32 {
        match self {
            BinOp::Add => a + b,
            BinOp::Sub => a - b,
            BinOp::Mul => a * b,
            BinOp::Div => a / b,
        }
    }

    #[inline(always)]
    fn i32(self, a: i32, b: i32) -> i32 {
        match self {
            BinOp::Add => a.wrapping_add(b),
            BinOp::Sub => a.wrapping_sub(b),
            BinOp::Mul => a.wrapping_mul(b),
            BinOp::Div => {
                if b == 0 {
                    0
                } else {
                    a.wrapping_div(b)
                }
            }
        }
    }
}

/// Per-dimension strides of `shape` broadcast to `out` (0 on broadcast dims).
fn broadcast_strides(shape: &[usize], out: &[usize]) -> Vec<usize> {
    let p = pad_shape(shape, out.len());
    let s = strides(&p);
    p.iter()
        .zip(s)
        .zip(out)
        .map(|((&d, s), &o)| if d == 1 && o != 1 { 0 } else { s })
        .collect()
}

/// Visits the output in rows of its last dimension, giving the offsets of
/// `a` and `b` at the start of each row.
fn for_each_row(
    out_shape: &[usize],
    sa: &[usize],
    sb: &[usize],
    mut f: impl FnMut(usize, usize, usize),
) {
    let rank = out_shape.len();
    let last = out_shape[rank - 1];
    let rows = numel(out_shape) / last.max(1);
    let mut coord = vec![0usize; rank];
    let (mut ao, mut bo) = (0usize, 0usize);
    for r in 0..rows {
        f(r * last, ao, bo);
        for d in (0..rank - 1).rev() {
            coord[d] += 1;
            ao += sa[d];
            bo += sb[d];
            if coord[d] < out_shape[d] {
                break;
            }
            ao -= sa[d] * out_shape[d];
            bo -= sb[d] * out_shape[d];
            coord[d] = 0;
        }
    }
}

/// `out = a op b` with broadcasting. `a == None` means `a` is `out` itself
/// (it already has the output shape).
fn broadcast_into<T: Copy>(
    a: Option<(&[T], &[usize])>,
    b: (&[T], &[usize]),
    out: &mut [T],
    out_shape: &[usize],
    f: impl Fn(T, T) -> T,
) {
    let shape1 = [1usize];
    let os = if out_shape.is_empty() { &shape1[..] } else { out_shape };
    let a_shape = a.map_or(os, |a| a.1);
    let sa = broadcast_strides(a_shape, os);
    let sb = broadcast_strides(b.1, os);
    let last = os[os.len() - 1];
    let (la, lb) = (sa[os.len() - 1], sb[os.len() - 1]);
    let bd = b.0;

    match a {
        None => for_each_row(os, &sa, &sb, |o, _, bo| {
            let row = &mut out[o..o + last];
            if lb == 1 {
                for (v, &y) in row.iter_mut().zip(&bd[bo..bo + last]) {
                    *v = f(*v, y);
                }
            } else {
                let y = bd[bo];
                for v in row.iter_mut() {
                    *v = f(*v, y);
                }
            }
        }),
        Some((ad, _)) => for_each_row(os, &sa, &sb, |o, ao, bo| {
            let row = &mut out[o..o + last];
            for (j, v) in row.iter_mut().enumerate() {
                *v = f(ad[ao + j * la], bd[bo + j * lb]);
            }
        }),
    }
}

/// `a op b`, numpy broadcasting. Two i32 tensors give i32, anything else f32.
pub fn binary(a: Tensor, b: Tensor, op: BinOp) -> Result<Tensor> {
    let out_shape = broadcast_shape(&a.shape, &b.shape)?;

    if let (Data::I32(x), Data::I32(y)) = (&a.data, &b.data) {
        let mut out = vec![0i32; numel(&out_shape)];
        broadcast_into(Some((&x[..], &a.shape[..])), (&y[..], &b.shape[..]), &mut out, &out_shape, |p, q| op.i32(p, q));
        return Ok(Tensor::i32(out, out_shape));
    }

    let a_full = a.shape == out_shape;
    let b_full = b.shape == out_shape;

    // Write into a's buffer.
    if a_full && (a.is_unique() || !(b_full && b.is_unique())) {
        let bs = b.shape.clone();
        let bv = f32_values(&b);
        let mut out = a.into_f32_vec();
        broadcast_into(None, (&bv[..], &bs[..]), &mut out, &out_shape, |p, q| op.f32(p, q));
        return Ok(Tensor::f32(out, out_shape));
    }

    // Write into b's buffer: out = a op out.
    if b_full {
        let as_ = a.shape.clone();
        let av = f32_values(&a);
        let mut out = b.into_f32_vec();
        broadcast_into(None, (&av[..], &as_[..]), &mut out, &out_shape, |q, p| op.f32(p, q));
        return Ok(Tensor::f32(out, out_shape));
    }

    let av = f32_values(&a);
    let bv = f32_values(&b);
    let mut out = vec![0.0f32; numel(&out_shape)];
    broadcast_into(Some((&av[..], &a.shape[..])), (&bv[..], &b.shape[..]), &mut out, &out_shape, |p, q| op.f32(p, q));
    Ok(Tensor::f32(out, out_shape))
}

fn f32_values(t: &Tensor) -> std::borrow::Cow<'_, [f32]> {
    match &t.data {
        Data::F32(v) => std::borrow::Cow::Borrowed(&v[..]),
        Data::I32(v) => std::borrow::Cow::Owned(v.iter().map(|&x| x as f32).collect()),
    }
}

// ----------------------------------------------------------------- unary

pub fn unary(x: Tensor, f: impl Fn(f32) -> f32) -> Tensor {
    let shape = x.shape.clone();
    let mut v = x.into_f32_vec();
    for e in v.iter_mut() {
        *e = f(*e);
    }
    Tensor::f32(v, shape)
}

pub fn relu(x: Tensor) -> Tensor {
    unary(x, |v| if v > 0.0 { v } else { 0.0 })
}

pub fn sigmoid(x: Tensor) -> Tensor {
    unary(x, math::sigmoid)
}

pub fn erf(x: Tensor) -> Tensor {
    unary(x, |v| math::erf(v as f64) as f32)
}

/// Fused `x * (erf(x / div) + add) * mul`.
pub fn gelu(par: &Par, x: Tensor, div: f32, add: f32, mul: f32) -> Tensor {
    let shape = x.shape.clone();
    let mut v = x.into_f32_vec();
    const CHUNK: usize = 4096;
    let work = v.len() * 8;
    par.for_row_chunks(&mut v, CHUNK, 1, work, |_, c| {
        math::gelu_in_place(c, div, add, mul)
    });
    Tensor::f32(v, shape)
}

// ---------------------------------------------------------------- matmul

/// numpy `matmul` for rank >= 2 (rank-1 operands are promoted).
pub fn matmul(par: &Par, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let mut ash = a.shape.clone();
    let mut bsh = b.shape.clone();
    let a_vec = ash.len() == 1;
    let b_vec = bsh.len() == 1;
    if a_vec {
        ash.insert(0, 1);
    }
    if b_vec {
        bsh.push(1);
    }
    if ash.len() < 2 || bsh.len() < 2 {
        bail!("MatMul on scalars");
    }
    let (ar, br) = (ash.len(), bsh.len());
    let (m, k) = (ash[ar - 2], ash[ar - 1]);
    let (k2, n) = (bsh[br - 2], bsh[br - 1]);
    if k != k2 {
        bail!("MatMul K mismatch {:?} x {:?}", a.shape, b.shape);
    }

    let batch_shape = broadcast_shape(&ash[..ar - 2], &bsh[..br - 2])?;
    let batches = numel(&batch_shape);
    let sa = broadcast_strides(&ash[..ar - 2], &batch_shape);
    let sb = broadcast_strides(&bsh[..br - 2], &batch_shape);
    let bstr = strides(&batch_shape);
    let offsets = |bi: usize| {
        let (mut ao, mut bo, mut t) = (0, 0, bi);
        for d in 0..batch_shape.len() {
            let c = t / bstr[d];
            t -= c * bstr[d];
            ao += c * sa[d];
            bo += c * sb[d];
        }
        (ao * m * k, bo * k * n)
    };

    let av = a.as_f32()?;
    let bv = b.as_f32()?;
    let mut out = vec![0.0f32; batches * m * n];

    if batches == 1 {
        let g = Gemm { m, k, n, w: av, x: bv, bias: None };
        gemm(par, &g, &mut out);
    } else {
        // Many small products: one thread per group of batches.
        par.for_row_chunks(&mut out, m * n, 1, batches * m * k * n, |b0, chunk| {
            for (i, o) in chunk.chunks_exact_mut(m * n).enumerate() {
                let (ao, bo) = offsets(b0 + i);
                let g = Gemm {
                    m,
                    k,
                    n,
                    w: &av[ao..ao + m * k],
                    x: &bv[bo..bo + k * n],
                    bias: None,
                };
                gemm(&Par::serial(), &g, o);
            }
        });
    }

    let mut shape = batch_shape;
    if !a_vec {
        shape.push(m);
    }
    if !b_vec {
        shape.push(n);
    }
    Ok(Tensor::f32(out, shape))
}

// ----------------------------------------------------------- shape ops

pub fn shape_of(x: &Tensor) -> Tensor {
    Tensor::i32(x.shape.iter().map(|&d| d as i32).collect(), vec![x.shape.len()])
}

pub fn reshape(x: &Tensor, spec: &[i64], allowzero: bool) -> Result<Tensor> {
    let total = x.len();
    let mut shape = Vec::with_capacity(spec.len());
    let mut infer = None;
    let mut known = 1usize;
    for (i, &v) in spec.iter().enumerate() {
        let d = if v == 0 && !allowzero {
            *x.shape.get(i).ok_or_else(|| crate::Error::new("Reshape 0 beyond input rank"))? as i64
        } else {
            v
        };
        if d == -1 {
            if infer.is_some() {
                bail!("Reshape has several -1");
            }
            infer = Some(i);
            shape.push(0);
        } else if d < 0 {
            bail!("Reshape dimension {}", d);
        } else {
            shape.push(d as usize);
            known *= d as usize;
        }
    }
    if let Some(i) = infer {
        if known == 0 || total % known != 0 {
            bail!("Reshape cannot infer {:?} -> {:?}", x.shape, spec);
        }
        shape[i] = total / known;
    }
    if numel(&shape) != total {
        bail!("Reshape size mismatch {:?} -> {:?}", x.shape, shape);
    }
    Ok(x.with_shape(shape))
}

pub fn unsqueeze(x: &Tensor, axes: &[i64]) -> Result<Tensor> {
    let rank = x.shape.len() + axes.len();
    let mut mark = vec![false; rank];
    for &a in axes {
        mark[normalize_axis(a, rank)?] = true;
    }
    let mut src = x.shape.iter();
    let shape = mark
        .iter()
        .map(|&m| if m { 1 } else { *src.next().unwrap() })
        .collect();
    Ok(x.with_shape(shape))
}

pub fn squeeze(x: &Tensor, axes: &[i64]) -> Result<Tensor> {
    let rank = x.shape.len();
    let shape = if axes.is_empty() {
        x.shape.iter().copied().filter(|&d| d != 1).collect()
    } else {
        let mut remove = vec![false; rank];
        for &a in axes {
            let a = normalize_axis(a, rank)?;
            if x.shape[a] != 1 {
                bail!("Squeeze axis {} has size {}", a, x.shape[a]);
            }
            remove[a] = true;
        }
        x.shape
            .iter()
            .zip(&remove)
            .filter(|(_, &r)| !r)
            .map(|(&d, _)| d)
            .collect()
    };
    Ok(x.with_shape(shape))
}

fn transpose_impl<T: Copy + Default>(x: &[T], shape: &[usize], perm: &[usize]) -> Vec<T> {
    let rank = shape.len();
    let mut out = vec![T::default(); x.len()];
    if x.is_empty() {
        return out;
    }
    if rank == 0 {
        out.copy_from_slice(x);
        return out;
    }
    let ins = strides(shape);
    let os: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let step: Vec<usize> = perm.iter().map(|&p| ins[p]).collect();
    let last = rank - 1;
    let (n_last, st_last) = (os[last], step[last]);
    let mut coord = vec![0usize; rank];
    let mut src = 0usize;
    let mut i = 0usize;
    while i < out.len() {
        let mut s = src;
        for o in &mut out[i..i + n_last] {
            *o = x[s];
            s += st_last;
        }
        i += n_last;
        for d in (0..last).rev() {
            coord[d] += 1;
            src += step[d];
            if coord[d] < os[d] {
                break;
            }
            src -= step[d] * os[d];
            coord[d] = 0;
        }
    }
    out
}

pub fn transpose(x: &Tensor, perm: Option<&[i64]>) -> Result<Tensor> {
    let rank = x.shape.len();
    let perm: Vec<usize> = match perm {
        Some(p) if !p.is_empty() => p
            .iter()
            .map(|&a| normalize_axis(a, rank))
            .collect::<Result<_>>()?,
        _ => (0..rank).rev().collect(),
    };
    if perm.len() != rank {
        bail!("Transpose perm {:?} for rank {}", perm, rank);
    }
    let shape: Vec<usize> = perm.iter().map(|&p| x.shape[p]).collect();
    let data = per_dtype!(x, |v| transpose_impl(v, &x.shape, &perm));
    Ok(Tensor { data, shape })
}

/// Concatenation along `axis`. Like the JavaScript reference executor,
/// inputs whose rank differs from the first one (shape vectors built from
/// rank-1 constants) are accepted as long as the element counts add up.
pub fn concat(xs: &[Tensor], axis: i64) -> Result<Tensor> {
    let first = xs.first().ok_or_else(|| crate::Error::new("Concat without inputs"))?;
    let rank = first.shape.len();
    let ax = normalize_axis(axis, rank)?;
    let outer: usize = first.shape[..ax].iter().product();
    let inner: usize = first.shape[ax + 1..].iter().product();
    let mut shape = first.shape.clone();
    shape[ax] = xs
        .iter()
        .map(|t| t.shape.get(ax).copied().unwrap_or(1))
        .sum();

    fn gather_chunks<T: Copy>(parts: &[(&[T], usize)], outer: usize) -> Vec<T> {
        let mut out = Vec::with_capacity(parts.iter().map(|p| p.0.len()).sum());
        for o in 0..outer {
            for &(d, chunk) in parts {
                out.extend_from_slice(&d[o * chunk..(o + 1) * chunk]);
            }
        }
        out
    }

    let chunk_of = |t: &Tensor| t.shape.get(ax).copied().unwrap_or(1) * inner;
    let data = if xs.iter().all(|t| !t.is_f32()) {
        let parts: Vec<(&[i32], usize)> = xs
            .iter()
            .map(|t| match &t.data {
                Data::I32(v) => (&v[..], chunk_of(t)),
                Data::F32(_) => unreachable!(),
            })
            .collect();
        Data::I32(Arc::new(gather_chunks(&parts, outer)))
    } else {
        let vals: Vec<std::borrow::Cow<[f32]>> = xs.iter().map(f32_values).collect();
        let parts: Vec<(&[f32], usize)> =
            vals.iter().zip(xs).map(|(v, t)| (&v[..], chunk_of(t))).collect();
        Data::F32(Arc::new(gather_chunks(&parts, outer)))
    };

    let t = Tensor { data, shape };
    if t.len() != numel(&t.shape) {
        bail!("Concat sizes do not add up");
    }
    Ok(t)
}

pub fn gather(data: &Tensor, indices: &Tensor, axis: i64) -> Result<Tensor> {
    let rank = data.shape.len();
    let ax = normalize_axis(axis, rank)?;
    let before: usize = data.shape[..ax].iter().product();
    let dim = data.shape[ax];
    let after: usize = data.shape[ax + 1..].iter().product();
    let idx: Vec<usize> = indices
        .to_i64_vec()
        .into_iter()
        .map(|i| {
            let j = if i < 0 { i + dim as i64 } else { i };
            if j < 0 || j >= dim as i64 {
                bail!("Gather index {} out of range {}", i, dim);
            }
            Ok(j as usize)
        })
        .collect::<Result<_>>()?;

    fn go<T: Copy>(v: &[T], before: usize, dim: usize, after: usize, idx: &[usize]) -> Vec<T> {
        let mut out = Vec::with_capacity(before * idx.len() * after);
        for b in 0..before {
            for &i in idx {
                let s = (b * dim + i) * after;
                out.extend_from_slice(&v[s..s + after]);
            }
        }
        out
    }

    let mut shape = data.shape[..ax].to_vec();
    shape.extend_from_slice(&indices.shape);
    shape.extend_from_slice(&data.shape[ax + 1..]);
    let d = per_dtype!(data, |v| go(v, before, dim, after, &idx));
    Ok(Tensor { data: d, shape })
}

pub fn slice(
    data: &Tensor,
    starts: &[i64],
    ends: &[i64],
    axes: Option<&[i64]>,
    steps: Option<&[i64]>,
) -> Result<Tensor> {
    let rank = data.shape.len();
    let mut begin = vec![0i64; rank];
    let mut step = vec![1i64; rank];
    let mut end: Vec<i64> = data.shape.iter().map(|&d| d as i64).collect();

    for i in 0..starts.len() {
        let ax = normalize_axis(axes.map_or(i as i64, |a| a[i]), rank)?;
        let dim = data.shape[ax] as i64;
        let st = steps.map_or(1, |s| s[i]);
        if st == 0 {
            bail!("Slice step 0");
        }
        let mut s = starts[i];
        let mut e = ends[i];
        if s < 0 {
            s += dim;
        }
        if e < 0 {
            e += dim;
        }
        if st > 0 {
            s = s.clamp(0, dim);
            e = e.clamp(0, dim);
        } else {
            s = s.clamp(-1, dim - 1);
            e = e.clamp(-1, dim - 1);
        }
        begin[ax] = s;
        end[ax] = e;
        step[ax] = st;
    }

    let shape: Vec<usize> = (0..rank)
        .map(|d| {
            let span = if step[d] > 0 { end[d] - begin[d] } else { begin[d] - end[d] };
            let st = step[d].abs();
            ((span + st - 1) / st).max(0) as usize
        })
        .collect();

    let ins = strides(&data.shape);
    let os = strides(&shape);
    let n = numel(&shape);
    let src_index = |i: usize| {
        let mut t = i;
        let mut src = 0i64;
        for d in 0..rank {
            let c = (t / os[d]) as i64;
            t %= os[d];
            src += (begin[d] + c * step[d]) * ins[d] as i64;
        }
        src as usize
    };

    let d = per_dtype!(data, |v| (0..n).map(|i| v[src_index(i)]).collect());
    Ok(Tensor { data: d, shape })
}

pub fn split(x: &Tensor, axis: i64, sizes: Option<&[usize]>, outputs: usize) -> Result<Vec<Tensor>> {
    let ax = normalize_axis(axis, x.shape.len())?;
    let dim = x.shape[ax];
    let sizes: Vec<usize> = match sizes {
        Some(s) if !s.is_empty() => s.to_vec(),
        _ => {
            if outputs == 0 || dim % outputs != 0 {
                bail!("Split of {} into {} parts", dim, outputs);
            }
            vec![dim / outputs; outputs]
        }
    };
    if sizes.iter().sum::<usize>() != dim {
        bail!("Split sizes {:?} do not cover {}", sizes, dim);
    }
    let outer: usize = x.shape[..ax].iter().product();
    let inner: usize = x.shape[ax + 1..].iter().product();

    let mut outs = Vec::with_capacity(sizes.len());
    let mut offset = 0;
    for &s in &sizes {
        let mut shape = x.shape.clone();
        shape[ax] = s;
        let data = per_dtype!(x, |v| take_block(v, outer, dim, offset, s, inner));
        outs.push(Tensor { data, shape });
        offset += s;
    }
    Ok(outs)
}

/// Elements `off..off + s` of the split axis from every outer block.
fn take_block<T: Copy>(v: &[T], outer: usize, dim: usize, off: usize, s: usize, inner: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(outer * s * inner);
    for o in 0..outer {
        let b = (o * dim + off) * inner;
        out.extend_from_slice(&v[b..b + s * inner]);
    }
    out
}

// ------------------------------------------------------------ reductions

pub fn reduce_mean(x: &Tensor, axes: &[i64], keepdims: bool) -> Result<Tensor> {
    let rank = x.shape.len();
    let mut reduce = vec![axes.is_empty(); rank];
    for &a in axes {
        reduce[normalize_axis(a, rank)?] = true;
    }
    let v = x.as_f32()?;
    let kept: Vec<usize> = (0..rank).map(|d| if reduce[d] { 1 } else { x.shape[d] }).collect();
    let shape: Vec<usize> = (0..rank)
        .filter(|&d| keepdims || !reduce[d])
        .map(|d| kept[d])
        .collect();
    let n_out = numel(&kept);
    let mut out = vec![0.0f32; n_out];

    let first_reduced = reduce.iter().position(|&r| r).unwrap_or(rank);
    if reduce[first_reduced..].iter().all(|&r| r) {
        // Reduced axes are exactly the trailing ones.
        let inner: usize = x.shape[first_reduced..].iter().product();
        for (o, chunk) in out.iter_mut().zip(v.chunks_exact(inner.max(1))) {
            let s: f32 = chunk.iter().sum();
            *o = s / inner as f32;
        }
    } else {
        let os = broadcast_strides(&kept, &x.shape);
        let count = (v.len() / n_out.max(1)) as f32;
        let ins = strides(&x.shape);
        for (i, &e) in v.iter().enumerate() {
            let mut t = i;
            let mut oi = 0;
            for d in 0..rank {
                let c = t / ins[d];
                t %= ins[d];
                if !reduce[d] {
                    oi += c * os[d];
                }
            }
            out[oi] += e;
        }
        for o in out.iter_mut() {
            *o /= count;
        }
    }
    Ok(Tensor::f32(out, shape))
}

pub fn softmax(x: Tensor, axis: i64) -> Result<Tensor> {
    let ax = normalize_axis(axis, x.shape.len())?;
    let outer: usize = x.shape[..ax].iter().product();
    let dim = x.shape[ax];
    let inner: usize = x.shape[ax + 1..].iter().product();
    let shape = x.shape.clone();
    let mut v = x.into_f32_vec();
    for o in 0..outer {
        for j in 0..inner {
            let base = o * dim * inner + j;
            let idx = |k: usize| base + k * inner;
            let mx = (0..dim).map(|k| v[idx(k)]).fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for k in 0..dim {
                let e = (v[idx(k)] - mx).exp();
                v[idx(k)] = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for k in 0..dim {
                v[idx(k)] *= inv;
            }
        }
    }
    Ok(Tensor::f32(v, shape))
}

// --------------------------------------------------------------- others

pub fn batch_norm(
    x: Tensor,
    scale: &[f32],
    bias: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
) -> Result<Tensor> {
    if x.shape.len() < 2 {
        bail!("BatchNormalization on rank {}", x.shape.len());
    }
    let n = x.shape[0];
    let c = x.shape[1];
    let spatial: usize = x.shape[2..].iter().product();
    if [scale.len(), bias.len(), mean.len(), var.len()].iter().any(|&l| l != c) {
        bail!("BatchNormalization parameter length mismatch");
    }
    let shape = x.shape.clone();
    let mut v = x.into_f32_vec();
    for ci in 0..c {
        let mul = scale[ci] / (var[ci] + eps).sqrt();
        let add = bias[ci] - mean[ci] * mul;
        for ni in 0..n {
            let b = (ni * c + ci) * spatial;
            for e in &mut v[b..b + spatial] {
                *e = *e * mul + add;
            }
        }
    }
    Ok(Tensor::f32(v, shape))
}

pub fn pad_constant(x: Tensor, pads: &[i64], value: f32) -> Result<Tensor> {
    let rank = x.shape.len();
    if pads.len() != 2 * rank {
        bail!("Pad expects {} pads, got {}", 2 * rank, pads.len());
    }
    if pads.iter().any(|&p| p < 0) {
        bail!("negative Pad is not supported");
    }
    if pads.iter().all(|&p| p == 0) {
        return Ok(x);
    }
    let shape: Vec<usize> = (0..rank)
        .map(|d| x.shape[d] + pads[d] as usize + pads[d + rank] as usize)
        .collect();
    let v = x.as_f32()?;
    let mut out = vec![value; numel(&shape)];
    let ins = strides(&x.shape);
    let os = strides(&shape);
    for (i, &e) in v.iter().enumerate() {
        let mut t = i;
        let mut dst = 0;
        for d in 0..rank {
            let c = t / ins[d];
            t %= ins[d];
            dst += (c + pads[d] as usize) * os[d];
        }
        out[dst] = e;
    }
    Ok(Tensor::f32(out, shape))
}

pub fn average_pool(
    x: &Tensor,
    kernel: [usize; 2],
    strides_: [usize; 2],
    pads: [usize; 4],
    count_include_pad: bool,
) -> Result<Tensor> {
    if x.shape.len() != 4 {
        bail!("AveragePool expects NCHW");
    }
    let (n, c, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    let [kh, kw] = kernel;
    let [sh, sw] = strides_;
    let [pt, pl, pb, pr] = pads;
    if h + pt + pb < kh || w + pl + pr < kw {
        bail!("AveragePool kernel larger than input");
    }
    let oh = (h + pt + pb - kh) / sh + 1;
    let ow = (w + pl + pr - kw) / sw + 1;
    let v = x.as_f32()?;
    let mut out = vec![0.0f32; n * c * oh * ow];
    for p in 0..n * c {
        let plane = &v[p * h * w..(p + 1) * h * w];
        for oy in 0..oh {
            for ox in 0..ow {
                let mut s = 0.0f32;
                let mut cnt = 0usize;
                for ky in 0..kh {
                    let iy = (oy * sh + ky) as isize - pt as isize;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for kx in 0..kw {
                        let ix = (ox * sw + kx) as isize - pl as isize;
                        if ix < 0 || ix >= w as isize {
                            continue;
                        }
                        s += plane[iy as usize * w + ix as usize];
                        cnt += 1;
                    }
                }
                let div = if count_include_pad { kh * kw } else { cnt.max(1) };
                out[(p * oh + oy) * ow + ox] = s / div as f32;
            }
        }
    }
    Ok(Tensor::f32(out, vec![n, c, oh, ow]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(v: &[f32], s: &[usize]) -> Tensor {
        Tensor::f32(v.to_vec(), s.to_vec())
    }

    #[test]
    fn broadcast_binary() {
        let a = t(&[1., 2., 3., 4., 5., 6.], &[1, 2, 3]);
        let b = t(&[10., 20.], &[2, 1]);
        let r = binary(a.clone(), b.clone(), BinOp::Add).unwrap();
        assert_eq!(r.shape, vec![1, 2, 3]);
        assert_eq!(r.as_f32().unwrap(), &[11., 12., 13., 24., 25., 26.]);
        let r = binary(b, a, BinOp::Sub).unwrap();
        assert_eq!(r.as_f32().unwrap(), &[9., 8., 7., 16., 15., 14.]);
        let c = t(&[1., 2., 3.], &[3]);
        let d = t(&[1., 10.], &[2, 1]);
        let r = binary(c, d, BinOp::Mul).unwrap();
        assert_eq!(r.shape, vec![2, 3]);
        assert_eq!(r.as_f32().unwrap(), &[1., 2., 3., 10., 20., 30.]);
    }

    #[test]
    fn transpose_5d() {
        let shape = [2, 1, 3, 2, 2];
        let v: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let x = t(&v, &shape);
        let perm = [2i64, 0, 3, 1, 4];
        let y = transpose(&x, Some(&perm)).unwrap();
        assert_eq!(y.shape, vec![3, 2, 2, 1, 2]);
        let ys = y.as_f32().unwrap();
        let is = strides(&shape);
        let os = strides(&y.shape);
        for a in 0..2 { for b in 0..1 { for c in 0..3 { for d in 0..2 { for e in 0..2 {
            let src = a * is[0] + b * is[1] + c * is[2] + d * is[3] + e * is[4];
            let dst = c * os[0] + a * os[1] + d * os[2] + b * os[3] + e * os[4];
            assert_eq!(ys[dst], v[src]);
        }}}}}
    }

    #[test]
    fn matmul_batched_broadcast() {
        let a = t(&(0..12).map(|i| i as f32).collect::<Vec<_>>(), &[2, 2, 3]);
        let b = t(&(0..6).map(|i| i as f32).collect::<Vec<_>>(), &[3, 2]);
        let r = matmul(&Par::serial(), &a, &b).unwrap();
        assert_eq!(r.shape, vec![2, 2, 2]);
        assert_eq!(r.as_f32().unwrap(), &[10., 13., 28., 40., 46., 67., 64., 94.]);
    }

    #[test]
    fn reduce_and_softmax() {
        let x = t(&[1., 2., 3., 4., 5., 6.], &[1, 2, 3]);
        let m = reduce_mean(&x, &[-1], false).unwrap();
        assert_eq!(m.shape, vec![1, 2]);
        assert_eq!(m.as_f32().unwrap(), &[2., 5.]);
        let m = reduce_mean(&x, &[1], true).unwrap();
        assert_eq!(m.shape, vec![1, 1, 3]);
        assert_eq!(m.as_f32().unwrap(), &[2.5, 3.5, 4.5]);
        let s = softmax(x, -1).unwrap();
        let sv = s.as_f32().unwrap();
        assert!((sv[0] + sv[1] + sv[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn slice_and_shape_ops() {
        let x = Tensor::i32(vec![1, 512, 8, 8], vec![4]);
        let s = slice(&x, &[0], &[2], Some(&[0]), None).unwrap();
        assert_eq!(s.to_i64_vec(), vec![1, 512]);
        let c = concat(&[s, Tensor::i32(vec![-1], vec![1])], 0).unwrap();
        assert_eq!(c.to_i64_vec(), vec![1, 512, -1]);
        let y = t(&vec![0.0; 512 * 64], &[1, 512, 8, 8]);
        let r = reshape(&y, &c.to_i64_vec(), false).unwrap();
        assert_eq!(r.shape, vec![1, 512, 64]);
    }
}
