//! Tensors with shared, copy-on-write buffers.
//!
//! A buffer is an `Arc<Vec<_>>`. Shape-only ops (Reshape, Squeeze,
//! Unsqueeze) share the buffer; an op that owns the only reference to its
//! input may overwrite it in place (`Arc::unwrap_or_clone`).

use std::sync::Arc;

use crate::error::{bail, Result};

#[derive(Clone, Debug)]
pub enum Data {
    F32(Arc<Vec<f32>>),
    I32(Arc<Vec<i32>>),
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Data,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn f32(data: Vec<f32>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(data.len(), numel(&shape));
        Tensor {
            data: Data::F32(Arc::new(data)),
            shape,
        }
    }

    pub fn i32(data: Vec<i32>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(data.len(), numel(&shape));
        Tensor {
            data: Data::I32(Arc::new(data)),
            shape,
        }
    }

    pub fn len(&self) -> usize {
        match &self.data {
            Data::F32(v) => v.len(),
            Data::I32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_f32(&self) -> bool {
        matches!(self.data, Data::F32(_))
    }

    pub fn dtype(&self) -> &'static str {
        match self.data {
            Data::F32(_) => "f32",
            Data::I32(_) => "i32",
        }
    }

    pub fn as_f32(&self) -> Result<&[f32]> {
        match &self.data {
            Data::F32(v) => Ok(v),
            Data::I32(_) => bail!("expected f32 tensor, got i32"),
        }
    }

    /// The f32 buffer, taken over if this is its only reference and
    /// copied otherwise. i32 data is converted.
    pub fn into_f32_vec(self) -> Vec<f32> {
        match self.data {
            Data::F32(v) => Arc::unwrap_or_clone(v),
            Data::I32(v) => v.iter().map(|&x| x as f32).collect(),
        }
    }

    /// True if the f32 buffer can be overwritten without copying.
    pub fn is_unique(&self) -> bool {
        match &self.data {
            Data::F32(v) => Arc::strong_count(v) == 1,
            Data::I32(v) => Arc::strong_count(v) == 1,
        }
    }

    /// Integer view of the values (shape inputs such as Reshape's target).
    pub fn to_i64_vec(&self) -> Vec<i64> {
        match &self.data {
            Data::F32(v) => v.iter().map(|&x| x as i64).collect(),
            Data::I32(v) => v.iter().map(|&x| x as i64).collect(),
        }
    }

    pub fn with_shape(&self, shape: Vec<usize>) -> Self {
        Tensor {
            data: self.data.clone(),
            shape,
        }
    }
}

pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![0; shape.len()];
    let mut p = 1;
    for i in (0..shape.len()).rev() {
        s[i] = p;
        p *= shape[i];
    }
    s
}

pub fn normalize_axis(axis: i64, rank: usize) -> Result<usize> {
    let a = if axis < 0 { axis + rank as i64 } else { axis };
    if a < 0 || a >= rank as i64 {
        bail!("axis {} out of range for rank {}", axis, rank);
    }
    Ok(a as usize)
}

/// Numpy-style broadcast of two shapes.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![0; rank];
    for i in 0..rank {
        let av = if i + a.len() >= rank { a[i + a.len() - rank] } else { 1 };
        let bv = if i + b.len() >= rank { b[i + b.len() - rank] } else { 1 };
        if av != bv && av != 1 && bv != 1 {
            bail!("broadcast mismatch {:?} vs {:?}", a, b);
        }
        out[i] = av.max(bv);
    }
    Ok(out)
}

/// `shape` left-padded with ones to `rank`.
pub fn pad_shape(shape: &[usize], rank: usize) -> Vec<usize> {
    let mut v = vec![1; rank - shape.len()];
    v.extend_from_slice(shape);
    v
}
