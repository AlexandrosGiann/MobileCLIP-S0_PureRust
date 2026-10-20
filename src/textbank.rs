//! Local text embedding bank and cosine top-k matching.

use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::error::{bail, Error, Result};

pub struct TextBank {
    pub labels: Vec<String>,
    pub dim: usize,
    /// `[labels][dim]`, each row L2-normalized.
    pub embeddings: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Match {
    pub index: usize,
    pub label: String,
    /// Cosine similarity, not a calibrated probability.
    pub score: f32,
}

impl TextBank {
    /// Loads `text-bank/labels.json` and `text-bank/embeddings.f32` from the
    /// repository root.
    pub fn load(root: &Path) -> Result<TextBank> {
        let dir = root.join("text-bank");
        let meta: Value = serde_json::from_str(&fs::read_to_string(dir.join("labels.json"))?)?;
        let labels: Vec<String> = meta
            .get("labels")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::new("labels.json: missing 'labels'"))?
            .iter()
            .map(|v| v.as_str().map(str::to_owned))
            .collect::<Option<_>>()
            .ok_or_else(|| Error::new("labels.json: labels must be strings"))?;
        let dim = meta.get("dim").and_then(Value::as_u64).unwrap_or(512) as usize;

        let bytes = fs::read(dir.join("embeddings.f32"))?;
        let embeddings: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        TextBank::new(labels, dim, embeddings)
    }

    pub fn new(labels: Vec<String>, dim: usize, mut embeddings: Vec<f32>) -> Result<TextBank> {
        if labels.is_empty() || dim == 0 {
            bail!("empty text bank");
        }
        if embeddings.len() != labels.len() * dim {
            bail!(
                "text bank size mismatch: {} floats for {} labels x {}",
                embeddings.len(),
                labels.len(),
                dim
            );
        }
        for (i, row) in embeddings.chunks_exact_mut(dim).enumerate() {
            let norm = l2(row);
            if !(norm > 0.0) || !norm.is_finite() {
                bail!("invalid text embedding norm at label {}", i);
            }
            for v in row.iter_mut() {
                *v = (*v as f64 / norm) as f32;
            }
        }
        Ok(TextBank { labels, dim, embeddings })
    }

    /// Cosine similarity of `image` to every label, best `k` first.
    pub fn top_k(&self, image: &[f32], k: usize) -> Result<Vec<Match>> {
        let scores = self.scores(image)?;
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        Ok(order
            .into_iter()
            .take(k)
            .map(|i| Match { index: i, label: self.labels[i].clone(), score: scores[i] })
            .collect())
    }

    pub fn scores(&self, image: &[f32]) -> Result<Vec<f32>> {
        if image.len() != self.dim {
            bail!("image embedding has {} values, text bank {}", image.len(), self.dim);
        }
        let norm = l2(image);
        if !(norm > 0.0) || !norm.is_finite() {
            bail!("invalid image embedding norm");
        }
        let img: Vec<f32> = image.iter().map(|&v| (v as f64 / norm) as f32).collect();
        Ok(self
            .embeddings
            .chunks_exact(self.dim)
            .map(|row| row.iter().zip(&img).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32)
            .collect())
    }
}

fn l2(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt()
}
