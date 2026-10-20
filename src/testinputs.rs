//! The deterministic inputs of `tools/validate_v3.js`, reproduced bit for
//! bit (including JavaScript's double-precision LCG arithmetic), so the
//! Rust embeddings can be compared with the JavaScript executors'.

use crate::{IMAGE_SIZE, INPUT_LEN};

/// Pseudo-random noise blended with a diagonal gradient, in [0, 1].
pub fn noisy_input() -> Vec<f32> {
    let mut x = vec![0.0f32; INPUT_LEN];
    let mut s: i64 = 12345;
    for (i, v) in x.iter_mut().enumerate() {
        // JS: s = (s * 1103515245 + 12345) & 0x7fffffff, evaluated in
        // doubles, then truncated to int32.
        let prod = s as f64 * 1103515245.0 + 12345.0;
        s = (prod as i64) & 0x7fff_ffff;
        let p = i % 65536;
        let noise = ((s >> 8) & 0xffff) as f64 / 65535.0;
        let grad = ((p % 256) + (p >> 8)) as f64 / 510.0;
        *v = (0.5 * noise + 0.5 * grad) as f32;
    }
    x
}

/// Smooth picture with flat regions, quantized to n / 255 like a photo.
pub fn flat_input() -> Vec<f32> {
    let plane = IMAGE_SIZE * IMAGE_SIZE;
    let mut x = vec![0.0f32; INPUT_LEN];
    let round = |v: f64| (v + 0.5).floor();
    for py in 0..IMAGE_SIZE {
        for px in 0..IMAGE_SIZE {
            let i = py * IMAGE_SIZE + px;
            let t = (px + py) as f64 / 510.0;
            let inside = py > 72
                && py < 240
                && ((px as f64) - 128.0).abs() < (py as f64 - 72.0) * 0.57;
            let dark = 16.0 / 255.0;
            x[i] = if inside { dark } else { round(32.0 + 208.0 * t) / 255.0 } as f32;
            x[plane + i] = if inside { dark } else { round(48.0 + 176.0 * t * t) / 255.0 } as f32;
            x[2 * plane + i] = if inside { dark } else { round(96.0 - 80.0 * t) / 255.0 } as f32;
        }
    }
    x
}
