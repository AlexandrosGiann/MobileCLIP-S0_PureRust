//! MobileCLIP-S0 FP32 in pure Rust.
//!
//! Runs the compiled MobileCLIP-S0 vision graph (`compiled-fp32-v1/plan.json`
//! plus the FP32 weights) with hand-written CPU kernels, and matches the
//! resulting image embedding against the local text embedding bank.
//!
//! No ONNX Runtime, no BLAS, no C libraries: only the Rust standard library,
//! `serde_json` for the plan and the pure-Rust `image` decoders.
//!
//! ```no_run
//! use mobileclip_s0::{MobileClip, TextBank, preprocess};
//! let root = std::path::Path::new("..");
//! let model = MobileClip::load(root, 4).unwrap();
//! let bank = TextBank::load(root).unwrap();
//! let pixels = preprocess::load_image("volcano.jpg").unwrap();
//! let emb = model.embed(&pixels).unwrap();
//! for m in bank.top_k(&emb, 5).unwrap() {
//!     println!("{:.6}  {}", m.score, m.label);
//! }
//! ```

pub mod error;
pub mod executor;
pub mod kernels;
pub mod par;
pub mod plan;
pub mod preprocess;
pub mod tensor;
pub mod testinputs;
pub mod textbank;

use std::path::Path;

pub use error::{Error, Result};
pub use executor::{Executor, Profile};
pub use textbank::{Match, TextBank};

/// Side length of the square model input.
pub const IMAGE_SIZE: usize = 256;
/// Number of floats in one `[1, 3, 256, 256]` input.
pub const INPUT_LEN: usize = 3 * IMAGE_SIZE * IMAGE_SIZE;
/// Image embedding width.
pub const EMBED_DIM: usize = 512;

/// The vision encoder: compiled plan + weights + executor.
pub struct MobileClip {
    executor: Executor,
}

impl MobileClip {
    /// Loads `compiled-fp32-v1/` from the repository root `root`.
    /// `threads` = 0 picks the number of available CPU cores.
    pub fn load(root: &Path, threads: usize) -> Result<Self> {
        let plan = plan::Plan::load(&root.join("compiled-fp32-v1"))?;
        Ok(MobileClip {
            executor: Executor::new(plan, par::Par::new(threads))?,
        })
    }

    pub fn executor(&self) -> &Executor {
        &self.executor
    }

    /// Image embedding (`image_embeds`, not normalized) for one
    /// preprocessed `[1, 3, 256, 256]` NCHW input in `[0, 1]`.
    pub fn embed(&self, pixels: &[f32]) -> Result<Vec<f32>> {
        self.embed_profiled(pixels, None)
    }

    pub fn embed_profiled(
        &self,
        pixels: &[f32],
        profile: Option<&mut Profile>,
    ) -> Result<Vec<f32>> {
        if pixels.len() != INPUT_LEN {
            return Err(Error::new(format!(
                "input must have {} floats, got {}",
                INPUT_LEN,
                pixels.len()
            )));
        }
        let out = self
            .executor
            .run(pixels.to_vec(), &[1, 3, IMAGE_SIZE, IMAGE_SIZE], profile)?;
        if out.len() != EMBED_DIM {
            return Err(Error::new(format!(
                "unexpected embedding length {}",
                out.len()
            )));
        }
        Ok(out)
    }
}
