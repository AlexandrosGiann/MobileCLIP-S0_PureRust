//! Image preprocessing, as in the browser page: center square crop, resize
//! to 256 x 256, RGB, divide by 255, NCHW float32. No mean/std
//! normalization.

use std::path::Path;

use image::imageops::{self, FilterType};
use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage};

use crate::error::Result;
use crate::IMAGE_SIZE;

/// Decodes an image file (JPEG, PNG, WebP, BMP, GIF), applies its EXIF
/// orientation like a browser does, and preprocesses it.
pub fn load_image(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let img = decode(path.as_ref())?;
    Ok(preprocess(&img))
}

pub fn decode(path: &Path) -> Result<DynamicImage> {
    let mut decoder = ImageReader::open(path)
        .map_err(|e| crate::Error::new(format!("{}: {}", path.display(), e)))?
        .with_guessed_format()?
        .into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut img = DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    Ok(img)
}

/// Center crop to a square, resize to 256 x 256 with an antialiasing
/// triangle filter, and lay out as `[3][256][256]` in `[0, 1]`.
pub fn preprocess(img: &DynamicImage) -> Vec<f32> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let side = w.min(h);
    let crop = imageops::crop_imm(&rgb, (w - side) / 2, (h - side) / 2, side, side).to_image();
    let s = IMAGE_SIZE as u32;
    let resized: RgbImage = if side == s {
        crop
    } else {
        imageops::resize(&crop, s, s, FilterType::Triangle)
    };
    to_nchw(&resized)
}

/// `[H][W][RGB]` bytes to `[3][H][W]` floats in `[0, 1]`.
pub fn to_nchw(img: &RgbImage) -> Vec<f32> {
    let plane = (img.width() * img.height()) as usize;
    let mut out = vec![0.0f32; 3 * plane];
    for (i, px) in img.pixels().enumerate() {
        for c in 0..3 {
            out[c * plane + i] = px.0[c] as f32 / 255.0;
        }
    }
    out
}
