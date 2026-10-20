//! Writes model inputs as raw little-endian f32 files, for cross-checking
//! against the JavaScript executors.
//!
//!   cargo run --release --example dump_inputs -- <out_dir> [image...]
//!
//! Always writes the two validation inputs; each image is preprocessed and
//! written as `<out_dir>/<file stem>.f32`.
use std::path::Path;

fn write(path: &Path, v: &[f32]) {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, b).unwrap();
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().expect("output dir"));
    write(&dir.join("rust_input_noise.f32"), &mobileclip_s0::testinputs::noisy_input());
    write(&dir.join("rust_input_flat.f32"), &mobileclip_s0::testinputs::flat_input());
    for img in args {
        let p = Path::new(&img);
        let v = mobileclip_s0::preprocess::load_image(p).unwrap();
        write(&dir.join(format!("{}.f32", p.file_stem().unwrap().to_string_lossy())), &v);
    }
}
