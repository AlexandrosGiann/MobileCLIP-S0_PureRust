# MobileCLIP-S0 FP32 in pure Rust

A native port of the MobileCLIP-S0 FP32 runtime from
[mobileclip-s0-purejs-oldandroid](https://github.com/AlexandrosGiann/mobileclip-s0-purejs-oldandroid).
It runs the same compiled graph (`compiled-fp32-v1/plan.json` plus the
three weight parts) and matches against the same text bank (`text-bank/`).

## Model files

This repository holds only the Rust code. The model files come from the
JavaScript repository above. Clone it and place this repository inside it
as `rust/`, or pass `--root <path to that repository>`:

```bash
git clone https://github.com/AlexandrosGiann/mobileclip-s0-purejs-oldandroid
git clone https://github.com/AlexandrosGiann/MobileCLIP-S0_PureRust mobileclip-s0-purejs-oldandroid/rust
```

The model weights are subject to their own upstream license (see
`MODEL_ATTRIBUTION.md` and `licenses/` there).

- No ONNX Runtime, BLAS, WASM or C libraries. Dependencies: `serde_json`
  and the pure-Rust `image` decoders (JPEG, PNG, WebP, BMP, GIF).
- Generic graph executor: all 22 operators of the plan. Values are dropped
  after their last reader, and buffers are reused in place when the reader
  holds the only reference.
- Packed GEMM (4x16 micro-kernel) for pointwise Conv, im2col Conv and
  MatMul. A zero-padded plane kernel handles depthwise Conv and the
  multiplier-2 grouped Conv.
- The 30 `Div -> Erf -> Add -> Mul -> Mul` chains run as one fused GELU.
- AVX2+FMA kernels are chosen at run time on x86-64, with a portable
  fallback. AArch64 uses FMA directly.
- A persistent std-only thread pool, so no threads are started per kernel.

## Build and run

```bash
cd rust
cargo build --release
./target/release/mobileclip-s0 photo.jpg
./target/release/mobileclip-s0 --top 5 --threads 4 a.jpg b.png
./target/release/mobileclip-s0 validate
./target/release/mobileclip-s0 bench --runs 20 --profile
```

The model files are found by searching upward from the current directory
and from the executable. You can also pass `--root <repository dir>`.

Other options:

| Option | Effect |
|---|---|
| `--embed-out file.f32` | write the raw 512-float embedding |
| `--raw` | inputs are already-preprocessed `[3,256,256]` f32 files |
| `--trace` | print time and output shape of every step |
| `MOBILECLIP_NO_FMA=1` | force the portable kernels |

Library use:

```rust
let root = std::path::Path::new("..");
let model = mobileclip_s0::MobileClip::load(root, 0)?;   // 0 = all cores
let bank = mobileclip_s0::TextBank::load(root)?;
let emb = model.embed(&mobileclip_s0::preprocess::load_image("photo.jpg")?)?;
for m in bank.top_k(&emb, 5)? { println!("{:.6} {}", m.score, m.label); }
```

## Preprocessing

The steps are the same as in `index.html`: center square crop, resize to
256 x 256, RGB, divide by 255, NCHW. No mean/std normalization. EXIF
orientation is applied, as browsers do. The resize uses the `image` crate's
antialiasing triangle filter, so pixels can differ slightly from a
browser canvas.

## Validation

`mobileclip-s0 validate` rebuilds the two deterministic inputs of
`tools/validate_v3.js` bit for bit. It compares the result with the
embeddings of the JavaScript optimized-v2 executor, which was validated
against ONNX Runtime FP32. Those embeddings are stored in `testdata/`.

```text
noise+gradient: finite=YES maxAbs=1.729e-6 cosine=1.000000000 PASS
flat regions:   finite=YES maxAbs=2.205e-6 cosine=1.000000000 PASS
RUST NUMERIC VERDICT=PASS
```

The AVX2+FMA and portable kernels both pass. A real photo
(`C:\Windows\Web\Wallpaper\Spotlight\img50.jpg`, preprocessed by Rust and
fed to both runtimes) gave maxAbs 3.2e-6 and cosine 1.000000000 against
JS v2. The pass limits are maxAbs <= 2e-5 and cosine >= 0.99999. Rust
accumulates in float32, while JavaScript accumulates in float64. GELU
uses a vectorizable rational erf (error < 1e-6) instead of the JS
lookup table.

`cargo test --release` covers the GEMM, every Conv path, the
broadcast/shape operators, erf/GELU and the thread pool, all against
direct reference implementations.

## Speed

AMD Ryzen 3 7320U (4 cores / 8 threads), Windows, one 256x256 image,
warm runs (`bench --runs 30`):

| Runtime | Time per image |
|---|---|
| JS v3, 4 threads, Node.js 24 | ~700 ms |
| Rust, 1 thread | ~130 ms |
| Rust, 4 threads | ~58 ms |
| Rust, 8 threads | ~57 ms |

Conv takes about 80% of the time. Model load (plan + 43.3 MiB weights)
takes about 60 ms.
