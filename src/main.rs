//! Command-line front end.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use mobileclip_s0::{kernels, preprocess, testinputs, Error, MobileClip, Profile, Result, TextBank};

const USAGE: &str = "\
MobileCLIP-S0 FP32, pure Rust

Usage:
  mobileclip-s0 [options] <image>...   classify images against the text bank
  mobileclip-s0 validate [options]     compare with the JavaScript reference
  mobileclip-s0 bench [options]        time the vision encoder

Options:
  --root <dir>        repository root holding compiled-fp32-v1/ and text-bank/
                      (default: searched from the current and executable dirs)
  --threads <n>       worker threads, 0 = all cores (default 0)
  --top <k>           labels to print per image (default 10)
  --embed-out <file>  also write the raw 512-float embedding (one image only)
  --raw               inputs are raw [3,256,256] little-endian f32 files
  --runs <n>          bench: timed runs after one warm-up (default 5)
  --profile           print time per operator
  --trace             print time and output shape of every step
  -h, --help          show this help
";

/// JavaScript optimized-v2 embeddings of the validation inputs; v2 was
/// checked against ONNX Runtime FP32 (maxAbs 1.97e-7).
const REF_NOISE: &[u8] = include_bytes!("../testdata/ref_v2_noise.f32");
const REF_FLAT: &[u8] = include_bytes!("../testdata/ref_v2_flat.f32");

/// Accepted distance from the JavaScript reference. Rust accumulates in
/// float32 (JavaScript in float64), so a few ulps per layer are expected.
const MAX_ABS_LIMIT: f32 = 2e-5;
const COSINE_LIMIT: f64 = 0.99999;

struct Args {
    command: Command,
    root: Option<PathBuf>,
    threads: usize,
    top: usize,
    embed_out: Option<PathBuf>,
    raw: bool,
    runs: usize,
    profile: bool,
    trace: bool,
    files: Vec<PathBuf>,
}

#[derive(PartialEq)]
enum Command {
    Classify,
    Validate,
    Bench,
    Help,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        command: Command::Classify,
        root: None,
        threads: 0,
        top: 10,
        embed_out: None,
        raw: false,
        runs: 5,
        profile: false,
        trace: false,
        files: Vec::new(),
    };
    let mut it = std::env::args().skip(1).peekable();
    match it.peek().map(String::as_str) {
        Some("validate") => {
            a.command = Command::Validate;
            it.next();
        }
        Some("bench") => {
            a.command = Command::Bench;
            it.next();
        }
        _ => {}
    }
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| Error::new(format!("{} needs a value", name)));
        let number = |s: String, name: &str| {
            s.parse::<usize>().map_err(|_| Error::new(format!("{}: '{}' is not a number", name, s)))
        };
        match arg.as_str() {
            "-h" | "--help" => a.command = Command::Help,
            "--root" => a.root = Some(value("--root")?.into()),
            "--threads" => a.threads = number(value("--threads")?, "--threads")?,
            "--top" => a.top = number(value("--top")?, "--top")?,
            "--runs" => a.runs = number(value("--runs")?, "--runs")?.max(1),
            "--embed-out" => a.embed_out = Some(value("--embed-out")?.into()),
            "--raw" => a.raw = true,
            "--profile" => a.profile = true,
            "--trace" => {
                a.profile = true;
                a.trace = true;
            }
            s if s.starts_with('-') => return Err(Error::new(format!("unknown option {}", s))),
            _ => a.files.push(arg.into()),
        }
    }
    if a.command == Command::Classify && a.files.is_empty() {
        a.command = Command::Help;
    }
    Ok(a)
}

fn is_root(p: &Path) -> bool {
    p.join("compiled-fp32-v1").join("plan.json").is_file()
}

fn find_root(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(r) = explicit {
        if is_root(&r) {
            return Ok(r);
        }
        return Err(Error::new(format!("{} has no compiled-fp32-v1/plan.json", r.display())));
    }
    let mut starts = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        starts.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        starts.push(exe);
    }
    starts.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    for s in starts {
        for dir in s.ancestors() {
            if is_root(dir) {
                return Ok(dir.to_path_buf());
            }
        }
    }
    Err(Error::new("model files not found; pass --root <repository dir>"))
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(path).map_err(|e| Error::new(format!("{}: {}", path.display(), e)))?;
    Ok(f32s(&b))
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn write_f32(path: &Path, v: &[f32]) -> Result<()> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).map_err(|e| Error::new(format!("{}: {}", path.display(), e)))
}

fn load_model(args: &Args, root: &Path) -> Result<MobileClip> {
    let t0 = Instant::now();
    let model = MobileClip::load(root, args.threads)?;
    let ex = model.executor();
    eprintln!(
        "loaded {} nodes ({} steps, {} fused GELU), {:.1} MiB weights in {:.0} ms; {} threads, {} kernels",
        ex.plan_nodes(),
        ex.steps(),
        ex.fused_gelus(),
        ex.weights_bytes() as f64 / (1 << 20) as f64,
        t0.elapsed().as_secs_f64() * 1e3,
        ex.threads(),
        kernels::kernel_set(),
    );
    Ok(model)
}

fn print_profile(p: &Profile, runs: usize) {
    eprintln!("per-operator time (mean of {} run(s)):", runs);
    for (op, calls, ms) in p.sorted() {
        eprintln!(
            "  {:<20} {:>4} calls {:>9.2} ms  {:>5.1}%",
            op,
            calls as usize / runs,
            ms / runs as f64,
            100.0 * ms / p.total_ms.max(1e-9)
        );
    }
}

fn classify(args: &Args) -> Result<()> {
    let root = find_root(args.root.clone())?;
    let model = load_model(args, &root)?;
    let bank = TextBank::load(&root)?;
    if args.embed_out.is_some() && args.files.len() != 1 {
        return Err(Error::new("--embed-out needs exactly one image"));
    }

    for file in &args.files {
        let t0 = Instant::now();
        let pixels = if args.raw { read_f32(file)? } else { preprocess::load_image(file)? };
        let t_pre = t0.elapsed().as_secs_f64() * 1e3;

        let mut prof = Profile { trace: args.trace, ..Profile::default() };
        let t1 = Instant::now();
        let emb = model.embed_profiled(&pixels, args.profile.then_some(&mut prof))?;
        let t_inf = t1.elapsed().as_secs_f64() * 1e3;

        println!("{}  (preprocess {:.0} ms, inference {:.0} ms)", file.display(), t_pre, t_inf);
        for (i, m) in bank.top_k(&emb, args.top)?.iter().enumerate() {
            println!("{:>3}. {:.6}  {}", i + 1, m.score, m.label);
        }
        if args.profile {
            print_profile(&prof, 1);
        }
        if let Some(out) = &args.embed_out {
            write_f32(out, &emb)?;
            eprintln!("embedding written to {}", out.display());
        }
    }
    Ok(())
}

fn compare(a: &[f32], b: &[f32]) -> (bool, f32, f64) {
    let mut max_abs = 0.0f32;
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    let finite = b.iter().all(|v| v.is_finite());
    for (&x, &y) in a.iter().zip(b) {
        max_abs = max_abs.max((x - y).abs());
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    (finite, max_abs, dot / (na * nb).sqrt())
}

fn validate(args: &Args) -> Result<bool> {
    let root = find_root(args.root.clone())?;
    let model = load_model(args, &root)?;
    let cases = [
        ("noise+gradient", testinputs::noisy_input(), f32s(REF_NOISE)),
        ("flat regions", testinputs::flat_input(), f32s(REF_FLAT)),
    ];
    let mut ok_all = true;
    for (name, input, reference) in cases {
        let t0 = Instant::now();
        let emb = model.embed(&input)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let (finite, max_abs, cos) = compare(&reference, &emb);
        let ok = finite && max_abs <= MAX_ABS_LIMIT && cos >= COSINE_LIMIT;
        ok_all &= ok;
        println!(
            "{}: ms={:.0} finite={} maxAbs={:.3e} cosine={:.9} {}",
            name,
            ms,
            if finite { "YES" } else { "NO" },
            max_abs,
            cos,
            if ok { "PASS" } else { "FAIL" }
        );
    }
    println!("RUST NUMERIC VERDICT={}", if ok_all { "PASS" } else { "FAIL" });
    Ok(ok_all)
}

fn bench(args: &Args) -> Result<()> {
    let root = find_root(args.root.clone())?;
    let model = load_model(args, &root)?;
    let input = testinputs::noisy_input();
    let t0 = Instant::now();
    model.embed(&input)?;
    eprintln!("warm-up: {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);

    let mut prof = Profile { trace: args.trace, ..Profile::default() };
    let mut times = Vec::new();
    for _ in 0..args.runs {
        let t = Instant::now();
        model.embed_profiled(&input, args.profile.then_some(&mut prof))?;
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(f64::total_cmp);
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "runs={} min={:.1} ms median={:.1} ms mean={:.1} ms",
        times.len(),
        times[0],
        times[times.len() / 2],
        mean
    );
    if args.profile {
        print_profile(&prof, args.runs);
    }
    Ok(())
}

fn main() -> ExitCode {
    let result = parse_args().and_then(|args| match args.command {
        Command::Help => {
            print!("{}", USAGE);
            Ok(true)
        }
        Command::Classify => classify(&args).map(|_| true),
        Command::Validate => validate(&args),
        Command::Bench => bench(&args).map(|_| true),
    });
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {}", e);
            ExitCode::FAILURE
        }
    }
}
