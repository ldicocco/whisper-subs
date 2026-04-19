//! whisper-subs: generate SRT subtitles from video/audio files.
//!
//! Pipeline: input media → ffmpeg → 16kHz mono f32 PCM → whisper.cpp → SRT with reflow.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use tracing::{info, warn};
use whisper_rs::WhisperContext;

mod audio;
mod srt;
mod transcribe;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    /// SubRip subtitles
    Srt,
    /// WebVTT subtitles
    Vtt,
    /// Plain text (no timing)
    Txt,
    /// Raw JSON segments (for downstream processing)
    Json,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Task {
    /// Transcribe in source language
    Transcribe,
    /// Translate to English
    Translate,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum LoopAction {
    /// Drop the looped tail, write what we had before it, log a warning
    Truncate,
    /// Abort with non-zero exit; no output file is written
    Fail,
    /// Ignore the check entirely
    Off,
}

#[derive(Parser, Debug)]
#[command(
    name = "whisper-subs",
    version,
    about = "Generate subtitles from video/audio files using whisper.cpp",
    long_about = None,
)]
struct Cli {
    /// Input media file (any format ffmpeg can read: avi, mkv, mp4, wav, mp3, ...)
    input: PathBuf,

    /// Path to ggml whisper model (e.g. ggml-small.bin). Download from:
    /// https://huggingface.co/ggerganov/whisper.cpp
    #[arg(
        short,
        long,
        env = "WHISPER_MODEL",
        default_value = "models/ggml-large-v3-turbo.bin"
    )]
    model: PathBuf,

    /// Output file path. Defaults to <input>.<ext> next to the input.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Output format
    #[arg(short = 'f', long, value_enum, default_value_t = OutputFormat::Srt)]
    format: OutputFormat,

    /// Language code (ISO 639-1, e.g. "en", "it", "de"). "auto" for detection.
    #[arg(short, long, default_value = "auto")]
    language: String,

    /// Task: transcribe or translate-to-English
    #[arg(short, long, value_enum, default_value_t = Task::Transcribe)]
    task: Task,

    /// Number of CPU threads (0 = auto)
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Maximum characters per subtitle line before wrapping
    #[arg(long, default_value_t = 42)]
    max_line_width: usize,

    /// Merge consecutive segments whose gap is below this many milliseconds
    #[arg(long, default_value_t = 0)]
    merge_gap_ms: u32,

    /// Initial prompt to bias vocabulary (proper nouns, jargon, style)
    #[arg(long)]
    prompt: Option<String>,

    /// Print progress bar
    #[arg(long, default_value_t = true)]
    progress: bool,

    /// ffmpeg binary path (defaults to `ffmpeg` on PATH)
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg: String,

    /// Path to a Silero VAD model (e.g. ggml-silero-v5.1.2.bin). If omitted,
    /// auto-detected next to --model. Use --no-vad to disable.
    #[arg(long, env = "VAD_MODEL")]
    vad_model: Option<PathBuf>,

    /// Disable VAD even if a Silero model is found next to --model.
    #[arg(long)]
    no_vad: bool,

    /// When input is a directory, process at most N files. Files that already
    /// have a subtitle of the selected format don't count toward the limit.
    #[arg(long, value_name = "N")]
    limit: Option<usize>,

    /// When input is a directory, also scan subdirectories.
    #[arg(short, long)]
    recursive: bool,

    /// What to do when whisper's "repeat-the-same-line-to-the-end" failure is
    /// detected (>= --loop-threshold consecutive identical trailing segments).
    #[arg(long, value_enum, default_value_t = LoopAction::Truncate)]
    on_loop: LoopAction,

    /// Minimum consecutive identical trailing segments that count as a loop.
    #[arg(long, default_value_t = 5, value_name = "N")]
    loop_threshold: usize,

    /// Beam search width. Larger = more robust against hallucinations,
    /// slower (~2× at beam=5 vs 1). Set to 1 for greedy decoding.
    #[arg(long, default_value_t = 5, value_name = "N")]
    beam_size: usize,

    /// Log-probability threshold — windows whose average logprob falls below
    /// this trigger a temperature-fallback re-decode. Closer to 0 = stricter.
    /// whisper-cli default: -1.0.
    #[arg(long, default_value_t = -0.5, value_name = "LP", allow_hyphen_values = true)]
    logprob_threshold: f32,

    /// Max length (seconds) of a single speech chunk when VAD is active.
    /// Smaller = smaller blast radius if a chunk loops; risk of mid-word
    /// boundary on long monologues.
    #[arg(long, default_value_t = 15.0, value_name = "SECS")]
    vad_max_speech: f32,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
}

/// File extensions treated as "video" when input is a directory.
/// Case-insensitive match.
const VIDEO_EXTS: &[&str] = &[
    "mp4", "mkv", "avi", "mov", "webm", "m4v", "wmv", "flv", "ogv", "ts", "mpg", "mpeg",
];

fn output_ext(fmt: OutputFormat) -> &'static str {
    match fmt {
        OutputFormat::Srt => "srt",
        OutputFormat::Vtt => "vtt",
        OutputFormat::Txt => "txt",
        OutputFormat::Json => "json",
    }
}

/// Collect videos in `dir` that don't already have a subtitle of `fmt` sitting
/// next to them. With `recursive`, descends into subdirectories; a canonicalized
/// visited-set prevents symlink loops. Result is sorted alphabetically.
fn collect_batch(dir: &Path, fmt: OutputFormat, recursive: bool) -> Result<Vec<PathBuf>> {
    let sub_ext = output_ext(fmt);
    let mut found = Vec::new();
    let mut skipped = 0usize;
    let mut stack: Vec<PathBuf> = vec![dir.to_path_buf()];
    let mut visited: HashSet<PathBuf> = HashSet::new();
    while let Some(d) = stack.pop() {
        let canonical = std::fs::canonicalize(&d).unwrap_or_else(|_| d.clone());
        if !visited.insert(canonical) {
            continue;
        }
        for entry in
            std::fs::read_dir(&d).with_context(|| format!("reading directory {}", d.display()))?
        {
            let entry = entry.context("reading directory entry")?;
            let path = entry.path();
            if path.is_dir() {
                if recursive {
                    stack.push(path);
                }
                continue;
            }
            if !path.is_file() {
                continue;
            }
            let ext = match path.extension().and_then(|e| e.to_str()) {
                Some(e) => e.to_ascii_lowercase(),
                None => continue,
            };
            if !VIDEO_EXTS.contains(&ext.as_str()) {
                continue;
            }
            if path.with_extension(sub_ext).exists() {
                skipped += 1;
                continue;
            }
            found.push(path);
        }
    }
    found.sort();
    if skipped > 0 {
        info!("skipping {skipped} file(s) that already have .{sub_ext} subtitles");
    }
    Ok(found)
}

/// Resolve the Silero VAD model path: explicit flag wins; otherwise look for
/// `ggml-silero-v5.1.2.bin` / `v6.2.0.bin` next to the whisper model, mirroring
/// the detection logic in whisper.cpp's `scripts/video-to-srt.sh`.
fn resolve_vad_model(cli: &Cli) -> Option<PathBuf> {
    if cli.no_vad {
        return None;
    }
    if let Some(p) = &cli.vad_model {
        return p.exists().then(|| p.clone());
    }
    let dir = cli.model.parent()?;
    for name in ["ggml-silero-v5.1.2.bin", "ggml-silero-v6.2.0.bin"] {
        let candidate = dir.join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    let filter = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| filter.into()),
        )
        .with_target(false)
        .compact()
        .init();

    if !cli.input.exists() {
        anyhow::bail!("input not found: {}", cli.input.display());
    }
    if !cli.model.exists() {
        anyhow::bail!(
            "model file not found: {}. Download from https://huggingface.co/ggerganov/whisper.cpp",
            cli.model.display()
        );
    }

    let vad_model = resolve_vad_model(&cli);
    match (&vad_model, cli.no_vad) {
        (Some(p), _) => info!("VAD: on ({})", p.display()),
        (None, true) => info!("VAD: off (--no-vad)"),
        (None, false) => warn!(
            "VAD: off — no Silero model found next to {}. \
             Pass --vad-model or drop ggml-silero-v5.1.2.bin next to the whisper model \
             to reduce hallucinations on long files.",
            cli.model.display()
        ),
    }

    info!("model:  {}", cli.model.display());

    if cli.input.is_dir() {
        if cli.output.is_some() {
            anyhow::bail!(
                "--output cannot be used with a directory input; \
                 each video gets a sibling subtitle file automatically"
            );
        }
        let mut files = collect_batch(&cli.input, cli.format, cli.recursive)?;
        if files.is_empty() {
            info!("nothing to do in {}", cli.input.display());
            return Ok(());
        }
        if let Some(n) = cli.limit
            && files.len() > n
        {
            info!("capping batch at {n}/{} file(s)", files.len());
            files.truncate(n);
        }
        info!("loading whisper model…");
        let ctx = transcribe::load_context(&cli.model)?;
        run_batch(&cli, &ctx, vad_model.as_deref(), &files)
    } else {
        if cli.limit.is_some() {
            warn!("--limit is ignored when input is a single file");
        }
        let output_path = cli
            .output
            .clone()
            .unwrap_or_else(|| cli.input.with_extension(output_ext(cli.format)));
        info!("loading whisper model…");
        let ctx = transcribe::load_context(&cli.model)?;
        process_one(&cli, &ctx, &cli.input, &output_path, vad_model.as_deref())?;
        if matches!(cli.language.as_str(), "auto") {
            warn!(
                "note: language was auto-detected; pass --language for better accuracy on long files"
            );
        }
        Ok(())
    }
}

fn run_batch(
    cli: &Cli,
    ctx: &WhisperContext,
    vad_model: Option<&Path>,
    files: &[PathBuf],
) -> Result<()> {
    info!(
        "processing {} file(s) from {}",
        files.len(),
        cli.input.display()
    );

    let sub_ext = output_ext(cli.format);
    let mut ok = 0usize;
    let mut failed = 0usize;
    let total = files.len();
    for (i, input) in files.iter().enumerate() {
        let output_path = input.with_extension(sub_ext);
        info!("[{}/{}] {}", i + 1, total, input.display());
        match process_one(cli, ctx, input, &output_path, vad_model) {
            Ok(()) => ok += 1,
            Err(e) => {
                failed += 1;
                warn!("failed on {}: {e:#}", input.display());
            }
        }
    }
    info!("batch done: {ok} ok, {failed} failed");
    if failed > 0 {
        anyhow::bail!("{failed} file(s) failed; see warnings above");
    }
    Ok(())
}

fn process_one(
    cli: &Cli,
    ctx: &WhisperContext,
    input: &Path,
    output_path: &Path,
    vad_model: Option<&Path>,
) -> Result<()> {
    info!("extracting audio with ffmpeg…");
    let samples = audio::extract_pcm_f32(input, &cli.ffmpeg).context("audio extraction failed")?;
    info!(
        "extracted {} samples ({:.1}s of audio)",
        samples.len(),
        samples.len() as f32 / 16_000.0
    );

    info!("transcribing…");
    let t0 = std::time::Instant::now();
    let mut segments = transcribe::run(
        ctx,
        transcribe::Config {
            samples: &samples,
            language: &cli.language,
            translate: matches!(cli.task, Task::Translate),
            threads: cli.threads,
            prompt: cli.prompt.as_deref(),
            progress: cli.progress,
            vad_model,
            vad_max_speech_s: cli.vad_max_speech,
            beam_size: cli.beam_size,
            logprob_thold: cli.logprob_threshold,
        },
    )
    .context("transcription failed")?;
    let elapsed = t0.elapsed().as_secs_f32();
    let audio_s = samples.len() as f32 / audio::SAMPLE_RATE as f32;
    let rtf = if elapsed > 0.0 {
        audio_s / elapsed
    } else {
        0.0
    };
    info!(
        "transcribed {audio_s:.0}s of audio in {elapsed:.1}s ({rtf:.1}× real-time), {} segments",
        segments.len()
    );

    if cli.merge_gap_ms > 0 {
        let before = segments.len();
        segments = srt::merge_close_segments(segments, cli.merge_gap_ms);
        if segments.len() != before {
            info!("merged {} → {} segments", before, segments.len());
        }
    }

    if !matches!(cli.on_loop, LoopAction::Off)
        && let Some(start) = srt::detect_tail_loop(&segments, cli.loop_threshold)
    {
        let run_len = segments.len() - start;
        let sample = segments[start].text.replace('\n', " ");
        let start_ms = segments[start].start_ms;
        let hh = start_ms / 3_600_000;
        let mm = (start_ms / 60_000) % 60;
        let ss = (start_ms / 1_000) % 60;
        let ts = format!("{hh:02}:{mm:02}:{ss:02}");
        match cli.on_loop {
            LoopAction::Fail => {
                anyhow::bail!(
                    "whisper repetition loop detected: \"{sample}\" repeats \
                     {run_len}× starting at {ts} (input: {}). \
                     Pass --on-loop truncate to salvage everything before the loop.",
                    input.display()
                );
            }
            LoopAction::Truncate => {
                warn!(
                    "whisper repetition loop detected: \"{sample}\" repeats \
                     {run_len}× starting at {ts} — truncating output there."
                );
                segments.truncate(start);
                if segments.is_empty() {
                    anyhow::bail!(
                        "whisper repetition loop covered the entire transcript; \
                         nothing to write"
                    );
                }
            }
            LoopAction::Off => unreachable!(),
        }
    }

    if matches!(cli.format, OutputFormat::Srt | OutputFormat::Vtt) {
        for seg in &mut segments {
            seg.text = srt::reflow_text(&seg.text, cli.max_line_width);
        }
    }

    let rendered = match cli.format {
        OutputFormat::Srt => srt::render_srt(&segments),
        OutputFormat::Vtt => srt::render_vtt(&segments),
        OutputFormat::Txt => srt::render_txt(&segments),
        OutputFormat::Json => srt::render_json(&segments)?,
    };
    std::fs::write(output_path, rendered)
        .with_context(|| format!("writing {}", output_path.display()))?;
    info!("wrote: {}", output_path.display());
    Ok(())
}
