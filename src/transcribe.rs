//! Whisper transcription wrapper around `whisper-rs`.

use std::path::Path;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperVadParams,
};

/// One transcribed segment with millisecond-precision start/end times.
#[derive(Debug, Clone)]
pub struct Segment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

pub struct Config<'a> {
    pub samples: &'a [f32],
    pub language: &'a str,
    pub translate: bool,
    pub threads: usize,
    pub prompt: Option<&'a str>,
    pub progress: bool,
    /// Silero VAD model (e.g. `ggml-silero-v5.1.2.bin`). When set, whisper will
    /// skip non-speech regions — kills silence-driven hallucinations and
    /// speeds things up on long files with pauses.
    pub vad_model: Option<&'a Path>,
}

/// Load the whisper model. Callers should do this once and reuse the returned
/// context across many `run()` calls — loading is several seconds for
/// large-v3 and dominates batch runtime otherwise.
pub fn load_context(model_path: &Path) -> Result<WhisperContext> {
    WhisperContext::new_with_params(
        model_path
            .to_str()
            .context("model path is not valid UTF-8")?,
        WhisperContextParameters::default(),
    )
    .context("loading whisper model")
}

pub fn run(ctx: &WhisperContext, cfg: Config<'_>) -> Result<Vec<Segment>> {
    let mut state = ctx.create_state().context("creating whisper state")?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some(cfg.language));
    params.set_translate(cfg.translate);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_suppress_blank(true);
    // Anti-repetition: mirrors the flags from whisper.cpp's video-to-srt.sh
    // (-mc 0 and -sns). Without these, a single hallucinated line gets fed
    // back as context and repeats to the end of the file on long inputs.
    params.set_no_context(true);
    params.set_suppress_nst(true);
    // Temperature fallback + quality gates: re-decode bad windows instead of
    // emitting them. Values match whisper.cpp defaults.
    params.set_temperature(0.0);
    params.set_temperature_inc(0.2);
    params.set_entropy_thold(2.4);
    params.set_logprob_thold(-1.0);
    params.set_no_speech_thold(0.6);
    if cfg.threads > 0 {
        params.set_n_threads(cfg.threads as i32);
    }
    if let Some(p) = cfg.prompt {
        params.set_initial_prompt(p);
    }
    if let Some(vad_path) = cfg.vad_model {
        let vad_str = vad_path
            .to_str()
            .context("VAD model path is not valid UTF-8")?;
        params.set_vad_model_path(Some(vad_str));
        // Cap chunk length to whisper's native 30s window. The library default
        // is f32::MAX, which can send a 10-minute uninterrupted monologue as a
        // single chunk — giving a repetition loop the whole chunk to spiral.
        // Splitting on silence points inside the cap bounds each decoder
        // call's blast radius so no_context can reset between chunks.
        let mut vad_params = WhisperVadParams::default();
        vad_params.set_max_speech_duration(29.0);
        params.set_vad_params(vad_params);
        params.enable_vad(true);
    }

    // Optional progress bar driven by whisper's progress callback.
    let pb = if cfg.progress {
        let bar = ProgressBar::new(100);
        bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} transcribing [{bar:40.cyan/blue}] {pos:>3}%  {elapsed}",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        Some(bar)
    } else {
        None
    };

    if let Some(ref bar) = pb {
        let bar = bar.clone();
        params.set_progress_callback_safe(move |p| {
            bar.set_position(p as u64);
        });
    }

    state
        .full(params, cfg.samples)
        .context("whisper inference failed")?;

    if let Some(bar) = pb {
        bar.finish_and_clear();
    }

    let n = state.full_n_segments();
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let seg = state
            .get_segment(i)
            .with_context(|| format!("segment {i} out of bounds"))?;
        let text = seg.to_str().context("segment text")?.trim().to_string();
        if text.is_empty() {
            continue;
        }
        // Whisper returns t0/t1 in "centiseconds" (hundredths of a second).
        let t0 = seg.start_timestamp() as u64 * 10;
        let t1 = seg.end_timestamp() as u64 * 10;
        out.push(Segment {
            start_ms: t0,
            end_ms: t1,
            text,
        });
    }
    Ok(out)
}
