# whisper-subs

Rust CLI that generates subtitles from video/audio files using
[`whisper-rs`](https://github.com/tazz4843/whisper-rs) (Rust bindings to
[`whisper.cpp`](https://github.com/ggerganov/whisper.cpp)).

Pipeline: **media file → ffmpeg → 16 kHz mono f32 PCM → whisper.cpp → SRT/VTT/TXT/JSON**.

## Features

- Any input ffmpeg can decode (AVI, MKV, MP4, MOV, WAV, MP3, FLAC, …)
- Output in **SRT**, **VTT**, **TXT**, or **JSON**
- Automatic language detection or explicit `--language`
- Transcribe or translate-to-English (`--task translate`)
- Line reflow (`--max-line-width`) and gap merging (`--merge-gap-ms`)
- Initial prompt (`--prompt`) to bias vocabulary (proper nouns, jargon)
- Backend feature flags: **Metal**, **CoreML**, **CUDA**, **Vulkan**, **OpenBLAS**, or plain CPU
- Progress bar driven by whisper's internal progress callback

## Prerequisites

- **Rust** 1.85+ (needed for edition 2024 and let-chains)
- **A C++ toolchain + `cmake`** — `whisper-rs-sys` builds whisper.cpp from
  vendored C++ source on first compile.
  - macOS: `xcode-select --install` then `brew install cmake`
  - Debian/Ubuntu: `sudo apt install build-essential cmake`
  - Windows: MSVC Build Tools + CMake
- **ffmpeg** on `PATH` (or pass `--ffmpeg /path/to/ffmpeg`)
- A `ggml-*.bin` Whisper model. Download from
  <https://huggingface.co/ggerganov/whisper.cpp>:

  ```bash
  curl -L -o ggml-small.bin \
      https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
  ```

  Recommended starting points:
  - `ggml-small.bin` (~466 MB) — good balance, real-time-ish on CPU
  - `ggml-medium.bin` (~1.5 GB) — clearly better, especially for non-English
  - `ggml-large-v3.bin` (~3.1 GB) — best quality
  - `ggml-large-v3-turbo.bin` (~1.6 GB) — near-large quality, much faster

## Build

Pick the backend for your hardware.

### macOS (Apple Silicon)

```bash
cargo build --release
# Metal is auto-enabled on Apple Silicon; no --features flag needed.
# For Apple Neural Engine too, add: --features coreml
# (CoreML also requires converting the model first; see whisper.cpp docs.)
```

### Linux with NVIDIA GPU

```bash
# CUDA toolkit must be installed and nvcc on PATH.
cargo build --release --features cuda
```

### Linux/Windows CPU (with BLAS)

```bash
# Requires libopenblas-dev (Debian/Ubuntu) or equivalent.
cargo build --release --features openblas
```

### Cross-platform GPU (Vulkan)

```bash
cargo build --release --features vulkan
```

### Plain CPU

```bash
cargo build --release
```

(On Apple Silicon this still pulls in Metal — see "Disabling Metal on Apple
Silicon" below.)

### Disabling Metal on Apple Silicon

Metal is wired into the Cargo manifest for the `aarch64-apple-darwin` target,
so passing `--no-default-features` or omitting `--features metal` on an
M-series Mac does **not** turn it off. This is a Cargo limitation:
target-specific dependency features are unconditional. Two ways around it:

1. **Comment out the auto-enable block in `Cargo.toml`** — find the
   `[target.'cfg(all(target_os = "macos", target_arch = "aarch64"))'.dependencies]`
   block and comment its two lines. Remember not to commit that change.

2. **Build for the Intel triple (runs under Rosetta 2)** — this dodges the
   `target_arch = "aarch64"` cfg entirely:

   ```bash
   rustup target add x86_64-apple-darwin
   cargo build --release --target x86_64-apple-darwin
   ```

   The binary lands at `target/x86_64-apple-darwin/release/whisper-subs`.
   Expect lower throughput than native-arm64 + Metal, obviously.

The binary lands at `target/release/whisper-subs`.

## Usage

```bash
# Simplest case — auto-detect language, output movie.srt next to input
whisper-subs movie.avi --model ./ggml-small.bin

# Italian source, 42-char line width, merge gaps under 200 ms
whisper-subs movie.avi \
    --model ./ggml-medium.bin \
    --language it \
    --max-line-width 42 \
    --merge-gap-ms 200

# Translate German audio → English subtitles
whisper-subs interview.mkv \
    --model ./ggml-large-v3.bin \
    --language de \
    --task translate \
    --format srt

# Bias vocabulary with an initial prompt
whisper-subs tech-talk.mp4 \
    --model ./ggml-small.bin \
    --prompt "Talk about Rust, ownership, borrow checker, async/await, tokio."

# JSON output for downstream pipelines
whisper-subs lecture.mp4 \
    --model ./ggml-small.bin \
    --format json \
    --output lecture.json

# Set model path via env var to avoid repeating it
export WHISPER_MODEL=$HOME/models/ggml-medium.bin
whisper-subs episode.mkv
```

Run `whisper-subs --help` for the full option list.

## Performance notes

- On Apple Silicon (Metal is auto-enabled), a 2-hour film with `small` takes
  a few minutes; `large-v3` is still faster than real-time.
- On CPU-only, `small` is roughly real-time with OpenBLAS; `medium` is 2–3×
  real-time; `large-v3` is 4–8× real-time.
- The `--threads` flag helps on many-core CPU machines; leave at 0 for auto.

## Hallucination control

Whisper's most common failure mode is a **repetition loop** — the same
sentence dumped to the end of the transcript. whisper-subs ships with
defaults tuned for quality over raw throughput. Every knob is overridable.

### Defaults (all on)

| Mitigation               | How it works                                        | Cost            | Flag                       |
|--------------------------|-----------------------------------------------------|-----------------|----------------------------|
| Silero VAD               | Skip non-speech; auto-detected next to `--model`    | tiny            | `--no-vad`, `--vad-model`  |
| Chunked decoding, 15 s   | Cap each decoder call so loops can't spiral         | possible boundary cut | `--vad-max-speech`   |
| No context carry-over    | Don't feed previous-chunk tokens into the next      | —               | (not exposed)              |
| Suppress non-speech      | Drops the "music/thanks" hallucination family       | —               | (not exposed)              |
| Beam search (width 5)    | Explore multiple decoder paths                      | ~2× slower      | `--beam-size` (1 = greedy) |
| Stricter logprob (-0.5)  | Re-decode low-confidence windows at higher temp     | slower on hard audio | `--logprob-threshold` |
| Tail-loop detector       | Truncate or fail if ≥5 identical trailing segments  | —               | `--on-loop`, `--loop-threshold` |

### If you still see loops

Whisper's failure rate on long audio is 1–5% even with everything enabled.
Over a batch you'll see the detector fire occasionally. To push the rate
lower, at increasing speed cost:

```bash
# tighter VAD chunks (smaller blast radius)
whisper-subs -r ~/Videos --vad-max-speech 10

# stricter confidence gate (more windows re-decoded)
whisper-subs -r ~/Videos --logprob-threshold -0.3

# wider beam (slowest, most robust)
whisper-subs -r ~/Videos --beam-size 8
```

Diminishing returns beyond that. If a specific file reliably fails,
try biasing the decoder with `--prompt "topical words"`, or switch to
a smaller model — paradoxically, larger models hallucinate more on
noisy or music-heavy audio.

### Trading accuracy for speed

The defaults prioritise transcription quality; if throughput matters more
than the occasional loop, flip the two dominant costs off:

```bash
whisper-subs -r ~/Videos --beam-size 1 --logprob-threshold -1.0
```

- **~2× faster** — matches whisper-cli's stock defaults.
- **Higher hallucination rate** — the loop detector (`--on-loop`) will fire
  noticeably more often on long batches, and subtler quality drops (phantom
  words, dropped trailing phrases) become more common too. The quality-first
  defaults exist specifically because those whisper-cli defaults miss too
  many bad decodes on feature-length audio.

You can dial either knob independently if you want a compromise: keep
`--beam-size 1` but leave `--logprob-threshold` at `-0.5` (about 1.5× faster,
middling robustness), or vice versa.

## Project layout

```
whisper-subs/
├── Cargo.toml
├── README.md
└── src/
    ├── main.rs        # CLI entry point, orchestration
    ├── audio.rs       # ffmpeg → f32 PCM extraction
    ├── transcribe.rs  # whisper-rs wrapper
    └── srt.rs         # SRT/VTT/TXT/JSON rendering + reflow/merge
```

## Tests

```bash
cargo test
```

Covers timestamp formatting, line reflow, segment merging, SRT structure,
and JSON escaping. The whisper/ffmpeg paths are integration-tested by
running the binary on a short sample (not included).

## Troubleshooting

- **`ffmpeg not found`** — install ffmpeg (`brew install ffmpeg`,
  `apt install ffmpeg`) or pass `--ffmpeg /full/path`.
- **`model file not found`** — download from the HuggingFace link above.
- **Repeated/looping output on silent sections** — try `--merge-gap-ms 500`
  or use a larger model; Whisper's known hallucination failure mode.
- **Wrong language auto-detected** — pass `--language <code>` explicitly
  for long files; detection runs on the first 30 s only.
- **CUDA build fails** — make sure `nvcc --version` works and
  `CUDA_PATH` / `LD_LIBRARY_PATH` point at your toolkit.

## License

MIT OR Apache-2.0
