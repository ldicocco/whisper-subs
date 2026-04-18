//! Audio extraction: any media → 16 kHz mono f32 PCM.
//!
//! We shell out to ffmpeg rather than vendoring it. This keeps the binary small
//! and avoids the ffmpeg-sys build-time pain. Swap in `symphonia` later if you
//! want a pure-Rust pipeline.

use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, anyhow};

pub const SAMPLE_RATE: u32 = 16_000;

/// Extract audio from `input` as 16 kHz mono f32 little-endian PCM via ffmpeg,
/// returning samples as a `Vec<f32>` in [-1.0, 1.0].
pub fn extract_pcm_f32(input: &Path, ffmpeg_bin: &str) -> Result<Vec<f32>> {
    // Check ffmpeg is on PATH.
    let version = Command::new(ffmpeg_bin)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    if version.is_err() {
        return Err(anyhow!(
            "`{}` not found; install ffmpeg or pass --ffmpeg <path>",
            ffmpeg_bin
        ));
    }

    // -f f32le: raw float32 little-endian
    // -ac 1:   mono
    // -ar SR:  16kHz
    // -vn:     no video
    // -       pipe to stdout
    let mut child = Command::new(ffmpeg_bin)
        .args(["-nostdin", "-loglevel", "error", "-i"])
        .arg(input)
        .args([
            "-vn",
            "-ac",
            "1",
            "-ar",
            &SAMPLE_RATE.to_string(),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawning ffmpeg")?;

    let stdout = child.stdout.take().expect("piped");
    let bytes = read_all(stdout).context("reading ffmpeg stdout")?;

    let output = child.wait_with_output().context("waiting on ffmpeg")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "ffmpeg exited with {}: {}",
            output.status,
            stderr.trim()
        ));
    }

    if bytes.len() % 4 != 0 {
        return Err(anyhow!(
            "ffmpeg produced {} bytes, not a multiple of 4 (f32)",
            bytes.len()
        ));
    }

    let mut samples = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        samples.push(v);
    }

    if samples.is_empty() {
        return Err(anyhow!("ffmpeg produced no audio samples"));
    }

    Ok(samples)
}

fn read_all<R: std::io::Read>(mut r: R) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(1 << 20);
    r.read_to_end(&mut buf)?;
    Ok(buf)
}
