//! SRT/VTT/TXT/JSON rendering plus segment post-processing.

use std::fmt::Write;

use anyhow::Result;

use crate::transcribe::Segment;

/// Merge segments whose inter-segment gap is ≤ `max_gap_ms`.
/// Useful when whisper produces many fragmented cues.
pub fn merge_close_segments(segments: Vec<Segment>, max_gap_ms: u32) -> Vec<Segment> {
    if segments.is_empty() {
        return segments;
    }
    let mut out: Vec<Segment> = Vec::with_capacity(segments.len());
    for seg in segments {
        match out.last_mut() {
            Some(prev) if seg.start_ms.saturating_sub(prev.end_ms) <= max_gap_ms as u64 => {
                prev.end_ms = seg.end_ms;
                if !prev.text.ends_with(' ') {
                    prev.text.push(' ');
                }
                prev.text.push_str(seg.text.trim());
            }
            _ => out.push(seg),
        }
    }
    out
}

/// Normalize a line for loop-detection comparison: lowercase, collapse any
/// run of non-alphanumeric chars to a single space, trim. This catches
/// whisper's typical repeat-to-end-of-file failure (exact-duplicate lines
/// modulo trailing punctuation / casing).
fn normalize_for_loop(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_space = true;
    for c in s.chars() {
        if c.is_alphanumeric() {
            for lower in c.to_lowercase() {
                out.push(lower);
            }
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

/// Detect whisper's classic failure mode: the tail of the transcript is the
/// same line repeated. Returns the index of the first segment in the loop if
/// `>= threshold` consecutive segments at the end share the same normalized
/// text. Otherwise `None`.
pub fn detect_tail_loop(segments: &[Segment], threshold: usize) -> Option<usize> {
    if threshold < 2 || segments.len() < threshold {
        return None;
    }
    let key = normalize_for_loop(&segments[segments.len() - 1].text);
    if key.is_empty() {
        return None;
    }
    let mut run_start = segments.len();
    for (i, seg) in segments.iter().enumerate().rev() {
        if normalize_for_loop(&seg.text) == key {
            run_start = i;
        } else {
            break;
        }
    }
    let run_len = segments.len() - run_start;
    if run_len >= threshold {
        Some(run_start)
    } else {
        None
    }
}

/// Reflow a single-line text into multiple lines no wider than `max_width`
/// characters, breaking on whitespace. Never hyphenates; if a single word is
/// longer than `max_width` it goes on its own line.
pub fn reflow_text(text: &str, max_width: usize) -> String {
    let text = text.trim();
    if max_width == 0 || text.chars().count() <= max_width {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len() + 4);
    let mut line_len = 0usize;
    for word in text.split_whitespace() {
        let w = word.chars().count();
        if line_len == 0 {
            out.push_str(word);
            line_len = w;
        } else if line_len + 1 + w > max_width {
            out.push('\n');
            out.push_str(word);
            line_len = w;
        } else {
            out.push(' ');
            out.push_str(word);
            line_len += 1 + w;
        }
    }
    out
}

pub fn render_srt(segments: &[Segment]) -> String {
    let mut s = String::new();
    for (i, seg) in segments.iter().enumerate() {
        let _ = writeln!(s, "{}", i + 1);
        let _ = writeln!(
            s,
            "{} --> {}",
            format_ts_srt(seg.start_ms),
            format_ts_srt(seg.end_ms)
        );
        let _ = writeln!(s, "{}", seg.text);
        s.push('\n');
    }
    s
}

pub fn render_vtt(segments: &[Segment]) -> String {
    let mut s = String::from("WEBVTT\n\n");
    for seg in segments {
        let _ = writeln!(
            s,
            "{} --> {}",
            format_ts_vtt(seg.start_ms),
            format_ts_vtt(seg.end_ms)
        );
        let _ = writeln!(s, "{}", seg.text);
        s.push('\n');
    }
    s
}

pub fn render_txt(segments: &[Segment]) -> String {
    let mut s = String::new();
    for seg in segments {
        s.push_str(&seg.text.replace('\n', " "));
        s.push('\n');
    }
    s
}

pub fn render_json(segments: &[Segment]) -> Result<String> {
    // Hand-rolled to avoid a serde dependency for one small struct.
    let mut s = String::from("[\n");
    for (i, seg) in segments.iter().enumerate() {
        let text = json_escape(&seg.text);
        let _ = write!(
            s,
            "  {{\"start_ms\": {}, \"end_ms\": {}, \"text\": \"{}\"}}",
            seg.start_ms, seg.end_ms, text
        );
        if i + 1 < segments.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push(']');
    s.push('\n');
    Ok(s)
}

fn format_ts_srt(ms: u64) -> String {
    let (h, m, s, ms) = hms(ms);
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, ms)
}

fn format_ts_vtt(ms: u64) -> String {
    let (h, m, s, ms) = hms(ms);
    format!("{:02}:{:02}:{:02}.{:03}", h, m, s, ms)
}

fn hms(total_ms: u64) -> (u64, u64, u64, u64) {
    let ms = total_ms % 1000;
    let total_s = total_ms / 1000;
    let s = total_s % 60;
    let total_m = total_s / 60;
    let m = total_m % 60;
    let h = total_m / 60;
    (h, m, s, ms)
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srt_timestamp_format() {
        assert_eq!(format_ts_srt(0), "00:00:00,000");
        assert_eq!(format_ts_srt(1_234), "00:00:01,234");
        assert_eq!(format_ts_srt(3_661_500), "01:01:01,500");
    }

    #[test]
    fn vtt_timestamp_format() {
        assert_eq!(format_ts_vtt(3_661_500), "01:01:01.500");
    }

    #[test]
    fn reflow_short_text_unchanged() {
        assert_eq!(reflow_text("hello world", 40), "hello world");
    }

    #[test]
    fn reflow_wraps_on_word_boundary() {
        let r = reflow_text("the quick brown fox jumps over the lazy dog", 20);
        for line in r.lines() {
            assert!(line.chars().count() <= 20, "line too long: {:?}", line);
        }
    }

    #[test]
    fn merge_joins_adjacent_segments() {
        let segs = vec![
            Segment {
                start_ms: 0,
                end_ms: 1_000,
                text: "hello".into(),
            },
            Segment {
                start_ms: 1_050,
                end_ms: 2_000,
                text: "world".into(),
            },
            Segment {
                start_ms: 5_000,
                end_ms: 6_000,
                text: "later".into(),
            },
        ];
        let merged = merge_close_segments(segs, 100);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].text, "hello world");
        assert_eq!(merged[0].end_ms, 2_000);
        assert_eq!(merged[1].text, "later");
    }

    #[test]
    fn srt_renders_numbered_blocks() {
        let segs = vec![
            Segment {
                start_ms: 0,
                end_ms: 1_000,
                text: "hi".into(),
            },
            Segment {
                start_ms: 1_000,
                end_ms: 2_000,
                text: "bye".into(),
            },
        ];
        let s = render_srt(&segs);
        assert!(s.contains("1\n00:00:00,000 --> 00:00:01,000\nhi\n"));
        assert!(s.contains("2\n00:00:01,000 --> 00:00:02,000\nbye\n"));
    }

    fn seg(t: &str) -> Segment {
        Segment {
            start_ms: 0,
            end_ms: 0,
            text: t.into(),
        }
    }

    #[test]
    fn detect_loop_returns_none_on_clean_transcript() {
        let segs = vec![
            seg("hello"),
            seg("world"),
            seg("foo"),
            seg("bar"),
            seg("baz"),
        ];
        assert_eq!(detect_tail_loop(&segs, 3), None);
    }

    #[test]
    fn detect_loop_finds_tail_repetition() {
        let segs = vec![
            seg("intro line"),
            seg("another line"),
            seg("Please subscribe."),
            seg("please subscribe"),
            seg("Please subscribe!"),
            seg("Please subscribe."),
        ];
        // normalized form collapses punct/case — last 4 all match "please subscribe".
        assert_eq!(detect_tail_loop(&segs, 3), Some(2));
    }

    #[test]
    fn detect_loop_ignores_short_runs_below_threshold() {
        let segs = vec![seg("a"), seg("b"), seg("c"), seg("c")];
        assert_eq!(detect_tail_loop(&segs, 3), None);
    }

    #[test]
    fn detect_loop_ignores_empty_tail() {
        let segs = vec![seg("real"), seg(""), seg(""), seg("")];
        assert_eq!(detect_tail_loop(&segs, 2), None);
    }

    #[test]
    fn json_escapes_quotes_and_newlines() {
        let segs = vec![Segment {
            start_ms: 0,
            end_ms: 100,
            text: "say \"hi\"\nthere".into(),
        }];
        let j = render_json(&segs).unwrap();
        assert!(j.contains(r#"\"hi\""#));
        assert!(j.contains(r"\n"));
    }
}
