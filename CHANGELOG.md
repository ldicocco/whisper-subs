# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-04-19

### Changed
- Prefer Silero VAD v6 by default and document where to download the model.

## [0.1.0] - 2026-04-19

Initial public release.

### Added
- AVI/video → SRT subtitle generation using whisper.cpp via `whisper-rs`.
- Hallucination mitigation, including handling of the "Transcription by CastingWords" artifact.
- Marking of heavily truncated segments in the output.
- `--loop-near-end-pct` flag to control loop re-entry near the end of a chunk.
- Documentation for speed/quality trade-offs.
- Backend feature flags: `metal`, `coreml`, `cuda`, `vulkan`, `openblas`.
- Auto-enabled Metal backend on Apple Silicon.

[0.1.1]: https://github.com/ldicocco/whisper-subs/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ldicocco/whisper-subs/releases/tag/v0.1.0
