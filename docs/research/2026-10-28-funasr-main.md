# FunASR main adoption — 2026-10-28

MindBridge now installs a fixed FunASR main snapshot rather than the 1.4.7 PyPI package.
The accepted scope is upstream speaker-assignment correctness, bounded clustering work,
explicit execution placement, and compatibility with Nano's zero-width punctuation spans.
No model weights, generation prompts, language coverage, or retrieval policy are changed.

## Reproducibility pins

| Component | Revision |
| --- | --- |
| FunASR source | `940b0d2463141a2e9ace8a9fd965b4aea93f2db5`, main checked on 2026-10-28 |
| Previous package | `funasr==1.4.7`, wheel SHA-256 `7a0a87c0eadef8e83c3c53f0be5f6c23c96a85394b299dec413420a3852364dd` |
| ASR | `FunAudioLLM/Fun-ASR-Nano-2512@05201c46f1c38592b1567f857c0d56eab3d0d8ef` |
| VAD | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch@f9a8b8274674755d925277e27063869038d41515` |
| Speaker | `iic/speech_campplus_sv_zh-cn_16k-common@a045b2afcaa9c3049c98a9215a2bc274407ab237` |
| Local environment | Python 3.13.12, NumPy 2.4.6, Torch/TorchAudio 2.11.0+cu128, Transformers 5.19.0 |
| Accelerator available for the real-model check | NVIDIA GeForce RTX 4090, 24 GiB |

The package still reports version 1.4.16; its installed `direct_url.json` records the full Git
commit. Both `local` and `all` use that direct Git requirement, including in built wheel metadata.
The pinned source is [available upstream](https://github.com/modelscope/FunASR/tree/940b0d2463141a2e9ace8a9fd965b4aea93f2db5).

## Deterministic evidence

The speaker-assignment fixture is one sentence covering 0–1000 ms. Diarization assigns
speaker 0 to 0–400 ms and speaker 1 to 400–1000 ms. Pass criterion: select speaker 1, which
has greater total overlap. The unmodified helper extracted from the hash-verified 1.4.7 wheel
selects speaker 0; the installed pinned main helper selects speaker 1. This reproduces the
[upstream attribution defect](https://github.com/modelscope/FunASR/pull/3733) without weights.
It proves the accounting correction, not acoustic speaker-recognition accuracy.

The adapter regression fixtures also check:

- Token spans `[0, 100]`, `[100, 100]`, `[400, 600]` are accepted and produce a positive
  0–600 ms speech turn. Equivalent dictionary timestamps in seconds have the same result.
  A transcript containing only a zero-width time range remains invalid.
- Non-finite or reversed sentence boundaries fail before persistence.
- VAD retains its 30-second segment ceiling when placed on CPU independently of CUDA ASR.
  Speaker placement and strict positive CPU-thread budgets reach the official constructor.
- The new runtime changes the speech space; execution placement does not. A directory opened
  with the old space rejects the new transcriber before loading weights.

The old space suffix was `b47a175e755dea61`; the new default suffix is `2f09571780abadb7`.
No compatibility reader or cache migration is provided. Replay inputs into a fresh directory.
The independently covered directory-ownership and stale-index hydration contracts remain in
the full test suite.

The following focused command passed 47 tests:

```bash
uv run --frozen pytest -q -W error \
  tests/unit/models/test_funasr.py \
  tests/contracts/test_funasr_contract.py \
  tests/unit/test_package.py
```

The installed upstream-helper check runs in a subprocess because FunASR and ModelScope install
process-global logging handlers on import. It does not change the parent suite's logger state.

The upstream suites `test_campplus_utils.py`, `test_auto_model_ncpu.py`,
`test_submodel_device.py`, `test_fun_asr_nano_punctuation_timestamps.py`, and
`test_pcm_input_format.py`, run from the same pinned checkout in MindBridge's environment,
passed 153 tests and 2 subtests. That upstream run emitted 80 warnings, including upstream
NumPy/PyTorch array-conversion deprecations; it was not a warnings-as-errors certification.
These are weights-free regressions, not a GPU performance or acoustic-quality benchmark.

## Real-model fixture and acceptance

The bounded check uses the public
[English sample at an immutable model revision](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/resolve/272c57b82523ada6fd87095e955f8e29100979ab/example/en.mp3).
The input is 57,441 bytes, SHA-256
`f10378336a4e584f3f63799e62f99d5add3c2a401b51d3abe7d3a3a82f255ada`.
The check calls the public `FunASRTranscriber.analyze()` twice, with ASR on CUDA, VAD and
CAM++ on CPU, and `ncpu=2`. It retains aggregate outcomes instead of response bodies.

Pass criteria are nonempty text, positive turn ranges, a returned centroid for every speaker
label, and identical repeated analyses. A mixed batch of the speech fixture and a generated
2-second, mono 16 kHz PCM16 silence WAV must preserve the speech analysis and return an empty
analysis for silence. Both execution configurations passed every criterion:

| Configuration | Actual ASR / VAD / CAM++ devices | Actual PyTorch / BLAS threads | Warm analysis | Peak CUDA allocated |
| --- | --- | --- | --- | --- |
| Default placement, `device=cuda`, `ncpu=4` | CUDA:0 / CUDA:0 / CUDA:0 | 4 / 4 | 1.036 s | 3,582,380,544 bytes |
| Explicit CPU submodels, `ncpu=2` | CUDA:0 / CPU / CPU | 2 / 2 | 1.409 s | 3,552,785,408 bytes |

Each run returned one timed speech turn and one speaker centroid. The second single-asset
analysis and the speech part of the mixed batch both equalled the first analysis. Device and
thread observations were taken after all calls, so these checks also cover repeated placement.
Aggregate outcomes are retained in [results.json](artifacts/funasr-main/results.json).

The first attempt hit a 300-second bound while downloading the 2,127,426,538-byte ASR weights;
the next attempt resumed the download and passed. The two tabled runs then reused that cache.
Cold elapsed times were 49.630 s and 48.623 s respectively, including startup, Hub checks and
loading. Warm time measures only one repeated analysis, and the runs were sequential rather
than randomized. These observations do not establish an end-to-end speedup, CER/WER,
Japanese or Chinese quality, multi-speaker accuracy, long-recording coverage, or performance
relative to the previous runtime.

To reproduce, download the immutable English fixture above, verify its SHA-256, and run the
following in the synchronized environment. Change the transcriber settings to the second row
for the mixed-device case; use a separate process for each CPU budget. No private audio or
model response body is required:

```python
from hashlib import sha256
from pathlib import Path

from mindbridge import AssetRef, FunASRTranscriber, Modality

path = Path("en.mp3").resolve()
data = path.read_bytes()
digest = sha256(data).hexdigest()
assert digest == "f10378336a4e584f3f63799e62f99d5add3c2a401b51d3abe7d3a3a82f255ada"
asset = AssetRef(
    digest,
    modality=Modality.AUDIO,
    media_type="audio/mpeg",
    size_bytes=len(data),
    sha256=digest,
    path=path,
)
transcriber = FunASRTranscriber(device="cuda")
try:
    first = transcriber.analyze((asset,))[0]
    assert first.turns and first.speakers
    assert first == transcriber.analyze((asset,))[0]
    assert all(turn.end_ms > turn.start_ms >= 0 for turn in first.turns)
    labels = {speaker.speaker_label for speaker in first.speakers}
    assert all(turn.speaker_label in labels for turn in first.turns)
finally:
    transcriber.close()
```

## Execution and deployment limits

Upstream's [clustering change](https://github.com/modelscope/FunASR/pull/3699) avoids computing
unused eigenpairs and applies `ncpu` to BLAS. Its published timing is upstream evidence from
one recording, not a MindBridge end-to-end speedup. Thread limits are process-wide, remain
after closing the transcriber, and also affect co-resident embedding/numerical work.

The default speaker backend and its serialized execution remain in place. Native Transformers,
MOSS, raw PCM input, and new language checkpoints are not added as alternate product paths.
They need their own contracts and task-specific evidence.

Git is required to install the pinned extras. Source and wheel builds succeeded, and both built
wheel requirements retain the full FunASR commit. Docker is unavailable in this environment;
the pinned Markdown and link tools are run directly at the repository-required versions.

## Final validation

`uv sync --locked --default-index https://pypi.org/simple --all-groups --all-extras`, the lock
check, Ruff formatting and lint checks, strict mypy, source/wheel builds, and `git diff --check`
passed. The final `uv run --frozen pytest -W error` run passed 3,308 tests with 12 skipped
in 786.31 seconds. The skips are the opt-in external formation-time evaluation; those checks
require `MINDBRIDGE_FORMATION_TIME_EVAL=1` and an OpenAI API key and are unrelated to ASR.

Markdownlint CLI2 0.23.0 and Lychee 0.23.0's offline local-link check passed. The final full
network check of the shared working tree checked 940 targets: 931 succeeded and 9 redirected,
with no errors. No host exclusions or accepted-error overrides were added.

See [local speech configuration](../configuration.md#local-speech-with-funasr) for supported
settings and the fresh-directory requirement.
