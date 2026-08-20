# Pinned AML evaluation contract

Source: <https://github.com/AML-memory/agent-memory-leaderboard>
Revision: 5761ed58502d24153115cbdc010e44957cb18c3a

These files are vendored from that revision and carry exactly one deliberate
local delta, recorded below. Do not edit them for any other reason: scores are
comparable to the leaderboard only while the scoring logic matches this
revision. To upgrade, re-pin the revision, re-apply the delta, and regenerate
this file.

## The one local delta: JSONL records are split on newlines

Every pipeline's JSONL reader (`read_jsonl`, or `rows` in `beam`,
`locomo-refined` and `longmemeval-s`) upstream splits with
`str.splitlines()`. That breaks on U+2028 LINE SEPARATOR, U+2029 and U+0085
NEL as well as on newlines: none is a JSON line delimiter, and a JSON string
may carry all three raw, so a record holding one is cut in half mid-string and
both halves fail to parse. It breaks on U+000B, U+000C and U+001C-U+001E too,
but those cost no valid record -- RFC 8259 forbids every unescaped character
below U+0020 inside a string, so the parser rejects such a record either way.
Each reader now splits on the newline alone; nothing else changed.

This is reachable through the documented workflow, not a hypothetical.
`docs/benchmarking.md` runs these files over shards this repo writes, and they
also write their own intermediate answers files with `ensure_ascii=False` (21
such writes across the seven) and read them straight back -- `answer` emits
`--output`, `evaluate` consumes it as `--answers`. That write is upstream code
this repo does not control, and the answers it serializes are model output
reproducing corpus text, so a raw separator reaches these readers regardless
of how this repo serializes its own shards. (It now escapes them: see
`ensure_ascii=True` in `mindbridge.benchmarks.aml.cli`, which closes the shard
hop but not this one.) The official
CL-Bench release (`tencent/CL-bench` revision
`b28a5832a09b0d96c0cf4c22e90d7c60ede25b80`) carries 343 bare U+2028
characters, so `pipeline.py answer --input <shard>` raised
`JSONDecodeError: Unterminated string` and CL-Bench could not be scored at
all.

The delta cannot move a score. It changes only which records get read, never
how a read record is judged: before it, affected records were lost or the run
died; after it, they are scored by upstream's unmodified logic. Silently
dropping records is what makes a score non-comparable, so this restores
comparability rather than spending it.

`tests/contracts/test_aml_pinned_sources.py` checks both halves of this file:
that every hash below matches, and — separately — that no vendored JSONL
reader has gone back to `splitlines()`. The second check exists because
re-pinning a fresh upstream copy would satisfy the first while silently
reinstating the bug.

```text
04ccace501f29a3ce808286bfbe5a9a34ed5ed30e42143e6289772b02bbb84f9  benchmarks/aml/api_config.py
1a790e5f567f691a53285cfb1bd447e831b05e1891a9806c6ceb532ea7a8d6e1  benchmarks/aml/pipelines/beam/pipeline.py
fea9b9d30946c28426e2d4815825cb72c48ae06404f010c69c49da9434e4a7a6  benchmarks/aml/pipelines/clbench/pipeline.py
ac35081e1368c8ebe6968abe54d1ba8bdb198a54804d35a2012d437f620c349d  benchmarks/aml/pipelines/locomo-refined/pipeline.py
b685e19c10b9c3da6a81bea5df1eed61601740ad92733f582a3bf4d95f8e97cf  benchmarks/aml/pipelines/longmemeval-s/pipeline.py
049ddf31800efd0b6d707b11191c048fb376f798ea563b97061771d3001c1c51  benchmarks/aml/pipelines/personamem/pipeline_v1.py
02e67bbca06ac5436fc9878b049d21644eea92866f60e9f38f9227234ed2aec0  benchmarks/aml/pipelines/personamem/pipeline_v2.py
1b8406c4edad105337e448a8a5609c6521a15e39e91b4fc10ced18785c198ac4  benchmarks/aml/pipelines/scriptmem/pipeline.py
```
