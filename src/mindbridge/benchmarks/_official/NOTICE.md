# Third-party benchmark notices

MindBridge's Apache-2.0 license does not replace the terms for the third-party benchmark material
described below.

## MIT evaluator material

- MEMLENS, copyright 2026 Xiyu Ren, Zhaowei Wang, and the MemLens Authors.
- ATM-Bench, copyright 2026 Jingbiao Mei.
- Mem-Gallery, copyright 2026 Yuanchen Bei.

Those components are provided under the MIT License:

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software
> and associated documentation files (the "Software"), to deal in the Software without
> restriction, including without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
> Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or
> substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
> BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
> NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
> DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Creative Commons evaluator material

The EgoTempo query and scorer instructions are adapted from
`google-research-datasets/egotempo@7022ba77b4d89f51cf34e499767995ccd5c90c7a`, by Chiara Plizzari,
Alessio Tonioni, Yongqin Xian, Ace Kulshrestha, and Federico Tombari. It is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). MindBridge changes the transport and
result parsing.

### LoCoMo Refined Attribution Notice

LoCoMo-Refined is a modified version of the LoCoMo release from Snap Research:

- Project: LoCoMo.
- Upstream repository: <https://github.com/snap-research/locomo>.
- Paper: "Evaluating Very Long-Term Conversational Memory of LLM Agents" (ACL 2024,
  arXiv:2402.17753).

The upstream LoCoMo material included in or adapted for LoCoMo-Refined is distributed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). LoCoMo-Refined corrects and
restructures benchmark data and adjusts the evaluator. MindBridge adapts its scorer transport and
result schema. Retain this notice, the license notice, and the modification statements when sharing
the benchmark or derivatives. Use is non-commercial unless the relevant rights holders grant
separate permission.

Source: `mem-eval-suite/LoCoMo_refined@887091190789e8d6760e70b9edd696539923dc4f`.

## Restricted evaluator material

The MM-Lifelong scorer instruction is copied from
`MM-Lifelong/MM-Lifelong@248aa82039a574e63a2e524746a7cd8f32330443/eval_acc.py`. The pinned
release permits academic research only, prohibits commercial use, and restricts distribution,
publication, copying, dissemination, and modification without prior approval. This notice grants
no additional permission; confirm authorization from the rights holder before redistributing the
instruction or a package that contains it.

## Apache evaluator material

The M3-Agent scorer prompt is derived from
`ByteDance-Seed/m3-agent@0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c`, licensed under the Apache
License, Version 2.0; the repository root `LICENSE` contains that license.
