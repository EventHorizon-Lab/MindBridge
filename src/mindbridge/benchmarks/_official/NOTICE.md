# Third-party scorer notices

The benchmark-only modules in this directory retain behavior from the following upstream
evaluators:

- MEMLENS, copyright 2026 Xiyu Ren, Zhaowei Wang, and the MemLens Authors.
- ATM-Bench, copyright 2026 Jingbiao Mei.
- Mem-Gallery, copyright its authors and contributors.

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

The LoCoMo-Refined scorer prompt and lexical behavior are adapted from
`mem-eval-suite/LoCoMo_refined@887091190789e8d6760e70b9edd696539923dc4f`, licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). MindBridge changes the transport
and result schema but preserves the evaluation behavior. Use of that scorer is subject to its
non-commercial restriction.

The M3-Agent scorer prompt is derived from
`ByteDance-Seed/m3-agent@0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c`, licensed under the Apache
License, Version 2.0; the repository root `LICENSE` contains that license.
