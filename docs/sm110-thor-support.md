# SM110 (NVIDIA Jetson Thor) Support

This document describes the experimental SM110 enablement carried by this fork,
how the SM100 kernel family is reused, exactly which files changed, how to build
and test it, and what the recorded verification results were.

- Base: `sgl-project/DeepGEMM` **v0.2.3** (`e6f9bd6`)
- Target: **NVIDIA Jetson AGX Thor Developer Kit**, `sm_110` / compute capability **11.0**
- Status: the dispatch path works and has been exercised **end to end** — SGLang
  serving `Qwen3-8B-FP8` on Thor with the DeepGEMM FP8 backend produces correct
  generation, and a steady-state prefill measurement puts DeepGEMM **1.47x** faster
  than the Triton backend. The full suite **has** now been run to completion:
  `test_bf16.py`, `test_fp8_fp4.py` and `test_layout.py` finished with exit 0,
  **1074 cases, zero assertion failures, zero tracebacks** (per-section numbers in
  [§5.1](#51-numerical-correctness)). Two further sections are skipped by the suite
  itself but were run unguarded and pass — FP4 k-grouped (189 cases) and the
  k-grouped psum SF layout (224 cases); see §5.1.1. SGLang integration has one
  extra prerequisite (see [§6](#6-sglang-integration-prerequisite)).

---

## 1. Why a patch is needed

`csrc/` (the host layer compiled into `deep_gemm/_C.so`) picks a kernel
implementation purely from the device's **major** architecture number, read at
runtime through DeepJIT:

```cpp
// csrc/runtime/jit.hpp
const auto major = runtime->device.get_arch_major();
```

Upstream v0.2.3 only ever compares against `9`, `10`, and `12`. Thor reports
**major 11**, so every public API fell into the catch-all branch and aborted
before any kernel was selected:

```
RuntimeError: Assertion error (csrc/apis/gemm.hpp:225): Unsupported architecture
```

That is a *host-side dispatch* gap only. The device code is already
architecture-generic for this case:

| Layer | Why SM110 already works |
|---|---|
| DeepGEMM device code | SM100 kernels are gated on `__CUDA_ARCH__ >= 1000`; `sm_110` compiles with `__CUDA_ARCH__ == 1100` |
| CUTLASS / CuTe | `tcgen05`/UMMA atoms used by SM100 resolve the same way for SM110 |
| DeepJIT target selection | `device.hpp::get_arch()` returns `"110f"` for major 11 (`arch = major*10+minor`, then `+ "f"` for `major > 9`) |

So the entire extension is: **teach the host dispatcher that SM110 is an
SM100-class device.** No device kernel, no CUTLASS file, and no DeepJIT file is
modified.

---

## 2. The change

45 comparison sites across 10 host headers. All of them are the same idea —
widen an SM100 comparison to also accept major 11 — and take only four textual
shapes:

| # | Before | After | Sites |
|---|---|---|---|
| 1 | `arch_major == 10` | `(arch_major == 10 or arch_major == 11)` | 30 |
| 2 | `arch_major == 10 or arch_major == 12` | `(arch_major == 10 or arch_major == 11 or arch_major == 12)` | 7 |
| 3 | `get_arch_major() == 10` | `(get_arch_major() == 10 or get_arch_major() == 11)` | 7 |
| 4 | `arch_major == 9 or arch_major == 10` | `(arch_major == 9 or arch_major == 10 or arch_major == 11)` | 1 |

Two details that matter:

- **Parenthesise before combining.** `==` binds tighter than `and`/`or`, so a
  rewritten comparison is wrapped in parentheses before it is combined with any
  further predicate (e.g. `weights.scalar_type() == torch::kFloat or
  ((arch_major == 10 or arch_major == 11) and weights.scalar_type() ==
  torch::kBFloat16)`). This mirrors what SM120 already does upstream.
- **Order the compound forms first.** `arch_major == 10 or arch_major == 12`
  must be rewritten before the bare `arch_major == 10` rule, otherwise the
  bare rule fires inside it and produces a redundant/incorrect chain.

### 2.1 Changed files

| File | Sites | Area |
|---|---:|---|
| `csrc/apis/attention.hpp` | 16 | MQA logits: dense, sparse, paged; metadata builders |
| `csrc/apis/gemm.hpp` | 10 | FP8/FP4 dense + grouped GEMM, BF16 dense + grouped GEMM, k-grouped |
| `csrc/apis/einsum.hpp` | 5 | `bmk_bnk_mn`, `bhr_hdr_bhd`, `bhd_hdr_bhr`, FP8 BMM |
| `csrc/apis/layout.hpp` | 5 | SF transform into required layout (dense + k-grouped) |
| `csrc/utils/layout.hpp` | 3 | FP4 packed-AB checks, default recipe |
| `csrc/apis/mega_moe.hpp` | 2 | `fp8_fp4_mega_moe`, `bf16_mega_moe` (SM100 path) |
| `csrc/apis/mega_gate.hpp` | 1 | `bf16_mega_gate` |
| `csrc/apis/mega_mhc.hpp` | 1 | `mega_mhc` |
| `csrc/apis/hyperconnection.hpp` | 1 | `tf32_hc_prenorm_gemm` |
| `csrc/jit_kernels/heuristics/runtime.hpp` | 1 | contiguous-MK alignment (`224/32/32`, the SGLang-tuned SM100 values) |

Coverage therefore includes FP8 and the separately checked k-grouped FP4 path,
BF16 (dense, grouped, k-grouped), einsum, scaling-factor layout handling,
the lightning-indexer MQA scoring kernels, and HyperConnection. Mega Gate, Mega
mHC and Mega MoE remain untested because their reference dependencies are not
installed.

### 2.2 Deliberately *not* changed

- `csrc/runtime/jit.hpp` — leaves `major == 10` alone. `major == 11` falls
  through to DeepJIT's own default, which is already the correct `"110f"`.
- `csrc/apis/sm90_mega.hpp` and `csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp`
  — these assert `arch_major == 9` for genuinely Hopper-only paths and must
  stay Hopper-only.
- All `arch_major == 12` branches — SM120 keeps its own separate kernel family.

### 2.3 The full patch

The complete change is a single commit,
[`db919b4`](https://github.com/yeasen-z/DeepGEMM-thor-sm110/commit/db919b4); to see it
against upstream v0.2.3, use the
[`e6f9bd6...main` comparison](https://github.com/yeasen-z/DeepGEMM-thor-sm110/compare/e6f9bd6...main).
`git diff --stat` of the host layer is the entire content of this section:

```
 csrc/apis/attention.hpp                 | 32 ++++++++++++++++----------------
 csrc/apis/einsum.hpp                    | 10 +++++-----
 csrc/apis/gemm.hpp                      | 20 ++++++++++----------
 csrc/apis/hyperconnection.hpp           |  2 +-
 csrc/apis/layout.hpp                    | 10 +++++-----
 csrc/apis/mega_gate.hpp                 |  2 +-
 csrc/apis/mega_mhc.hpp                  |  2 +-
 csrc/apis/mega_moe.hpp                  |  4 ++--
 csrc/jit_kernels/heuristics/runtime.hpp |  2 +-
 csrc/utils/layout.hpp                   |  6 +++---
 10 files changed, 45 insertions(+), 45 deletions(-)
```

---

## 3. Build

`_C.so` is the only compiled artifact; it is a plain `TORCH_LIBRARY` shared
object with no CPython dependency, so one build serves every **Python** version.
It is **not** portable across torch releases or CPU architectures, though — see
§3.2 — and this repository deliberately ships no prebuilt copy.

```bash
# CUTLASS/DeepJIT submodules (host compile needs cutlass; JIT needs deep_jit)
git submodule update --init --recursive

python sgl_deep_gemm/build_ext.py . /path/to/site-packages/deep_gemm/_C.so
```

Build with the interpreter of the environment you will install into. The
`python` on your `PATH` is not necessarily that one.

`build_ext.py` shells out to `${CXX:-c++}` with `-std=c++20`; the compiler must
support C++20 `<format>`. It compiles **only the host layer**
(`csrc/torch_library_api.cpp`) and resolves `<cutlass/...>` from
`third-party/cutlass/include`, which is one of its include dirs — no extra
symlink is needed for this step. Install the result over the existing extension:

```bash
cp _C.so /path/to/site-packages/deep_gemm/_C.so
```

Alternatively `bash build_sgl_deep_gemm.sh` produces a wheel whose import name
is still `deep_gemm`.

### 3.1 Build/test environment used for the recorded results

| Component | Value |
|---|---|
| Board | NVIDIA Jetson AGX Thor Developer Kit (Tegra, aarch64) |
| GPU | `NVIDIA Thor`, compute capability **11.0** (`sm_110`), **20 SMs**, 128 GB unified |
| L4T / kernel | R39 (release 2.0), `6.8.12-1021-tegra` |
| OS | Ubuntu 24.04.4 LTS (noble), 14 CPU cores |
| Driver | `595.78` (nvidia-smi reports CUDA 13.2) |
| CUDA toolkit | **13.0** — `nvcc` V13.0.88 |
| Host compiler | GCC/G++ 13.3.0 |
| PyTorch | `2.14.1+cu130` |
| Python env | conda env `inscope`, Python 3.12 |
| DeepGEMM base | `v0.2.3` (`e6f9bd6`) |

> **No hash of `_C.so` is recorded here, and none should be.** The binary embeds
> the absolute path of the sources it was compiled from, so the same sources
> built in a different directory produce a binary of identical size that differs
> only in that embedded path — and therefore in its hash. A library you build
> yourself will never match someone else's; compare behaviour instead (§5.3).

The JIT compiles at run time with `--gpu-architecture=sm_110f`, which is what
DeepJIT derives from major/minor 11/0.

> The CUDA `bin/` tree in that environment is assembled from the PyTorch CUDA
> wheels, so it does **not** contain `cuobjdump`. DeepJIT only requires it when
> SASS dumping is on (`JIT_DEBUG`/`JIT_DUMP_ASM`); all recorded runs left it off
> and only ever exercised the cubin + PTX path.

### 3.2 Why `_C.so` is not portable (and why none is committed here)

A `_C.so` is only valid for a specific **(CPU architecture, torch ABI, CUDA
major)** combination. The recorded build is aarch64 + torch 2.14.1 + CUDA 13:

| Dimension | This build | What breaks otherwise |
|---|---|---|
| CPU architecture | `ARM aarch64` | will not load on x86_64 |
| torch ABI | torch **2.14.1**: `c10_cuda_check_implementation` takes a `CUDAErrorLogCapture*` | under torch ≤ 2.13, `import deep_gemm` raises `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb` |
| CUDA major | `libcudart.so.13`, `libcublas.so.13` | missing shared libraries under CUDA 12 |

The torch split was observed directly on the development machine: `inscope`
(torch 2.14.1) defines the signature **with** the extra argument, while
`prefillbench-thor` (torch 2.13.0) defines it **without**. A `_C.so` compiled
against one cannot be imported under the other, which is why:

* building it yourself is the supported path (the runtime JIT already needs a
  CUDA toolchain, so this adds no new requirement), and
* an md5 comparison against someone else's binary is meaningless — see the note
  in §3.1.

`build_sgl_deep_gemm.sh` therefore selects its interpreter explicitly
(`PYTHON_EXE="${PYTHON_EXE:-$(which python3 || which python)}"`), prints the
Python and torch versions it will build against, and fails early if that
interpreter cannot import torch.

The wheel version is `0.2.3+sm110` (`sgl_deep_gemm/VERSION`), a PEP 440 local
version. It sorts above the upstream `0.2.3` and still satisfies SGLang's pin
`sgl-deep-gemm==0.2.3`, because `==` ignores the local segment. That matters for
installation: pip treats a wheel at the *same* version as the installed one as a
no-op ("already installed with the same version… Use `--force-reinstall`"), so a
wheel left at `0.2.3` would silently fail to replace an existing DeepGEMM.

> **Host compile vs. runtime JIT — two separate header lookups.** The runtime
> JIT emits `#include <deep_gemm/impls/...>` and compiles it with
> `--include-path <package>/include`. 55 files under
> `deep_gemm/include/deep_gemm/` include `<cutlass/...>` or `<cute/...>`, so the
> **package's** `include/cutlass` and `include/cute` must exist when a kernel is
> compiled at run time. The wheel build copies them in; a source-tree run
> (`PYTHONPATH=$PWD`) needs the two symlinks created by hand:
>
> ```bash
> ln -sfn "$PWD/third-party/cutlass/include/cutlass" deep_gemm/include/cutlass
> ln -sfn "$PWD/third-party/cutlass/include/cute"    deep_gemm/include/cute
> ```

---

## 4. Running the test suite

```bash
export CUDA_HOME=/path/to/cuda CUDACXX=$CUDA_HOME/bin/nvcc
cd sgl_deep_gemm/tests
python -u test_bf16.py
python -u test_fp8_fp4.py
python -u test_layout.py
```

Everything here was run from `sgl_deep_gemm/tests/`, which is the superset: it
contains all 14 files of the repository-root `tests/` plus 13 more Mega MoE /
TRT-LLM / SiTU variants that need extra dependencies (TileKernels, TileLang,
DeepEP). The shared files are newer there as well — they are the copies that
carry the SM120 `skip:` guards and the `test_bf16_repeatability` /
`test_sm90_block32` entry points.

`test_bf16.py` covers dense, m-grouped (contiguous and masked) and k-grouped
BF16 by sweeping shapes and asserting `calc_diff(d, ref_d) < 1e-5` against a
cuBLAS reference; `test_fp8_fp4.py` does the same for FP8/FP4 with
`quant_config.max_diff()`; `test_layout.py` checks the scale-factor
pack/transpose kernels for bit-exactness.

A `Perf (...)` line is emitted only after that shape's assertion has passed, so
the number of `Perf` lines is a direct progress/completeness signal.

### 4.0 Two sections skip themselves on SM110 — and why that matters

`test_layout.py`'s third section returns before doing anything unless the arch
major is exactly 10:

```python
def test_k_grouped_psum_sf_layout_kernels() -> None:
    print('Testing k-grouped psum SF layout kernels:')
    if get_arch_major() != 10:
        print(' > Skipped (packed k-grouped SF only supported on SM100)')
        return
```

`test_fp8_fp4.py` gates FP4 the same way, on the option list rather than an
early return:

```python
test_options = [(torch.float8_e4m3fn, QuantConfig(), <fp8 kernel>)]
if arch_major == 10:
    test_options.append((torch.float4_e2m1fn_x2, QuantConfig((32, 32, True, True)),
                         deep_gemm.k_grouped_fp4_gemm_nt_contiguous))
```

Upstream treats both as SM100-only — but this fork's patch routes arch 11 into
the very same paths (`csrc/apis/layout.hpp`, and `csrc/utils/layout.hpp:64,73`
which accepts `kPackedFP4` for `10 or 11 or 12`).

So on Thor both are **reachable but unverified by the suite**: the guards are
statements about which architectures the *test* trusts, not about what the
*patched binary* will dispatch to. That is precisely the failure mode this
document is most concerned with — a kernel family that runs where nobody looked.

Note what is *not* gated: the generators. `enumerate_k_grouped_psum_sf_layout`
and the FP4 branch of `enumerate_k_grouped_contiguous` yield cases for any arch.
Only the test bodies were holding them back, which is why both could be checked
by removing the guard and keeping the assertions. See §5.1.1.

### 4.1 Build the SGLang distribution

The supported installation is `sgl-deep-gemm`, built by
`bash build_sgl_deep_gemm.sh`; the Python import name is `deep_gemm`.
Install the resulting wheel using the source-build instructions in README §4.
Version `0.2.3+sm110` satisfies SGLang's `sgl-deep-gemm==0.2.3` dependency.

Do not use root-level `pip install .` for SGLang. Its legacy `setup.py`
declares a different distribution, `deep_gemm`. Both distributions own files
under `deep_gemm/`, so installing both can replace shared package files while
leaving both distribution metadata directories installed. Uninstalling either
can damage the other. Use only `sgl-deep-gemm` in the SGLang environment.

The legacy build is outside this fork's supported scope. Upstream v0.2.3 has
both a schema-parser build failure and a pybind11/TORCH_LIBRARY entry-point
mismatch in that route; fixing it is unnecessary for the SGLang wheel build.
These failures do not mean the SM110 header changes are absent.

SGLang's local SM110 integration imports `deep_gemm` and queries the compiled
capability via `supported_arch_majors()` / `supports_arch(...)`, falling back
to the `sm110` version marker for older builds. This runtime capability check
is separate from pip's distribution-name dependency check.

### 4.2 `CUDA_HOME` is required even to import

`import deep_gemm` resolves a CUDA install at module-import time, because the
kernels are compiled by a JIT at run time. If no `CUDA_HOME`/`CUDA_PATH` is set,
no `nvcc` is on `PATH`, and `/usr/local/cuda` does not exist, the import fails.
This fork raises a `RuntimeError` that names the two directories you actually
need:

```
RuntimeError: Cannot locate a CUDA installation.
  ...
  Point CUDA_HOME at a directory that contains both of:

    <CUDA_HOME>/bin/nvcc                  the CUDA compiler
    <CUDA_HOME>/include/curand_kernel.h   the CUDA device headers

  On Jetson these are sometimes split across two prefixes. Either point
  CUDA_HOME at a directory that holds both (a symlink farm is fine), or
  add the device-header directory to CPATH.
```

### 4.3 Ask the build what it supports

The Python files and `_C.so` are installed separately and can disagree. The
capability query is therefore answered by the compiled binary
(`csrc/utils/arch.hpp` → `csrc/torch_library_api.cpp`), not by Python:

```python
import deep_gemm
deep_gemm.supported_arch_majors()   # (9, 10, 11, 12)
deep_gemm.supports_arch('sm110')    # True
```

It works without a CUDA context, so it can be called during import — which is
when frameworks decide whether to enable, for example, the UE8M0 FP8 scale path.
On a stock upstream `_C.so` the symbol does not exist and the call raises
`AttributeError` rather than returning a misleading `True`.

### 4.4 Diagnosing an install

```bash
python -m deep_gemm.doctor
```

Reports version, `_C.so` path/size/build time, CUDA home, `nvcc`, the device
header, torch, the supported arch set, the device, the JIT cache location and
entry count, and finally runs one 128×128×128 bf16 GEMM through the dispatcher.
Exits non-zero on a real problem. It deliberately does **not** print a hash of
`_C.so` — see §3.1.

---

## 5. Results

### 5.1 Numerical correctness

**All three suites ran to completion.** Every file exited 0, and every case
passed its assertion — **zero assertion failures, zero tracebacks**:

| Suite | Cases | Sections | Exit |
|---|---:|---:|---:|
| `test_bf16.py` | 449 | 7 | 0 |
| `test_fp8_fp4.py` | 353 | 4 | 0 |
| `test_layout.py` | 272 | 2 ran, 1 skipped | 0 |
| **Total** | **1074** | | |

Section sizes below are the exact generator yields on major 11, counted by
running the generators themselves — they are *not* copied from SM100
expectations, because several generators emit fewer cases on this arch
(`use_alpha_options` is `(False,)` unless the major is 10, and FP4 variants are
added only for major 10).

| Suite / section | Cases run | Cases in section | Assertion failures |
|---|---:|---:|---:|
| `test_bf16.py` → `Testing GEMM` (dense) | 84 | 84 | 0 |
| `test_bf16.py` → m-grouped contiguous | 48 | 48 | 0 |
| `test_bf16.py` → m-grouped masked | 32 | 32 | 0 |
| `test_bf16.py` → k-grouped contiguous | 105 | 105 | 0 |
| `test_bf16.py` → cuBLASLt GEMM | 84 | 84 | 0 |
| `test_bf16.py` → cuBLASLt batched SYRK | 32 | 32 | 0 |
| `test_bf16.py` → cuBLASLt batched SYMM | 64 | 64 | 0 |
| `test_fp8_fp4.py` → `Testing GEMM` (FP8 only on SM110) | 42 | 42 | 0 |
| `test_fp8_fp4.py` → m-grouped contiguous | 48 | 48 | 0 |
| `test_fp8_fp4.py` → m-grouped masked | 32 | 32 | 0 |
| `test_fp8_fp4.py` → k-grouped GEMM (FP8 only) | 231 | 231 | 0 |
| `test_layout.py` → `Testing SF layout kernels` | 216 | 216 | 0 |
| `test_layout.py` → k-grouped SF layout | 56 | 56 | 0 |
| `test_layout.py` → k-grouped psum SF layout | 0 (skipped) | 0 | — |
| `test_fp8_fp4.py` → FP4 k-grouped GEMM | 0 (skipped) | 0 | — |
| **Total** | **1074** | | **0** |

Two further sections are skipped by the suite but were run unguarded; they are
*not* part of the 1074 above (see §5.1.1):

| Unguarded check | Cases | Failures |
|---|---:|---:|
| k-grouped psum SF layout | 224 | 0 |
| FP4 k-grouped GEMM | 189 | 0 |

The `Perf` counts for the sections that *do* print one line per case are the
totals above; some sections print inside an inner loop, so the mapping from
`Perf` lines to enumerated cases is not always 1:1 — it was established by
reading each section's loop nesting, not assumed.

A `Perf (...)` line is printed only after that case's assertion has already
passed, so "cases run with 0 failures" is directly evidenced by the `Perf`
count with no accompanying traceback.

The psum section reports zero because upstream skips it on any arch but 10. See
§4.0 and the separate run below.

Independent spot checks against a dequantized reference:

| Check | Result |
|---|---|
| JIT `bf16_gemm_nt` | `rel_L2 = 1.66e-3` |
| JIT `fp8_gemm_nt` | `rel_L2 = 1.65e-3` |
| SGLang real path, `M=N=K=4096`, block 128 | `rel_L2 = 1.66e-3` |

The ~1.6e-3 residual is the expected FP8 block-quantization error, not a
kernel bug: the same figure is produced by the dequantized reference built from
the exact tensors the kernel consumed, and the pre-requant comparison differs by
two orders of magnitude (0.41 vs 0.0017).

### 5.1.1 Paths reachable on SM110 that the suite skips, checked unguarded

Two sections became reachable on Thor because of the patch while the upstream
test files decline to run them. In both cases the skip lives in the *test body*,
not in the generator, so the same assertions can be run with the guard removed.

**k-grouped psum SF layout** (`test_layout.py`). The section returns immediately
on any arch but 10 (§4.0), yet the patch routes arch 11 into that same
packed-psum path. Re-run with the guard removed (`psum_sm110_check.py`):

| Property | Value |
|---|---|
| Cases | 224 (112 tuples × `compact_sf` ∈ {False, True}) |
| Failures | 0 |
| Comparison | `torch.equal` against the PyTorch reference packing — bit-exact |
| `num_groups` covered | 8, 16, 72 |
| `gran_k` covered | 32, 128 |
| `k_alignment` covered | 32, 128, 256, 384 |
| `ks` forms covered | explicit aligned list, `None`, empty list |
| `use_padded_sf_layout` | both (`compact_sf=False` exercises the padded path) |

Both output widths are correct for every case: the compact form is compared on
its prefix (`[:sum(packed_sf_ks)]`) and the padded form on the whole tensor,
matching how upstream distinguishes them.

**FP4 k-grouped GEMM** (`test_fp8_fp4.py`). The test appends the FP4 option only
under `if arch_major == 10` (line 233), so on Thor that file tests FP8 alone.
But the patch accepts `kPackedFP4` for major 11 (`csrc/utils/layout.hpp:64,73`),
and the FP4 *cases* are not arch-gated —
`enumerate_k_grouped_contiguous` has a complete FP4 branch (K-major only,
`gran_k=32`, `k_alignment` 256/512/768) that yields for any arch. Only the test
body held it back.

Re-run with the guard removed (`fp4_sm110_check.py`), reusing the test's own
generators and assertions:

| Property | Value |
|---|---|
| Cases | 189 (7 grouped shapes × 3 `k_alignment` × 3 `cd_options` × 3 `ks` variants) |
| Failures | 0 |
| Exit code | 0 |
| Per-case checks | 20-iteration bitwise determinism; FP4 vs converted FP8 `< 1e-14`; direct output vs FP32 accumulation |
| Final check | `calc_diff < max_diff()` = 0.02 for FP4 |
| `num_groups` covered | 4, 8, 16 |
| `accumulate` | both |
| `out_dtype` | FP32, BF16 |

So FP4 is **verified** on SM110, but by a harness of ours rather than the
upstream suite, and only for the k-grouped path. Dense and m-grouped FP4, and
the Mega MoE FP4 path, are still unchecked.

### 5.2 End-to-end on a real model (SGLang + `Qwen3-8B-FP8`)

The strongest check: load a real FP8 checkpoint in SGLang on Thor and generate,
with the FP8 GEMM backend pointed at DeepGEMM, then repeat with the Triton
backend as a control. Both runs use the patched `_C.so` and
`SGLANG_ENABLE_JIT_DEEPGEMM=1`.

Setup: `Qwen3-8B-FP8` (offline snapshot), `dtype=bfloat16`, single device, greedy
decoding, four fixed prompts. `--fp8-gemm-backend` selects the backend.

| Backend | First prompt completion | Wall (first run, cold) |
|---|---|---:|
| **`deep_gemm`** | `' Paris. The capital of Germany is Berlin. The capital of Italy is Rome.'` | 506 s |
| `triton` (control) | `' Paris. The capital of Italy is Rome. The capital of Germany is Berlin.'` | 311 s |

All four completions from the DeepGEMM run are coherent and semantically
equivalent to the Triton control:

```
['The capital of France is']                  -> ' Paris. The capital of Germany is Berlin. The capital of Italy is Rome.'
['1 + 1 =']                                   -> ' 2, 2 + 1 = 3, 3 + '
['def fibonacci(n):']                         -> '\n    if n == 0:\n        return 0\n    elif n'
['The largest planet in the solar system is'] -> ' Jupiter, with a diameter of about 142,796 km'
```

(The Triton control returns the same facts; only the surface ordering and the
rounding of Jupiter's diameter differ, `142,000` vs `142,796` km.) This exercises
the path SGLang actually uses in production — FP8 block-scaled GEMM for every
linear, including the requantized UE8M0 weight scales — and produces correct
text, which is a far stronger signal than any single-kernel spot check.

> **The 506 s / 311 s figures above are cold-start numbers, not a performance
> result.** Both runs include model load plus one-off JIT compile/warm-up, and the
> DeepGEMM run had to JIT-compile every shape it touched for the first time. A
> later steady-state measurement — see [§5.6](#56-steady-state-throughput) —
> shows DeepGEMM is substantially **faster** than Triton on this part.

Note this was run with the model's own scales requantized to UE8M0, i.e. the
configuration described in §6 — that is the intended deployment shape.

### 5.3 Distinguishing a patched build from a stock one

Two recorded probes separate the builds. Both leave the CUDA context untouched.

**(a) A GPU-free metadata probe.** `reg_check.py` calls
`transform_sf_into_required_layout` with an `int32` scale, `gran_mn=1`,
`gran_k=128`. On SM110 the int branch only performs metadata checks, so an
unpatched host layer falls through to its catch-all before any kernel launch:

| `_C.so` | Result |
|---|---|
| stock (`ab82a63…`) | `UNSUPPORTED (Unknown SF transformation)` → patch is required |

**(b) A real kernel launch.** On the patched build the same style of check
actually runs a kernel, which is the stronger evidence:

```
JIT bf16_gemm_nt:                         SUCCESS rel_L2=0.001661
JIT fp8_gemm_nt (correct SF shapes):      SUCCESS rel_L2=0.001652
```

Two intermediate failed attempts along the way are worth recording because they
were *test-harness* errors, not kernel errors: with SF tensors shaped for a
different granularity the host aborts first at
`utils/layout.hpp:113: sf.size(-2) == ceil_div(mn, gran_mn)`. Once the SF shapes
matched the declared recipe, both kernels produced the correct ~1.6e-3 result.
This is also why the probe is only meaningful when the SF shapes are right.

### 5.4 Failure mode when only one half is deployed

If the SGLang side is fixed but the stock `_C.so` is left in place, the failure
is loud, not silent:

```
RuntimeError: Assertion error (csrc/apis/gemm.hpp:225): Unsupported architecture
```

This matters because SGLang can be configured to requantize weight scales to
UE8M0 and then fall back to a different GEMM backend; that combination would
compute wrong results. Instead the unsupported-architecture assertion fires and
no bad output is produced.

### 5.5 Performance

**Per-shape ratios from the kernel suite.** Relative to cuBLAS on the same Thor
part, per-shape ratios from the suite were roughly at parity for large-N shapes
and below parity for some small/skinny ones (e.g. `1.04x`, `1.02x`, `0.98x` on
the measured BF16 NT shapes). SM110 is a 20-SM mobile-class part, so absolute
TFLOPS are low and these numbers should be read as "the SM100 kernel family runs
correctly here". No SM110-specific tuning was added.

**Steady state end to end.** See §5.6 for the throughput measurement, which is
the meaningful end-to-end performance result.

### 5.6 Steady-state throughput

The 506 s / 311 s figures in §5.2 are cold-start and must not be used to compare
the backends. A separate steady-state measurement was therefore run, with both
backends measured on the **same real request sample** after warm-up.

> **Scope.** Real data from the PrefillBench-thor corpus (`wildchat` +
> `swe_rebench`), 41 requests, **1,440,227 tokens**, run through SGLang's official
> `sglang.benchmark.one_batch` path (one `ModelRunner.extend()` per request,
> timings bracketed by `torch.cuda.synchronize()`; 3 warm-up passes; no server,
> no chunked prefill). Prefill runs decoder blocks 0..23 and stops, so this
> measures the **prefill stage of a 24-of-36-layer model** — the absolute tok/s
> are therefore not whole-model prefill throughput, but the *ratio* between
> backends is valid because both were measured under identical truncation.
> Same 41 requests, same order, same token counts for both backends.

| Group | DeepGEMM | Triton | Ratio |
|---|---:|---:|---:|
| All | **5,331 tok/s** | 3,639 tok/s | **1.47x** |
| Short (~4k) | **12,018 tok/s** | 5,428 tok/s | **2.21x** |
| Long (~64k) | **5,165 tok/s** | 3,571 tok/s | **1.45x** |

Latency p50 (lower is better):

| Group | DeepGEMM | Triton | Ratio |
|---|---:|---:|---:|
| All | **10.49 s** | 15.90 s | 1.52x |
| Short (~4k) | **0.322 s** | 0.729 s | 2.26x |
| Long (~64k) | **12.13 s** | 17.65 s | 1.45x |

Total wall: DeepGEMM **270.3 s** vs Triton **396.0 s** for the identical
1,440,227-token workload. A single layer-level GEMM spot check on real checkpoint
weights (`gate_proj`, N=12288 K=4096) agrees in direction: 3.15–3.48x at
M ≥ 1024, though only 1.43x at M=256 and ~1.0x at M=16/64, where neither backend
is GEMM-bound.

**Read this as a prefill-only comparison on one part.** Decode was not measured
here, only 24 of 36 layers were executed, the sample is 41 of 20,000 available
requests, and both backends were measured sequentially rather than interleaved.

---

## 6. SGLang integration prerequisite

Routing SM110 into the SM100 kernels is necessary but **not sufficient** for
SGLang. The SM100 block-FP8 kernels require **UE8M0** (non-negative
power-of-two) scales, enforced by a device-side trap in the scale-packing
kernel:

```cpp
// deep_gemm/include/deep_gemm/impls/smxx_layout.cuh
// FP32 SFs must have a zero sign and mantissa (only the exponent is packed)
DG_TRAP_ONLY_DEVICE_ASSERT((values[j] & 0x807fffffu) == 0);
```

SGLang only requests the UE8M0 weight requant when `DEEPGEMM_SCALE_UE8M0` is
true, and on Thor its gate only listed `is_sm100` and SM `120/121`:

```python
# python/sglang/srt/layers/deep_gemm_wrapper/configurer.py
DEEPGEMM_SCALE_UE8M0 = ENABLE_JIT_DEEPGEMM and (
    get_platform().is_sm100 or get_device_sm() in (120, 121)   # 110 missing
)
```

So with a patched `_C.so` but an unpatched SGLang, arbitrary FP32 scales reach
the pack kernel and it traps with `CUDA_ERROR_LAUNCH_FAILED (719)`. Only
exactly-representable scales survive; a scale matrix of `1.5`, `1e30`, `-2.0`,
`nan`, or even `1.0 + 1 ulp` aborts, while `0.0`, `1.0`, `2.0`, `2^±20` are
fine. A second gate in `fp8_utils.py` also forced the backend to `triton` for
SM110, which would have made the DeepGEMM path unreachable even when `auto` was
requested.

Two SGLang-side edits are therefore required (kept in the SGLang checkout, not
in this repository):

```python
# 1. configurer.py — include SM110
DEEPGEMM_SCALE_UE8M0 = ENABLE_JIT_DEEPGEMM and (
    get_platform().is_sm100 or get_device_sm() in (110, 120, 121)
)

# 2. fp8_utils.py initialize_fp8_gemm_config() — only fall back when unsupported
elif (
    backend == "auto"
    and get_platform().is_sm110
    and not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
):
    backend = "triton"
```

With both applied, the end-to-end path resolves as expected:
`DEEPGEMM_SCALE_UE8M0 = True` → `should_deepgemm_weight_requant_ue8m0() = True`
→ weights requantized to `int32` packed UE8M0 → dispatch lands on
`deepgemm_w8a8_block_fp8_linear_with_fallback` → output finite with
`rel_L2 = 1.66e-3`.

Deployment requires **both** halves together: this patched `_C.so` *and* the
SGLang edits, with `SGLANG_ENABLE_JIT_DEEPGEMM=1`.

---

## 7. Limitations

- Results are from a single Thor developer kit; no other SM110 part was tested.
- The `pip install .` path is broken in upstream v0.2.3 itself (two independent
  defects, §4.1) and is left unfixed here, since fixing upstream's pybind11 /
  `TORCH_LIBRARY` split is a separate change from SM110 enablement.
- The SGLang side is not part of this repository, so the pairing between this
  `_C.so` and those two SGLang edits is a documented contract, not something CI
  can enforce.
- Mega Gate and Mega mHC dispatch through SM110 by the same rule, but their
  reference comparisons need TileKernels/TileLang and were not exercised here.
- FP4 is verified only for the **k-grouped** path, and only by our own harness
  (§5.1.1). Dense FP4, m-grouped FP4 and the Mega MoE FP4 path are unchecked.
- The main-sweep `_C.so` was replaced part-way through by an unrelated process,
  so `test_bf16.py` ran against one patched build and `test_fp8_fp4.py` /
  `test_layout.py` against another (same source, different embedded build path).
  See README §2.2 for why this does not affect the conclusions.
- `Qwen3-8B-FP8` end-to-end generation succeeded (§5.2), but that is a **dense**
  model: it exercises the dense block-FP8 GEMM path only. Grouped / MoE
  inference (and `DeepSeek-V3`-class models) was not run against this revision.
- The end-to-end **cold-start** run in §5.2 was not pre-compiled with
  `sglang.compile_deep_gemm`, so its wall time (506 s) includes one-off JIT
  compilation and is not a performance result. The steady-state measurement in
  §5.6 is the performance result, and it is prefill-only (24 of 36 layers).
- If upstream adopts this, folding SM110 into an `is_sm100_family`-style
  predicate would be cleaner than enumerating `10 or 11` at 45 sites.

---

## 9. Version handling

`sgl_deep_gemm/VERSION` (`0.2.3+sm110`) is the single source of truth. Both build
paths read it, so one source tree reports one version:

- `build_sgl_deep_gemm.sh` stages it into the wheel's `deep_gemm/VERSION`,
  spelled exactly as `setup.py` does, and the staged file is the one
  `deep_gemm.__version__` reads at runtime.
- `setup.py`'s `get_public_version()` reads it verbatim and returns it
  unchanged. Upstream appended a git revision or `+local`; that is dropped,
  because `0.2.3+sm110+local` has two local segments and is not a legal PEP 440
  version. A dirty tree is still reported as a warning.
- `deep_gemm/__init__.py`'s `_read_version()` resolves `VERSION` beside the
  module, falling back to the sibling `sgl_deep_gemm/VERSION` in a source tree,
  and to `0.0.0+unknown` if neither exists.

The value it replaces was `__version__ = '2.8.0'`, read by `setup.py` through a
regex and `ast.literal_eval`. That `2.8.0` belonged to an internal build series,
not to the `v0.2.x` release tags: its lineage is
`2.1.1 → 2.2.0 → 2.3.0 → 2.4.2 → 2.5.0 → 2.6.1 → 2.8.0`, with `2.8.0` entering in
`66081d4` (2026-09-10). Because it was the newest value on that axis, describing
it as "the original DeepGEMM version" would be wrong.

Verified: `python setup.py --version` and `setup.py egg_info`'s `PKG-INFO` both
report `0.2.3+sm110`, matching the build script. The supported path is unaffected,
since its `deep_gemm/__init__.py` is copied from `sgl_deep_gemm/__init__.py`,
which already read `VERSION`.
