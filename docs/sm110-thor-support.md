# SM110 (NVIDIA Jetson Thor) Support

This document describes the experimental SM110 enablement carried by this fork,
how the SM100 kernel family is reused, exactly which files changed, how to build
and test it, and what the recorded verification results were.

- Base: `sgl-project/DeepGEMM` **v0.2.3** (`e6f9bd6`)
- Target: **NVIDIA Jetson AGX Thor Developer Kit**, `sm_110` / compute capability **11.0**
- Status: the dispatch path works and has been exercised **end to end** — SGLang
  serving `Qwen3-8B-FP8` on Thor with the DeepGEMM FP8 backend produces correct
  generation, and every DeepGEMM test case that has been run passed with zero
  assertion failures. The full test suite has **not** been run end to end on this
  part; per-section coverage is in [§5.1](#51-numerical-correctness). SGLang
  integration has one extra prerequisite
  (see [§6](#6-sglang-integration-prerequisite)).

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

Coverage therefore includes FP8 and FP4 (dense, grouped-contiguous, grouped-
masked, k-grouped), BF16 (dense, grouped, k-grouped), einsum, scaling-factor
layout handling, the lightning-indexer MQA scoring kernels, HyperConnection
prenorm, Mega Gate, Mega mHC, and Mega MoE.

### 2.2 Deliberately *not* changed

- `csrc/runtime/jit.hpp` — leaves `major == 10` alone. `major == 11` falls
  through to DeepJIT's own default, which is already the correct `"110f"`.
- `csrc/apis/sm90_mega.hpp` and `csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp`
  — these assert `arch_major == 9` for genuinely Hopper-only paths and must
  stay Hopper-only.
- All `arch_major == 12` branches — SM120 keeps its own separate kernel family.

### 2.3 The full patch

`git diff` of the change is the entire content of this section:

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
object with no CPython dependency (one build serves every Python version, though
it is linked against one PyTorch build).

```bash
# CUTLASS/DeepJIT submodules (host compile needs cutlass; JIT needs deep_jit)
git submodule update --init --recursive

python sgl_deep_gemm/build_ext.py . /path/to/site-packages/deep_gemm/_C.so
```

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
| Patched `_C.so` md5 | `66e8ee99e3c3ae7425b42713766a6e4a` |
| Stock `_C.so` md5 | `ab82a63b329f5366c04ec1a0244402a9` |

The JIT compiles at run time with `--gpu-architecture=sm_110f`, which is what
DeepJIT derives from major/minor 11/0.

> The CUDA `bin/` tree in that environment is assembled from the PyTorch CUDA
> wheels, so it does **not** contain `cuobjdump`. DeepJIT only requires it when
> SASS dumping is on (`JIT_DEBUG`/`JIT_DUMP_ASM`); all recorded runs left it off
> and only ever exercised the cubin + PTX path.

---

## 4. Running the test suite

```bash
export CUDA_HOME=/path/to/cuda CUDACXX=$CUDA_HOME/bin/nvcc
cd sgl_deep_gemm/tests
python -u test_bf16.py
python -u test_fp8_fp4.py
python -u test_layout.py
```

The `sgl_deep_gemm/tests/` copies are the ones used here; the 14 files under the
repository-root `tests/` are largely the same tests plus additional Mega MoE
variants (Hopper/TRT-LLM/SiTU coverage) that need extra dependencies.
`test_bf16.py` covers dense, m-grouped (contiguous and masked) and k-grouped
BF16 by sweeping shapes and asserting `calc_diff(d, ref_d) < 1e-5` against a
cuBLAS reference; `test_fp8_fp4.py` does the same for FP8/FP4 with
`quant_config.max_diff()`; `test_layout.py` checks the scale-factor
pack/transpose kernels for bit-exactness.

A `Perf (...)` line is emitted only after that shape's assertion has passed, so
the number of `Perf` lines is a direct progress/completeness signal.

---

## 5. Results

### 5.1 Numerical correctness

Every case that was run passed its assertion; **zero assertion failures and
zero tracebacks** were recorded across all suites. Coverage is stated
per *section* of each test file, because the suite is long-running and the
recorded runs were stopped part-way (a full `test_bf16.py` sweep takes many
minutes on a 20-SM part, and shares the GPU with other work).

Section sizes below are the exact generator yields, counted with the
generators themselves for major 11:

| Suite / section | Cases run | Cases in section | Assertion failures |
|---|---:|---:|---:|
| `test_bf16.py` → `Testing GEMM` (dense) | 84 | 84 | 0 |
| `test_bf16.py` → m-grouped contiguous | 48 | 48 | 0 |
| `test_bf16.py` → m-grouped masked | 8 | 32 | 0 |
| `test_bf16.py` → k-grouped contiguous | not run | 105 | — |
| `test_bf16.py` → cuBLASLt GEMM / SYRK / SYMM | not run | — | — |
| `test_fp8_fp4.py` → `Testing GEMM` | 26 | 42 | 0 |
| `test_fp8_fp4.py` → grouped sections | not run | — | — |
| `test_layout.py` → `Testing SF layout kernels` | 7 | 216 | 0 |
| `test_layout.py` → k-grouped / psum sections | not run | — | — |

A `Perf (...)` line is printed only after that case's assertion has already
passed, so "cases run with 0 failures" is directly evidenced by the `Perf`
count with no accompanying traceback. The sections marked *not run* are simply
not covered by the recorded runs — they are not known to fail.

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

### 5.2 End-to-end on a real model (SGLang + `Qwen3-8B-FP8`)

The strongest check: load a real FP8 checkpoint in SGLang on Thor and generate,
with the FP8 GEMM backend pointed at DeepGEMM, then repeat with the Triton
backend as a control. Both runs use the patched `_C.so` and
`SGLANG_ENABLE_JIT_DEEPGEMM=1`.

Setup: `Qwen3-8B-FP8` (offline snapshot), `dtype=bfloat16`, single device, greedy
decoding, four fixed prompts. `--fp8-gemm-backend` selects the backend.

| Backend | First prompt completion | Wall |
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
text, which is a far stronger signal than any single-kernel spot check. Wall time
is higher than Triton on this 20-SM part; see §5.5.

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

Relative to cuBLAS on the same Thor part, per-shape ratios from the suite were
roughly at parity for large-N shapes and below parity for some small/skinny
ones (e.g. `1.04x`, `1.02x`, `0.98x` on the measured BF16 NT shapes). SM110 is
a 20-SM mobile-class part, so absolute TFLOPS are low and these numbers should
be read as "the SM100 kernel family runs correctly here", not as a tuned
performance claim. No SM110-specific tuning was added.

End-to-end, the `Qwen3-8B-FP8` run took **506 s** with DeepGEMM vs **311 s**
with Triton (same four prompts, both including model load and JIT/compile
warm-up). That gap is not a fair steady-state comparison — the DeepGEMM run pays
a one-off JIT compile for the shapes it hits and was not pre-compiled with
`sglang.compile_deep_gemm`. Treat the number as "it runs correctly", not as a
perf verdict; correctness was the goal of this work.

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
- The SGLang side is not part of this repository, so the pairing between this
  `_C.so` and those two SGLang edits is a documented contract, not something CI
  can enforce.
- Mega Gate and Mega mHC dispatch through SM110 by the same rule, but their
  reference comparisons need TileKernels/TileLang and were not exercised here.
- `Qwen3-8B-FP8` end-to-end generation succeeded (§5.2), but that is a **dense**
  model: it exercises the dense block-FP8 GEMM path only. Grouped / MoE
  inference (and `DeepSeek-V3`-class models) was not run against this revision.
- The end-to-end run was not pre-compiled with `sglang.compile_deep_gemm`, so its
  wall time includes one-off JIT compilation and is not a performance result.
- If upstream adopts this, folding SM110 into an `is_sm100_family`-style
  predicate would be cleaner than enumerating `10 or 11` at 45 sites.