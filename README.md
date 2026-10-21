# DeepGEMM-SM110

Experimental **SM110 (NVIDIA Jetson AGX Thor)** support for DeepGEMM, plus a
record of exactly how far that support has been verified.

- **Base**: `sgl-project/DeepGEMM` **v0.2.3** (`e6f9bd6`)
- **Target**: NVIDIA Jetson AGX Thor Developer Kit — `sm_110`, compute capability **11.0**
- **Patched artifact**: `deep_gemm/_C.so`, md5 `66e8ee99e3c3ae7425b42713766a6e4a`
- **Full write-up**: [docs/sm110-thor-support.md](docs/sm110-thor-support.md)
- **Upstream README** (this fork replaced it, kept for reference): [README.upstream.md](README.upstream.md)

---

## 1. What was implemented

DeepGEMM's host layer (`csrc/`, compiled into `deep_gemm/_C.so`) chooses a kernel
implementation purely from the device's **major** architecture number. Upstream
v0.2.3 only compares against `9`, `10`, and `12`; Thor reports **major 11**, so
every public API aborted before a kernel was selected:

```
RuntimeError: Assertion error (csrc/apis/gemm.hpp:225): Unsupported architecture
```

SM110 shares the SM100 `tcgen05` kernel family — the device code is gated on
`__CUDA_ARCH__ >= 1000` (true for `sm_110` = 1100), the CUTLASS UMMA/tcgen05
macros resolve identically, and DeepJIT already derives the correct target
(`"110f"` for major 11). So the entire change is teaching the **host
dispatcher** that SM110 is an SM100-class device. Device code, CUTLASS, and
DeepJIT are untouched.

**45 comparison sites across 10 host headers**, in four textual shapes:

| # | Before | After | Sites |
|---|---|---|---|
| 1 | `arch_major == 10` | `(arch_major == 10 or arch_major == 11)` | 30 |
| 2 | `arch_major == 10 or arch_major == 12` | `(arch_major == 10 or arch_major == 11 or arch_major == 12)` | 7 |
| 3 | `get_arch_major() == 10` | `(get_arch_major() == 10 or get_arch_major() == 11)` | 7 |
| 4 | `arch_major == 9 or arch_major == 10` | `(arch_major == 9 or arch_major == 10 or arch_major == 11)` | 1 |

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

Deliberately **not** changed: `csrc/runtime/jit.hpp` (major 11 already falls
through to the correct DeepJIT default), the `arch_major == 9` Hopper-only paths,
and all `arch_major == 12` branches (SM120 keeps its own kernel family).

---

## 2. Verification progress

**Read this before trusting the fork.** The 45 sites are mechanically uniform,
but each gates a *different* kernel family, so passing one proves little about
another. Two failure modes are in scope and neither is covered by "another
kernel worked":

1. A kernel that was previously unreachable may now be **compiled for
   `sm_110` for the first time** — PTXAS may reject it.
2. Worse, an `__CUDA_ARCH__ >= 1000` guard may hold on SM110 while the kernel's
   assumptions do not, producing a **silently wrong** result.

Verified so far (every case run passed its assertion — **zero assertion
failures, zero tracebacks**):

| Kernel path | Cases run | Cases in section | Status |
|---|---:|---:|---|
| **SGLang end-to-end, `Qwen3-8B-FP8`, DeepGEMM FP8 backend** | 4 prompts | 4 prompts | **Verified** — correct generation, matches Triton control |
| BF16 dense GEMM (`test_bf16.py` → `Testing GEMM`) | 84 | 84 | **Verified** |
| BF16 m-grouped contiguous | 48 | 48 | **Verified** |
| BF16 m-grouped masked | 8 | 32 | **Partially verified** |
| FP8 / FP4 dense GEMM (`test_fp8_fp4.py` → `Testing GEMM`) | 26 | 42 | **Partially verified** |
| SF layout pack/transpose (`test_layout.py`) | 7 | 216 | **Partially verified** |
| JIT `bf16_gemm_nt` vs dequantized ref | — | — | **Verified**, `rel_L2 = 1.66e-3` |
| JIT `fp8_gemm_nt` vs dequantized ref | — | — | **Verified**, `rel_L2 = 1.65e-3` |
| SGLang real layer path, block-FP8, `M=N=K=4096` | — | — | **Verified**, `rel_L2 = 1.66e-3` |
| Stock build fails loudly, not silently | — | — | **Verified** (`Unsupported architecture`) |

The end-to-end row is the strongest evidence here: it drives every linear in the
model through DeepGEMM's block-scaled FP8 kernel, including UE8M0-requantized
weight scales, and yields coherent — not merely finite — text.

**Not yet covered — unproven, not known-bad:** k-grouped BF16 (105 cases),
the remaining FP8/FP4 GEMM and grouped sections, the k-grouped and psum SF
layout sections, the cuBLASLt reference sections, and the entire
`attention` / `einsum` / `hyperconnection` / `mega_gate` / `mega_mhc` /
`mega_moe` API families.

**Requires its own verification before being enabled on Thor:** Mega MoE (its
reference needs TileKernels/TileLang), Mega Gate, Mega mHC, the
lightning-indexer MQA scoring kernels, HyperConnection, and FP4 paths. These
dispatch through SM110 by the same rule but have not been numerically checked.

The suite has **not** been run end to end on this part; per-section coverage
above is the authoritative statement. See
[docs/sm110-thor-support.md §5](docs/sm110-thor-support.md) for the raw numbers.

Scope note: `Qwen3-8B-FP8` is a dense model, so the end-to-end run exercised only
the **dense** block-FP8 GEMM path. Grouped / MoE paths (the `m-grouped` and
`k-grouped` sections, Mega MoE) are **not** covered by that run, which is why
they are listed separately above.

---

## 3. Test device

All recorded results come from this machine and no other:

| Component | Value |
|---|---|
| Board | NVIDIA Jetson AGX Thor Developer Kit (Tegra, aarch64) |
| GPU | `NVIDIA Thor`, compute capability **11.0** (`sm_110`), **20 SMs**, 128 GB unified |
| L4T / kernel | R39 (release 2.0), `6.8.12-1021-tegra` |
| OS | Ubuntu 24.04.4 LTS, 14 CPU cores |
| Driver | `595.78` (nvidia-smi reports CUDA 13.2) |
| CUDA toolkit | **13.0** — `nvcc` V13.0.88 |
| Host compiler | GCC/G++ 13.3.0 |
| PyTorch | `2.14.1+cu130` |
| Python env | conda env `inscope`, Python 3.12 |

JIT compiles at run time with `--gpu-architecture=sm_110f`.

---

## 4. Build and run

### 4.1 Prerequisites

- An **SM110 / SM100 / SM90** GPU. On Thor: driver `595.78`+ and CUDA **13.0**+.
- **CUDA Toolkit 12.9 or newer** with `nvcc` on `PATH`.
- A **C++20 compiler with `<format>` support** (GCC 13 works; the toolkit's own
  `c++` is not enough on Jetson).
- `libdw-dev` — required by the runtime JIT:
  ```bash
  sudo apt-get install -y libdw-dev
  ```
- **PyTorch** matching your CUDA. On Thor the working combination is
  `torch 2.14.1+cu130`.
- `git` with submodule access to `github.com/NVIDIA/cutlass` and
  `github.com/deepseek-ai/DeepJIT`.

### 4.2 Get the source and submodules

```bash
git clone --recursive https://github.com/yeasen-z/DeepGEMM-SM110.git
cd DeepGEMM-SM110

# If you cloned without --recursive, or the submodules are stale:
git submodule update --init --recursive
```

Both submodules are mandatory: `third-party/cutlass` supplies the CuTe UMMA
headers, and `third-party/deep_jit` supplies the JIT runtime headers.

### 4.3 Point the build at your CUDA install

The build reads `CUDA_HOME` / `CUDA_PATH` and falls back to
`dirname(dirname(which nvcc))`:

```bash
export CUDA_HOME=/usr/local/cuda          # or e.g. /home/<user>/cuda_home_thor
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export CXX=g++                            # a real C++20 g++, not nvcc

nvcc --version && $CXX --version          # sanity check
```

### 4.4 Build and install

`_C.so` is the only compiled artifact — a plain `TORCH_LIBRARY` shared object
with no CPython dependency. **Two supported routes:**

**Route A — build a wheel (recommended).** `build_sgl_deep_gemm.sh` initialises
the submodules, links the CUTLASS headers into `deep_gemm/include`, stages the
package, compiles `_C.so`, and produces a wheel whose import name is still
`deep_gemm`:

```bash
bash build_sgl_deep_gemm.sh
pip install dist/sgl_deep_gemm-*.whl      # use the actual name it prints
```

**Route B — compile `_C.so` in place.** Faster when iterating on `csrc/`;
you compile straight over the installed extension:

```bash
SP=$(python -c 'import deep_gemm,os;print(os.path.dirname(deep_gemm.__file__))')
python sgl_deep_gemm/build_ext.py "$PWD" "$SP/_C.so"
```

`build_ext.py` shells out to `${CXX:-c++}` with `-std=c++20`. It compiles
`csrc/torch_library_api.cpp` — the *host* layer only — and finds CUTLASS through
`-I third-party/cutlass/include`, which is already in its include list, so this
step needs **no** extra links.

Back up the stock extension before overwriting it, so you can A/B the two:

```bash
cp "$SP/_C.so" "$SP/_C.so.orig"
```

> **Two different header lookups — don't confuse them.**
> `build_ext.py` compiles the host layer and resolves `<cutlass/...>` from
> `third-party/cutlass/include`. The **runtime JIT** is separate: it emits
> `#include <deep_gemm/impls/sm100_bf16_gemm.cuh>` and compiles it with
> `--include-path <package>/include`. Since 55 files under
> `deep_gemm/include/deep_gemm/` pull in `<cutlass/...>` / `<cute/...>`, the
> package's `include/cutlass` and `include/cute` must exist **at run time**.
> Route A copies them into the wheel (`build_sgl_deep_gemm.sh` lines 55-56), so a
> `pip install` of the wheel is self-contained. If you instead run `deep_gemm`
> straight from a source checkout (`PYTHONPATH=$PWD`), create them yourself:
>
> ```bash
> ln -sfn "$PWD/third-party/cutlass/include/cutlass" deep_gemm/include/cutlass
> ln -sfn "$PWD/third-party/cutlass/include/cute"    deep_gemm/include/cute
> ```

### 4.5 Verify the patch is actually live

`get_arch_major()` must return **11** and the extension must expose the SM100
routing path. The strongest quick check is a real kernel launch (run it inside
the Python environment that has both `torch` and `deep_gemm` installed):

```bash
python -c "
import torch, deep_gemm
print('cap', torch.cuda.get_device_capability())
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda')
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda')
d = torch.empty(4096, 4096, dtype=torch.bfloat16, device='cuda')
deep_gemm.bf16_gemm_nt(a, b, d)
print('OK', float(d.abs().mean()))
"
```

Expected on Thor with the patched build:

```
cap (11, 0)
OK 51.0
```

With the stock `_C.so` this dies with
`RuntimeError: Assertion error (csrc/apis/gemm.hpp:225): Unsupported architecture`.
The patched `_C.so` md5 is `66e8ee99e3c3ae7425b42713766a6e4a`; the stock one is
`ab82a63b329f5366c04ec1a0244402a9`.

### 4.6 Run the test suites

```bash
cd sgl_deep_gemm/tests
python -u test_bf16.py
python -u test_fp8_fp4.py
python -u test_layout.py
```

A `Perf (...)` line is printed only after that case's assertion has passed, so
the number of `Perf` lines is a direct completeness signal. These suites are
long (the FP8 sweep is heavy) and the JIT compiles each shape on first use;
`test_bf16.py` alone takes many minutes on a 20-SM part. To run one subset,
call the individual test function instead of the whole file.

> The repo-root `tests/` directory contains the same suites plus additional
> Mega MoE / TRT-LLM / SiTU variants that need extra dependencies
> (TileKernels, TileLang, DeepEP). The `sgl_deep_gemm/tests/` copies are the
> ones used for the results recorded here.

---

## 5. End-to-end result (SGLang + `Qwen3-8B-FP8`)

The strongest evidence that the patch is useful, not merely harmless: SGLang
serving a real FP8 checkpoint on Thor with DeepGEMM as the FP8 GEMM backend.

```bash
export CUDA_HOME=/path/to/cuda
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export PYTHONPATH=/path/to/sglang/python

# DeepGEMM backend
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B-FP8 \
    --fp8-gemm-backend deep_gemm \
    --dtype bfloat16

# Control: same model, Triton backend
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B-FP8 \
    --fp8-gemm-backend triton \
    --dtype bfloat16
```

On Thor with `--fp8-gemm-backend deep_gemm`, all four recorded completions were
coherent and semantically matched the Triton control:

```
['The capital of France is']                  -> ' Paris. The capital of Germany is Berlin. The capital of Italy is Rome.'
['1 + 1 =']                                   -> ' 2, 2 + 1 = 3, 3 + '
['def fibonacci(n):']                         -> '\n    if n == 0:\n        return 0\n    elif n'
['The largest planet in the solar system is'] -> ' Jupiter, with a diameter of about 142,796 km'
```

This drives every linear in the model through DeepGEMM's block-scaled FP8 kernel,
including UE8M0-requantized weight scales. `Qwen3-8B-FP8` is **dense**, so this
covers the dense FP8 path only — not the grouped / MoE paths. Full details,
including the Triton wall-clock comparison, are in
[docs/sm110-thor-support.md §5.2](docs/sm110-thor-support.md).

---

## 6. SGLang integration prerequisite

Routing SM110 into the SM100 kernels is necessary but **not sufficient** for
SGLang. The SM100 block-FP8 kernels require **UE8M0** scales, and SGLang gates
that on a device check that excludes SM110 — without the two SGLang-side edits,
arbitrary FP32 scales reach the pack kernel and trap with
`CUDA_ERROR_LAUNCH_FAILED (719)`. The exact edits and the failure mode are
documented in
[docs/sm110-thor-support.md §6](docs/sm110-thor-support.md); they live in the
SGLang tree, not here.

---

## 7. Fork branch notes

This fork inherits the conventions of the `sgl-project/DeepGEMM` fork:

- `main` tracks upstream updates and should stay clean.
- `dev` is the active SGL integration branch and can be force-updated during rebases.
- `release/*` branches are archives for past wheel releases.

> In **this** fork the SM110 enablement is carried on `main` directly; the
> `dev` / `release/*` conventions above describe the upstream fork layout.

---

## 8. Upstream project

This fork replaces the upstream DeepGEMM README with the page you are reading.
The original is preserved verbatim at
[README.upstream.md](README.upstream.md) and upstream remains
[sgl-project/DeepGEMM](https://github.com/sgl-project/DeepGEMM).