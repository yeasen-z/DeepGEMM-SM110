# DeepGEMM-SM110

Experimental **SM110 (NVIDIA Jetson AGX Thor)** support for DeepGEMM, plus a
record of exactly how far that support has been verified.

- **Base**: `sgl-project/DeepGEMM` **v0.2.3** (`e6f9bd6`)
- **Target**: NVIDIA Jetson AGX Thor Developer Kit — `sm_110`, compute capability **11.0**
- **Patched artifact**: `deep_gemm/_C.so` — this fork changes **host code only**, and `_C.so` embeds its build-time source path, so its md5 identifies *one particular build* (ours: `66e8ee99…`), not the patch. Verify your own build behaviourally, with §4.5.
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
| **Steady-state throughput vs Triton** (prefill, real data) | 41 req / 1.44 M tok | — | **Verified** — **1.47x** overall, 2.21x on ~4k requests |

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

### 4.3 Point the build at your environment

Two things must be right before you build: the CUDA install, and — more
importantly — **which Python builds the extension**.

```bash
export CUDA_HOME=/usr/local/cuda          # or e.g. /home/<user>/cuda_home_thor
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CXX=g++                            # a real C++20 g++, not nvcc

nvcc --version && $CXX --version          # sanity check
```

> #### Build with the Python you are going to install into
>
> `_C.so` is linked against torch's C++ ABI, and that ABI is **not** stable
> across torch releases: `c10::cuda::c10_cuda_check_implementation` gained a
> `CUDAErrorLogCapture*` parameter in torch 2.14. A `_C.so` built against 2.14
> fails to import under 2.13 with
> `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb`,
> and vice versa.
>
> `build_sgl_deep_gemm.sh` picks its interpreter with
> `PYTHON_EXE="${PYTHON_EXE:-$(which python3 || which python)}"`. If your `PATH`
> has more than one Python — very common when you also export `$CUDA_HOME/bin`,
> which often contains its own `python3` — the *wrong* one can be chosen and the
> resulting wheel is silently unusable in your target environment. Select it
> explicitly:
>
> ```bash
> which -a python3                                     # ALL of them, in PATH order
> export PYTHON_EXE=/path/to/your/env/bin/python       # pin the one you mean
>
> # Or simply put your env first and confirm what wins:
> export PATH="/path/to/your/env/bin:$PATH"
> which python3
> python3 -c 'import torch,sys; print(sys.version.split()[0], torch.__version__, torch.version.cuda)'
> ```
>
> The build script now prints the interpreter and torch version it will use, and
> refuses to run if that interpreter cannot import torch. Set
> `DG_SKIP_TORCH_CHECK=1` to silence the extra ABI probe.

### 4.4 Build and install

`_C.so` is the only compiled artifact — a plain `TORCH_LIBRARY` shared object
with no CPython dependency. **Two supported routes:**

> #### This repository ships no prebuilt `_C.so`, on purpose
>
> Build it yourself. A `_C.so` is only valid for an exact
> **(CPU architecture, torch ABI, CUDA major)** triple, so a binary we upload
> would be unusable for most readers:
>
> | Dimension | Our build | Consequence for someone else |
> |---|---|---|
> | CPU arch | `ARM aarch64` (Thor) | will not load on x86_64 at all |
> | torch ABI | `torch 2.14.1` (`c10_cuda_check_implementation` **with** `CUDAErrorLogCapture*`) | `undefined symbol` under torch ≤ 2.13 |
> | CUDA major | `libcudart.so.13` / `libcublas.so.13` | missing libraries under CUDA 12 |
>
> Both halves of that torch-ABI split are present on the development machine
> itself (`inscope` → 2.14.1, `prefillbench-thor` → 2.13.0), which is exactly
> why this is easy to get wrong. Since the runtime JIT already requires a CUDA
> toolchain on the target machine, building the host layer adds no new
> requirement. Building also avoids the checksum trap described below: a
> self-built `_C.so` legitimately has a different md5 from ours.

**Route A — build a wheel.** `build_sgl_deep_gemm.sh` initialises the
submodules, links the CUTLASS headers into `deep_gemm/include`, stages the
package, compiles `_C.so`, and produces a wheel whose import name is still
`deep_gemm`:

```bash
bash build_sgl_deep_gemm.sh
pip install --no-deps dist/sgl_deep_gemm-*.whl   # use the actual name it prints
```

#### Distributing the patch: the wheel version is `0.2.3+sm110`

The wheel's version comes from `sgl_deep_gemm/VERSION`, which this fork sets to
**`0.2.3+sm110`** — PEP 440 local version, so it stays comparable to the upstream
`0.2.3` it is patched from while being distinguishable from it. This is not
cosmetic; pip keys installation on the version:

| Installed | Wheel you install | What pip does |
|---|---|---|
| `0.2.3` | `0.2.3` (stock version) | **Silently does nothing** — "already installed with the same version as the provided wheel. Use `--force-reinstall`…" |
| `0.2.3` | `0.2.3+sm110` (**this fork**) | Installs normally; **no** `--force-reinstall` needed |

The first row is the trap: a wheel built at the stock `0.2.3` does **not**
replace an existing DeepGEMM, so the user keeps running the unpatched `_C.so`
while believing the install succeeded. The checked-in `0.2.3+sm110` avoids it.

`0.2.3+sm110` also still satisfies SGLang's pin `sgl-deep-gemm==0.2.3`, because
`==` ignores the local segment — so dependency resolution is unaffected, and it
sorts **above** `0.2.3` as expected. The resulting wheel is
`sgl_deep_gemm-0.2.3+sm110-py3-none-any.whl` (the `+` is kept literally in the
filename).

If you rebuild at a different version, the remaining options are:

1. **Force the reinstall** if you deliberately build with the stock `0.2.3`:

   ```bash
   pip install --force-reinstall --no-deps dist/sgl_deep_gemm-*.whl
   ```

   `sgl-deep-gemm` currently declares **no** runtime dependencies
   (`dependencies = []` in its `pyproject.toml`), so `--force-reinstall` alone
   would be harmless today. Keep `--no-deps` anyway: it is the belt-and-braces
   form that stays correct if the wheel later gains dependencies, and it stops
   pip resolving or upgrading anything you did not ask for. (It does **not**
   protect you against re-downloading torch — that is not a dependency here.)

2. **Skip the wheel entirely** — overwrite the extension by hand. Fewest moving
   parts for a single machine, and it cannot disturb pip's metadata at all:

   ```bash
   SP=$(python -c 'import deep_gemm,os;print(os.path.dirname(deep_gemm.__file__))')
   cp "$SP/_C.so" "$SP/_C.so.orig"     # keep the stock one for A/B
   cp /path/to/built/_C.so "$SP/_C.so"
   ```

   Note this replaces the binary only; if you also need the Python-side changes
   from a newer revision, this route will not bring them.

> **Don't use the md5 to recognise the patch.** `_C.so` embeds the absolute path
> of the sources it was compiled from (the assertion messages reference them, and
> they are visible via `strings`). Two builds of byte-identical sources in
> different directories therefore produce **different** md5s — we measured
> 2,045,576-byte binaries that differ only in that embedded path and hash as
> `66e8ee99…` (built in the source tree) vs `515dea72…` (built in a fresh clone).
> A build you make yourself will not reproduce our md5, so an md5 mismatch is not
> evidence of a bad build. Use the behavioural check in §4.5 instead: `bf16_gemm_nt`
> must run, and the stock library must fail with `Unsupported architecture`.

Whichever route you use, confirm you really got the patched build — do it
behaviourally, as in §4.5, not by md5 (see the note above).

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

That runtime difference — not a checksum — is the reliable way to tell the two
apart. `_C.so` embeds the absolute path of the sources it was built from (the
assertion messages point at them), so its md5 changes with the build directory:
the binary in this environment hashes as `66e8ee99e3c3ae7425b42713766a6e4a`
(path `/home/becool/Workspace/zmosg/src/DeepGEMM`, the stock one as
`ab82a63b329f5366c04ec1a0244402a9`), while a byte-identical build from a fresh
clone (`/tmp/sm110_fresh`) hashes as `515dea727c68e77b6b247d88188b3aff`. A
self-built library will legitimately differ.

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
covers the dense FP8 path only — not the grouped / MoE paths.

> The wall times in that cold-start run (506 s DeepGEMM vs 311 s Triton) include
> one-off JIT compilation and are **not** a performance comparison. A separate
> steady-state measurement on real data — **prefill only, 24 of 36 layers, 41
> requests / 1.44 M tokens** — puts DeepGEMM at **1.47x Triton overall** (2.21x on
> ~4k requests). See
> [docs/sm110-thor-support.md §5.6](docs/sm110-thor-support.md#56-steady-state-throughput).

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