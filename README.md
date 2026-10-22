# DeepGEMM for SM110 (Thor)

Experimental **SM110 (NVIDIA Jetson AGX Thor)** support for DeepGEMM, plus a
record of exactly how far that support has been verified.

- **Base**: `sgl-project/DeepGEMM` **v0.2.3** (`e6f9bd6`)
- **Target**: NVIDIA Jetson AGX Thor Developer Kit — `sm_110`, compute capability **11.0**
- **Implementation**: the dispatch change is a single commit, [`db919b4`](https://github.com/yeasen-z/DeepGEMM-thor-sm110/commit/db919b4) — 45 sites in 10 host headers. A later commit adds packaging, diagnostics and a capability query (§7); it does not touch dispatch.
- **Full diff vs upstream v0.2.3**: [`e6f9bd6...main`](https://github.com/yeasen-z/DeepGEMM-thor-sm110/compare/e6f9bd6...main)
- **All changes in this fork**: [`1853080...main`](https://github.com/yeasen-z/DeepGEMM-thor-sm110/compare/1853080...main)
- **Patched artifact**: `deep_gemm/_C.so` — this fork ships **no prebuilt binary**, on purpose: `_C.so` is tied to your CPU arch, your torch C++ ABI and your CUDA major, and it embeds its build-time source path, so no single binary is portable. Build it yourself (§4.4) and verify it behaviourally (§4.5).
- **Full write-up**: [docs/sm110-thor-support.md](docs/sm110-thor-support.md)
- **Upstream README** (this fork replaced it, kept for reference): [README.upstream.md](README.upstream.md)

> ### Build for SGLang
>
> This fork supports the `sgl-deep-gemm` distribution, imported as `deep_gemm`.
> Build with `bash build_sgl_deep_gemm.sh`, then install the resulting wheel
> as described in §4.4. Its version is `0.2.3+sm110`, which satisfies
> SGLang's dependency `sgl-deep-gemm==0.2.3`.
>
> **Do not run `pip install .` for SGLang, and do not install the `deep_gemm`
> distribution alongside `sgl-deep-gemm`.** They have different distribution
> names but own the same `deep_gemm/` package directory. Pip can record both
> as installed even when their package files have been replaced; uninstalling
> one can remove files required by the other.
>
> The root `setup.py` is retained from upstream and is not a supported build
> route here. Version readers use `sgl_deep_gemm/VERSION` consistently; this
> does not make the legacy distribution a SGLang installation route.

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

**45 comparison sites across 10 host headers**, in four textual shapes. The whole
change is one commit — [`db919b4`](https://github.com/yeasen-z/DeepGEMM-thor-sm110/commit/db919b4),
45 insertions / 45 deletions:

| # | Before | After | Sites |
|---|---|---|---|
| 1 | `arch_major == 10` | `(arch_major == 10 or arch_major == 11)` | 30 |
| 2 | `arch_major == 10 or arch_major == 12` | `(arch_major == 10 or arch_major == 11 or arch_major == 12)` | 7 |
| 3 | `get_arch_major() == 10` | `(get_arch_major() == 10 or get_arch_major() == 11)` | 7 |
| 4 | `arch_major == 9 or arch_major == 10` | `(arch_major == 9 or arch_major == 10 or arch_major == 11)` | 1 |

| File | Sites | Area |
|---|---:|---|
| [`csrc/apis/attention.hpp`](https://github.com/yeasen-z/DeepGEMM-thor-sm110/blob/main/csrc/apis/attention.hpp) | 16 | MQA logits: dense, sparse, paged; metadata builders |
| [`csrc/apis/gemm.hpp`](https://github.com/yeasen-z/DeepGEMM-thor-sm110/blob/main/csrc/apis/gemm.hpp) | 10 | FP8/FP4 dense + grouped GEMM, BF16 dense + grouped GEMM, k-grouped |
| [`csrc/apis/einsum.hpp`](https://github.com/yeasen-z/DeepGEMM-thor-sm110/blob/main/csrc/apis/einsum.hpp) | 5 | `bmk_bnk_mn`, `bhr_hdr_bhd`, `bhd_hdr_bhr`, FP8 BMM |
| [`csrc/apis/layout.hpp`](https://github.com/yeasen-z/DeepGEMM-thor-sm110/blob/main/csrc/apis/layout.hpp) | 5 | SF transform into required layout (dense + k-grouped) |
| [`csrc/utils/layout.hpp`](https://github.com/yeasen-z/DeepGEMM-thor-sm110/blob/main/csrc/utils/layout.hpp) | 3 | FP4 packed-AB checks, default recipe |
| [`csrc/apis/mega_moe.hpp`](https://github.com/yeasen-z/DeepGEMM-thor-sm110/blob/main/csrc/apis/mega_moe.hpp) | 2 | `fp8_fp4_mega_moe`, `bf16_mega_moe` (SM100 path) |
| [`csrc/apis/mega_gate.hpp`](https://github.com/yeasen-z/DeepGEMM-thor-sm110/blob/main/csrc/apis/mega_gate.hpp) | 1 | `bf16_mega_gate` |
| [`csrc/apis/mega_mhc.hpp`](https://github.com/yeasen-z/DeepGEMM-thor-sm110/blob/main/csrc/apis/mega_mhc.hpp) | 1 | `mega_mhc` |
| [`csrc/apis/hyperconnection.hpp`](https://github.com/yeasen-z/DeepGEMM-thor-sm110/blob/main/csrc/apis/hyperconnection.hpp) | 1 | `tf32_hc_prenorm_gemm` |
| [`csrc/jit_kernels/heuristics/runtime.hpp`](https://github.com/yeasen-z/DeepGEMM-thor-sm110/blob/main/csrc/jit_kernels/heuristics/runtime.hpp) | 1 | contiguous-MK alignment (`224/32/32`, the SGLang-tuned SM100 values) |

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

**These are complete runs, not samples.** All three files ran to the end with
exit status 0 — **1074 cases, zero assertion failures, zero tracebacks** — and
every section hit its expected case count exactly:

| Suite | Cases | Exit |
|---|---:|---|
| `test_bf16.py` | 449 | 0 |
| `test_fp8_fp4.py` | 353 | 0 |
| `test_layout.py` | 272 | 0 |

Per section (`Perf` lines = cases that passed their assertion):

| Kernel path | Cases run | Cases in section | Status |
|---|---:|---:|---|
| BF16 dense GEMM | 84 | 84 | **Verified** |
| BF16 m-grouped, contiguous | 48 | 48 | **Verified** |
| BF16 m-grouped, masked | 32 | 32 | **Verified** |
| BF16 k-grouped, contiguous | 105 | 105 | **Verified** |
| cuBLASLt GEMM (reference) | 84 | 84 | **Verified** |
| cuBLASLt batched SYRK | 32 | 32 | **Verified** |
| cuBLASLt batched SYMM | 64 | 64 | **Verified** |
| FP8 dense GEMM | 42 | 42 | **Verified** |
| FP8 m-grouped, contiguous | 48 | 48 | **Verified** |
| FP8 m-grouped, masked | 32 | 32 | **Verified** |
| FP8 k-grouped GEMM | 231 | 231 | **Verified** |
| SF layout pack/transpose | 216 | 216 | **Verified** |
| k-grouped SF layout | 56 | 56 | **Verified** |
| k-grouped psum SF layout | 224 (unguarded) | 224 | **Verified**, §2.1 |
| FP4 k-grouped GEMM | 189 (unguarded) | 189 | **Verified**, §2.1 |
| FP4 dense / m-grouped / Mega MoE | 0 | — | **not tested** |
| JIT `bf16_gemm_nt` vs dequantized ref | — | — | **Verified**, `rel_L2 = 1.66e-3` |
| JIT `fp8_gemm_nt` vs dequantized ref | — | — | **Verified**, `rel_L2 = 1.65e-3` |
| SGLang real layer path, block-FP8, `M=N=K=4096` | — | — | **Verified**, `rel_L2 = 1.66e-3` |
| **SGLang end-to-end, `Qwen3-8B-FP8`** | 4 prompts | 4 prompts | **Verified** — coherent text, matches Triton control |
| Stock build fails loudly, not silently | — | — | **Verified** (`Unsupported architecture`) |
| **Steady-state throughput vs Triton** (prefill, real data) | 41 req / 1.44 M tok | — | **Verified** — **1.47x** overall, 2.21x on ~4k requests |

The end-to-end row is the strongest evidence here: it drives every linear in the
model through DeepGEMM's block-scaled FP8 kernel, including UE8M0-requantized
weight scales, and yields coherent — not merely finite — text.

**Two kernel paths are reachable on SM110 but skipped by the original test
bodies**: FP4 is omitted from the option list, and psum returns early unless the
major is 10. Both were exercised separately using the existing generators and
correctness checks, without changing the real device architecture:

- **FP4 (k-grouped).** `test_fp8_fp4.py` only appends the FP4 option under
  `if arch_major == 10` (line 233), so on Thor the file enumerates FP8 alone.
  But `enumerate_k_grouped_contiguous` has a complete FP4 branch (K-major only,
  `gran_k` 32 with `k_alignment` 256/512/768) that is *not* arch-gated, and
  `csrc/utils/layout.hpp` accepts `kPackedFP4` for major 11. Re-ran the FP4
  half of that test unguarded: **189 cases, 0 failures** — including bitwise
  determinism over 20 iterations per case, FP4-vs-converted-FP8 agreement to
  `1e-14`, and direct-output agreement with FP32 accumulation. See §2.1.
- **`test_layout.py`'s psum SF section.** Skips itself on any arch but 10 via
  `if get_arch_major() != 10`. Re-ran unguarded: **224 cases, 0 failures**,
  bit-exact against the PyTorch reference. See §2.1.

**Not yet covered — unproven, not known-bad:** Mega Gate, Mega mHC, Mega MoE,
and dense / m-grouped FP4. The attention, einsum and hyperconnection test files
have now passed as whole files — see §2.3.

**Requires its own verification before being enabled on Thor:** Mega MoE (its
reference needs TileKernels/TileLang), Mega Gate, Mega mHC, and dense /
m-grouped FP4. The k-grouped FP4 path is verified separately above.

See [docs/sm110-thor-support.md §5](docs/sm110-thor-support.md) for the raw
numbers.

### 2.1 Paths reachable on SM110 that the suite refuses to test

Two kernel paths became reachable on Thor *because of this patch*, while the
upstream test files decline to exercise them. Both were re-run with the guard
removed, and both pass.

**k-grouped psum scale-factor layout.** `test_layout.py` returns before doing
anything unless the arch major is exactly 10 — but `csrc/apis/layout.hpp` routes
arch 11 into that same packed-psum path (its two predicates are still commented
`FP32 on SM100` / `Pre-packed UE8M0 ... on SM100`, and now read
`arch_major == 10 or arch_major == 11`). The guard describes what the *test*
trusts, not what the *patched binary* will dispatch to.

Re-ran the same assertions with the guard removed: **224 cases, bit-exact
against the PyTorch reference, 0 failures**, covering `compact_sf` in both
states, three `ks` forms (explicit aligned, `None`, empty), `num_groups` of
8/16/72, `gran_k` of 32 and 128, and `k_alignment` of 32/128/256/384.

**FP4 (k-grouped GEMM).** `test_fp8_fp4.py:233` appends the FP4 option only when
`arch_major == 10`, so on Thor that file tests FP8 alone — yet the patch accepts
`kPackedFP4` for major 11 (`csrc/utils/layout.hpp:64,73`:
`arch_major == 10 or arch_major == 11 or arch_major == 12`). Crucially the FP4
*cases* are not arch-gated: `enumerate_k_grouped_contiguous` has a full FP4
branch (K-major only, `gran_k=32`, `k_alignment` 256/512/768) that runs for any
arch. Only the test body was holding it back.

Re-ran the FP4 half of that test unguarded: **189 cases, 0 failures**, exit 0.
Each case ran 20-iteration bitwise determinism against an initial kernel call.
Accumulation cases additionally checked FP4-vs-converted-FP8 agreement below
`1e-14` and `calc_diff(d, ref_d) < 0.02`; direct-output cases checked agreement
with FP32 accumulation. Covered 7 grouped shapes
(`num_groups` 4/8/16), all three `k_alignment` values, `accumulate` both ways,
and `out_dtype` FP32/BF16.

So FP4 is **verified, not merely reachable** — but by a harness of ours, not by
the upstream suite, and only for the k-grouped path. The dense/`m-grouped` FP4
paths and the Mega MoE FP4 path remain unchecked.

Scope note: `Qwen3-8B-FP8` is a dense model, so the end-to-end run exercised only
the **dense** block-FP8 GEMM path. Grouped / MoE paths are **not** covered by
that run, which is why they are listed separately above.

### 2.2 One caveat about the run itself

The installed package changed during the sweep (its files have an 11:03
modification time). Thus this was not a controlled, immutable-build run:
BF16 started before the change, while FP8 and layout started afterwards.
All three processes finished successfully, and the later build passed 625
cases, demonstrating SM110 support for the exercised paths. We did not retain
both binaries or establish how the replacement occurred, so we cannot claim
they differ only in embedded paths or prove that an atomic replacement left
the first process unaffected. A single-build rerun is needed for strict
artifact-level reproducibility.

### 2.3 The other API families

Beyond the three main files, the remaining runnable test files were executed as
whole files against the same patched build. All exited 0 with **zero
tracebacks**:

| File | Cases | Exit | Notes |
|---|---:|---:|---|
| `test_attention.py` | 4 sections | 0 | GEMM skip-head-mid + MQA logits (dense/paged/sparse) |
| `test_einsum.py` | 60 | 0 | BF16 and FP8 batched matmuls |
| `test_hyperconnection.py` | 24 | 0 | `tf32_hc_prenorm_gemm` |

`test_mega_gate.py`, `test_mega_mhc.py` and `test_mega_moe.py` cannot run here:
they need TileKernels / TileLang, which are not installed on this machine. Those
paths remain unchecked.

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
git clone --recursive https://github.com/yeasen-z/DeepGEMM-thor-sm110.git
cd DeepGEMM-thor-sm110

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
with no CPython dependency. There are two ways to build it, and **either is
fine**; they differ only in what they leave behind:

| | `build_sgl_deep_gemm.sh` (wheel) | `build_ext.py` (in place) |
|---|---|---|
| Compiles | `csrc/torch_library_api.cpp` | `csrc/torch_library_api.cpp` |
| Produces | `sgl-deep-gemm` wheel | `_C.so` only |
| Install | `pip install` the wheel | overwrites the extension |
| Use when | shipping / clean environments | iterating on `csrc/` |

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
> requirement. Building also avoids the checksum trap described below.

**Build a wheel.** `build_sgl_deep_gemm.sh` initialises the
submodules, links the CUTLASS headers into `deep_gemm/include`, stages the
package, compiles `_C.so`, and produces a wheel whose import name is still
`deep_gemm`: `docs/` and `csrc/` are also copied into the wheel so that
assertion messages and error-lookup material survive an install.

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

> **Don't try to recognise the patch by hashing `_C.so`.** The binary embeds the
> absolute path of the sources it was compiled from (the assertion messages
> reference them, and they are visible via `strings`). Two builds of
> byte-identical sources in different directories therefore produce **different**
> binaries of identical size, differing only in that embedded path. A hash you
> compute yourself will not match anyone else's, so a mismatch is not evidence of
> a bad build. Use the behavioural check in §4.5 instead: `bf16_gemm_nt` must
> run, and the stock library must fail with `Unsupported architecture`.

Whichever route you use, confirm you really got the patched build — do it
behaviourally, as in §4.5.

**Compile `_C.so` in place.** Faster when iterating on `csrc/`;
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
> The wheel build copies them into the wheel (`build_sgl_deep_gemm.sh` lines 55-56), so a
> `pip install` of the wheel is self-contained. If you instead run `deep_gemm`
> straight from a source checkout (`PYTHONPATH=$PWD`), create them yourself:
>
> ```bash
> ln -sfn "$PWD/third-party/cutlass/include/cutlass" deep_gemm/include/cutlass
> ln -sfn "$PWD/third-party/cutlass/include/cute"    deep_gemm/include/cute
> ```

### 4.5 Verify the build is live

**Run this from outside the source tree.** The repo root contains a `deep_gemm/`
source directory, and Python prefers the current directory over `site-packages`.
Run the snippet from inside the clone and you import the *source tree*, which has
no compiled `_C.so`, so you get:

```
ImportError: DeepGEMM extension is missing; build the TORCH_LIBRARY _C module first.
```

That is not a failed install — it is the wrong `sys.path`. `cd` somewhere else
(simplest), or use the doctor command in §4.7, which reports where it loaded from.

```bash
cd /tmp   # or anywhere outside the clone
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
assertion messages point at them), so any hash of it changes with the build
directory even when the sources are byte-identical.

### 4.6 Ask the build what it supports

The Python side and `_C.so` are installed separately, so they can disagree —
typically when the wheel's Python files are current but the extension is stale.
A capability query answered by the **Python** package would lie in exactly that
case, so the answer comes from the compiled binary:

```python
import deep_gemm
deep_gemm.supported_arch_majors()   # -> (9, 10, 11, 12)
deep_gemm.supports_arch('sm110')    # -> True
deep_gemm.supports_arch(11)         # -> True
deep_gemm.supports_arch('sm90')     # -> True
deep_gemm.supports_arch('sm80')     # -> False
```

`supports_arch` accepts a major (`11`), a CUDA-style number (`110`) or an
`sm`-style name (`'sm110'`, `'sm_110'`). This is safe to call at **import time**,
before any CUDA context exists — which is what frameworks need, because choices
like "enable the UE8M0 FP8 scale path" are made before the model loads.

A stock upstream `_C.so` has no such symbol, so the call raises `AttributeError`
instead of returning a wrong `True`. That asymmetry is the point.

### 4.7 Diagnose an install: `python -m deep_gemm.doctor`

One command, everything needed to explain a broken install, with a non-zero exit
code when something is actually wrong:

```
$ python -m deep_gemm.doctor
DeepGEMM doctor
============================================================
version               : 0.2.3+sm110
package               : .../site-packages/deep_gemm
_C.so                 : .../deep_gemm/_C.so (2,050,896 bytes, built 2026-10-22 11:44)  [ok]
CUDA_HOME             : /path/to/cuda  [ok]
nvcc                  : /path/to/cuda/bin/nvcc  [ok]
device headers        : /path/to/cuda/include/curand_kernel.h  [ok]
torch                 : 2.14.1+cu130 (cuda 13.0)  [ok]
supported arch majors : sm9, sm10, sm11, sm12  [ok]
device                : NVIDIA Thor, sm_110 (compute capability 11.0), 20 SMs  [ok]
arch check            : sm11 is supported by this build  [ok]
JIT cache             : /home/user/.dj  (default; set DG_JIT_CACHE_DIR to move)
JIT cached kernels    : 530 entries in /home/user/.dj/cache

running a 128x128x128 bf16 GEMM through the dispatcher ...
dispatch              : bf16_gemm_nt ran, max|diff| vs fp32 ref = 0.0078  [ok]
============================================================

RESULT: OK
```

That is real output from this machine (paths shortened); the `_C.so` size and
build time are this build's, so yours will differ.

It reports the `_C.so` **path, size and build time** but never a hash: the
binary embeds its build-time source path, so a hash differs between
byte-identical builds and only produces false alarms.

### 4.8 Environment variables and the JIT cache

The runtime JIT compiles kernels on first use and caches the resulting cubins.
DeepGEMM's variable prefix is `DG`; unset variables fall back to the global
`DJ_` prefix, and then to the built-in default.

| Variable | Default | Meaning |
|---|---|---|
| `DG_JIT_CACHE_DIR` | `$HOME/.dj` | Where compiled kernels are cached |
| `DJ_JIT_CACHE_DIR` | — | Global fallback if the `DG_` form is unset |
| `CUDA_HOME` | — | Required, even to `import deep_gemm` |

`DG_JIT_CACHE_DIR` accepts **colon-separated paths**. The **first is the writable
root**; the rest are **read-only lookup roots**:

```bash
export DG_JIT_CACHE_DIR=/var/cache/deepgemm:/opt/deepgemm-baked
```

A kernel is looked up in every root in order, so a container image can bake a
warm cache into `/opt/deepgemm-baked` and still let each run write new kernels to
the writable root. Layout is `<root>/cache/<family>.<hash>/`.

**First run is slow, and that is expected.** The cache is per (kernel, shape,
option) combination, so warm-up cost scales with how many shapes you touch — a
serving stack that pre-warms thousands of `M` values can spend minutes compiling
before the first token. It happens **once** per cache directory. If startup time
matters, pre-warm deliberately (SGLang: `python -m sglang.compile_deep_gemm`, or
`SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1` to cut the warm-up set).

### 4.9 Run the test suites

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

> **Which directory to run.** `sgl_deep_gemm/tests/` is the superset: it holds
> all 14 files of the repo-root `tests/` plus 13 more Mega MoE / TRT-LLM / SiTU
> variants that need extra dependencies (TileKernels, TileLang, DeepEP). The
> shared files are also **newer** there — they carry the SM120 `skip:` guards
> and `test_bf16_repeatability` / `test_sm90_block32` entry points that the
> repo-root copies lack. All results recorded here come from
> `sgl_deep_gemm/tests/`.

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

## 7. Beyond the dispatch patch

The SM110 enablement itself is the 45-site change in §1. Everything below is
**separate** — it does not alter kernel selection, and none of it is needed for
the patch to work. It exists because integrating this fork into a serving stack
surfaced avoidable failure modes:

| Change | Problem it removes | Where |
|---|---|---|
| `deep_gemm.supported_arch_majors()` / `supports_arch()` | Frameworks had to guess capability from the version string, which cannot distinguish a stale `_C.so` from a current one | §4.6 |
| `python -m deep_gemm.doctor` | Diagnosing an install meant manually checking five things by hand | §4.7 |
| Actionable `CUDA_HOME` error | A bare `AssertionError` at import with no hint about the device headers | §4.2 |
| Build-time prerequisite checks | `nvcc`/headers problems surfaced later as opaque compiler or linker errors | §4.4 |
| One version source for both build paths | Two build paths reported two different versions for one source tree | §10 |
| `DG_JIT_CACHE_DIR` semantics | The read-only lookup-root feature was undocumented, so baked caches were unusable | §4.8 |

The capability query is the one piece that touches `csrc/`: it adds a
`csrc/utils/arch.hpp` constant and one registered read-only accessor. It does not
change dispatch, and it reads a constant rather than querying the device, so it
answers without a CUDA context and cannot drift from the device it describes.

---

## 8. Fork branch notes

This fork inherits the conventions of the `sgl-project/DeepGEMM` fork:

- `main` tracks upstream updates and should stay clean.
- `dev` is the active SGL integration branch and can be force-updated during rebases.
- `release/*` branches are archives for past wheel releases.

> In **this** fork the SM110 enablement is carried on `main` directly; the
> `dev` / `release/*` conventions above describe the upstream fork layout.

---

## 9. One version, one file

`sgl_deep_gemm/VERSION` (`0.2.3+sm110`) is the **only** place this repository
stores a version. Both build paths read it:

| Reader | How |
|---|---|
| `build_sgl_deep_gemm.sh` | stages it to `deep_gemm/VERSION` in the wheel (`tr -d '[:space:]'`) |
| `setup.py` (`pip install .`) | `get_public_version()` reads the same file |
| `deep_gemm/__init__.py` | `_read_version()` reads `VERSION` next to itself, falling back to `../sgl_deep_gemm/VERSION` in a source tree |

This was previously two values. `deep_gemm/__init__.py` hard-coded
`__version__ = '2.8.0'`, and `setup.py` parsed *that* with a regex, so the same
tree produced `sgl-deep-gemm 0.2.3+sm110` through one path and `deep_gemm 2.8.0`
through the other.

**`2.8.0` was not an older release.** It was an *internal* build number on a
separate axis from the release tags. `git log` over `deep_gemm/__init__.py`
gives the internal series as
`2.1.1 → 2.2.0 → 2.3.0 → 2.4.2 → 2.5.0 → 2.6.1 → 2.8.0`, and `2.8.0` arrives in
`66081d4` ("Public Release 26/09"), dated 2026-09-10 — before the `v0.2.2`
(2026-10-01) and `v0.2.3` (2026-10-16) tags, but *newer* than everything else on
its own axis. The two numbers never referred to the same thing.

Two details are handled so that real packaging tools keep working:

- **A second `+` would be illegal.** PEP 440 allows one local segment. Upstream
  appended `+<commit>` / `+local`, which on top of `0.2.3+sm110` would produce
  `0.2.3+sm110+local` — rejected by `packaging.version.parse`. `setup.py` now
  returns the file's value verbatim, and prints a dirty-tree warning instead of
  encoding it. Verified: `0.2.3+sm110+abc1234` is invalid, `0.2.3+sm110` is not.
- **The wheel ships the resolved value.** `CustomBuildPy.copy_version_file()`
  writes `get_package_version()`'s *result* into `deep_gemm/VERSION` inside the
  wheel, so `deep_gemm.__version__` equals
  `importlib.metadata.version('deep_gemm')` for any given install rather than
  drifting from it.

Verified end to end: `python setup.py --version` → `0.2.3+sm110`,
`egg_info`'s `PKG-INFO` → `Version: 0.2.3+sm110`, and `build_sgl_deep_gemm.sh`
→ `0.2.3+sm110`. The wheel's `deep_gemm/__init__.py` is copied from
`sgl_deep_gemm/__init__.py`, so an already-installed package is untouched.


---

## 10. Upstream project

This fork replaces the upstream DeepGEMM README with the page you are reading.
The original is preserved verbatim at
[README.upstream.md](README.upstream.md) and upstream remains
[sgl-project/DeepGEMM](https://github.com/sgl-project/DeepGEMM).