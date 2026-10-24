# DeepGEMM SM110 (Jetson AGX Thor)

**English** | [简体中文](README.zh-CN.md)

SM110 support for DeepGEMM v0.2.3.post2, for use with SGLang on the NVIDIA Jetson AGX Thor.

- Device: NVIDIA Jetson AGX Thor, `sm_110` / compute capability 11.0
- Distribution: `sgl-deep-gemm==0.2.3.post2+sm110`
- Python import: `deep_gemm`
- Detailed verification record: [docs/sm110-thor-support.md](docs/sm110-thor-support.md)
- Original upstream README: [README.upstream.md](README.upstream.md)

## What changed

Upstream DeepGEMM v0.2.3.post2 did not recognize architecture major 11. Thor therefore stopped with `Unsupported architecture` before selecting a kernel.

This fork:

- adds SM110 to the existing SM100-family host dispatch conditions;
- updates 45 dispatch checks across 10 host headers;
- leaves the device kernels, CUTLASS, and JIT compilation logic unchanged;
- exposes `deep_gemm.supports_arch('sm110')` so SGLang can detect support.

The change reuses the SM100 kernel family already used by DeepGEMM. The detailed file list and test evidence are in the [verification document](docs/sm110-thor-support.md).

## Install

### Requirements

- NVIDIA Jetson AGX Thor or another supported NVIDIA GPU
- CUDA Toolkit 13.0 or newer, including `nvcc`
- GCC/G++ 13 with C++20 support
- PyTorch built for the CUDA version in use
- `libdw-dev`

On the test machine, the working environment was CUDA 13.0, PyTorch 2.14.1+cu130, GCC 13.3, and Python 3.12.

Install the system dependency if missing:

```bash
sudo apt-get install -y libdw-dev
```

### Clone

```bash
git clone https://github.com/yeasen-z/DeepGEMM-thor-sm110.git
cd DeepGEMM-thor-sm110
```

The build script downloads the required `cutlass` and `deep_jit` submodules automatically. You may also clone with `--recursive`:

```bash
git clone --recursive https://github.com/yeasen-z/DeepGEMM-thor-sm110.git
```

### Build and install

Activate the environment that will run SGLang first (replace `your-env` with your environment name, or activate your venv). Use the same Python for building and installing:

```bash
conda activate your-env
export PYTHON_EXE="$(which python)"

# Optional: set this when CUDA is not discoverable automatically.
# export CUDA_HOME=/usr/local/cuda

bash build_sgl_deep_gemm.sh
"$PYTHON_EXE" -m pip install --force-reinstall --no-deps dist/sgl_deep_gemm-*.whl
```

The build script prints the Python and PyTorch versions it uses. Check that they are the versions in your target environment.

### Verify

Run the check outside the source directory, so Python does not import the checkout instead of the installed package:

```bash
cd /tmp
python -c "
import deep_gemm, torch
print('version:', deep_gemm.__version__)
print('device :', torch.cuda.get_device_capability())
print('SM110  :', deep_gemm.supports_arch('sm110'))
a = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
b = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
c = torch.empty(128, 128, device='cuda', dtype=torch.bfloat16)
deep_gemm.bf16_gemm_nt(a, b, c)
torch.cuda.synchronize()
print('GEMM   : OK')
"
```

Expected on Thor:

```text
version: 0.2.3.post2+sm110
device : (11, 0)
SM110  : True
GEMM   : OK
```

For a broader diagnostic:

```bash
python -m deep_gemm.doctor
```

The installed version `0.2.3.post2+sm110` satisfies SGLang's `sgl-deep-gemm==0.2.3.post2` pin: PEP 440 ignores the local segment, so the `+sm110` build is accepted while never being confused with the upstream release.

## Configure SGLang

```bash
export SGLANG_ENABLE_JIT_DEEPGEMM=1
```

For FP8 models, use the DeepGEMM backend when appropriate:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B-FP8 \
  --fp8-gemm-backend deep_gemm \
  --dtype bfloat16
```

For the corresponding SGLang SM110 adaptation, use [yeasen-z/sglang-inscope](https://github.com/yeasen-z/sglang-inscope) and follow its installation instructions. This repository contains the DeepGEMM changes only. The SGLang-side integration, including the UE8M0 scale gate, is explained in [the integration section](docs/sm110-thor-support.md#6-sglang-integration-prerequisite).

## CUDA lookup

You do **not** always need to set `CUDA_HOME`. DeepGEMM looks for CUDA in this order:

1. `CUDA_HOME` or `CUDA_PATH`;
2. `nvcc` on `PATH`;
3. `/usr/local/cuda`.

Set it when CUDA is installed in a non-standard location, when several CUDA versions are installed, or when automatic detection fails:

```bash
export CUDA_HOME=/path/to/cuda
```

The selected CUDA installation must provide `bin/nvcc` and the device headers needed by the runtime JIT. The first kernel use may take longer because the kernel is compiled and cached. The default cache is `$HOME/.dj`; override it with `DG_JIT_CACHE_DIR` if needed.

## Important notes

1. **Use the SGLang wheel path above.** Do not use the repository-root `pip install .` path for SGLang.
2. **Do not install both `sgl-deep-gemm` and a separate `deep_gemm` distribution.** They use the same Python package directory and can overwrite each other.
3. Use the same Python environment for building, installing, and running SGLang. The extension is tied to the PyTorch/CUDA environment.
4. The first run may be slow while JIT kernels are compiled. Later runs use the cache.
5. This fork does not provide a universal prebuilt binary. Build on the target machine.
6. The tested device was one Jetson AGX Thor. Mega MoE/Gate/mHC and dense or m-grouped FP4 paths were not covered by the recorded tests.

## Test summary

On the Jetson AGX Thor test machine:

- On the `v0.2.3.post2` base: `test_bf16` (391), `test_einsum` (60), `test_layout` (216),
  `test_hyperconnection` (24) and `test_arch_traits` completed with zero failures — 691 cases.
  `test_fp8_fp4` passed 232 cases before being stopped part-way through its k-grouped
  section, so it is not claimed as a completed run.
- MQA prefill and paged decode (`test_attention.py`) also pass on the same wheel.
- Mega Gate and Mega mHC are refused on SM110 by design and filtered out of the suite.
- Recorded earlier on the `v0.2.3` base and **not** repeated on post2: unguarded
  k-grouped FP4 (189/189) and unguarded psum layout (224/224).
- SGLang with Qwen3-8B-FP8 produced correct text.
- Steady-state prefill throughput: 5,331 tok/s with DeepGEMM vs 3,639 tok/s with Triton on the same workload (1.47x).

The performance comparison covers prefill only: 24 of 36 model layers, 41 requests, and 1,440,227 tokens. It is not a whole-model or decode benchmark.

See [docs/sm110-thor-support.md](docs/sm110-thor-support.md) for the full test device, commands, limitations, and measurements.

## Upstream

This fork is based on [sgl-project/DeepGEMM](https://github.com/sgl-project/DeepGEMM) v0.2.3.post2 — the revision SGLang pins as `sgl-deep-gemm==0.2.3.post2`, so every entry point SGLang calls keeps its post2 signature. The original README is preserved in [README.upstream.md](README.upstream.md).
