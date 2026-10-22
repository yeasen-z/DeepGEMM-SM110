# DeepGEMM SM110（Jetson AGX Thor）适配

让 DeepGEMM 能在 **NVIDIA Jetson AGX Thor**（SM110）上跑起来，供 SGLang 使用。

- 基线：`sgl-project/DeepGEMM` **v0.2.3**
- 目标设备：Jetson AGX Thor（`sm_110`，compute capability 11.0）
- 详细验证记录：[docs/sm110-thor-support.md](docs/sm110-thor-support.md)
- 上游原始 README（本 fork 替换了首页，原文保留）：[README.upstream.md](README.upstream.md)

---

## 1. 改了什么

DeepGEMM 原来只认 SM90 / SM100 / SM120 三种架构。Thor 报的是 **11**，所以任何调用都会在选核之前直接报错：

```
RuntimeError: Assertion error (csrc/apis/gemm.hpp:225): Unsupported architecture
```

SM110 和 SM100 共用同一套 kernel，所以只需要告诉「调度层」：**SM110 属于 SM100 这一类**。

- 改动范围：**10 个头文件、45 处判断**，只是把 `11` 加进已有的条件里。
- 设备端代码、CUTLASS、JIT 编译流程**都没动**。
- 附带加了一个能力查询接口 `deep_gemm.supports_arch('sm110')`，让 SGLang 能自动判断当前装的是不是支持 SM110 的版本（否则它会退回 Triton 后端）。

完整的逐文件改动清单见 [docs/sm110-thor-support.md](docs/sm110-thor-support.md) 第 1 节。

---

## 2. 怎么安装

### 2.1 前置条件

| 项目 | 要求 |
|---|---|
| GPU | Thor（SM110）。也兼容 SM90 / SM100 |
| 驱动 / CUDA | 驱动 `595.78`+，CUDA Toolkit **13.0**+（必须有 `nvcc`，运行时 JIT 要用） |
| 编译器 | g++ 13（需要 C++20） |
| PyTorch | `2.14.1+cu130` |
| 系统依赖 | `sudo apt-get install -y libdw-dev` |

### 2.2 拉代码

```bash
git clone --recursive https://github.com/yeasen-z/DeepGEMM-thor-sm110.git
cd DeepGEMM-thor-sm110

# 如果忘了加 --recursive
git submodule update --init --recursive
```

两个子模块都必须有：`third-party/cutlass` 和 `third-party/deep_jit`。

### 2.3 编译并安装

```bash
# 0) 先进入你要安装的目标环境（conda activate / source venv/bin/activate）
conda activate your-env

# 1) 指定 CUDA
export CUDA_HOME=/usr/local/cuda          # Thor 上可能是 /home/<user>/cuda_home_thor

# 2) 指定用哪个 Python 编译 —— 必须就是上面这个环境
export PYTHON_EXE="$(which python)"

# 3) 构建 wheel（脚本会打印它实际使用的解释器和 torch 版本，请核对）
bash build_sgl_deep_gemm.sh

# 4) 安装（用同一个环境的 pip）
pip install --force-reinstall --no-deps dist/sgl_deep_gemm-*.whl
```

> 第 0 步不能跳过：`pip install` 用的是当前环境的 pip。如果你只 `export PYTHON_EXE`
> 而没激活环境，编译和安装可能落到两个不同的环境里，装完 import 会失败。

装出来的包名是 **`sgl-deep-gemm`**，版本 **`0.2.3+sm110`**，导入名仍然是 `deep_gemm`（SGLang 的代码不用改）。

`0.2.3+sm110` 满足 SGLang 的依赖 `sgl-deep-gemm==0.2.3`，所以不影响 SGLang 安装依赖。

### 2.4 验证装好了

**在代码目录外面执行**（否则会 import 到源码目录而不是装好的包）：

```bash
cd /tmp
python -c "
import deep_gemm, torch
print('version  :', deep_gemm.__version__)
print('capability:', torch.cuda.get_device_capability())
print('supports sm110:', deep_gemm.supports_arch('sm110'))
a = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda')
b = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda')
d = torch.empty(4096, 4096, dtype=torch.bfloat16, device='cuda')
deep_gemm.bf16_gemm_nt(a, b, d)
print('GEMM OK', float(d.abs().mean()))
"
```

Thor 上正常应该输出：

```
version  : 0.2.3+sm110
capability: (11, 0)
supports sm110: True
GEMM OK 51.0
```

一条命令做全面体检：

```bash
python -m deep_gemm.doctor
```

---

## 3. 怎么配置

### 3.1 SGLang 环境变量

```bash
export CUDA_HOME=/usr/local/cuda
export SGLANG_ENABLE_JIT_DEEPGEMM=1

python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B-FP8 \
    --fp8-gemm-backend deep_gemm \
    --dtype bfloat16
```

### 3.2 SGLang 侧还需要两处改动

**下面的改动在 SGLang 仓库里，不在本仓库。** SM100 的 FP8 kernel 要求权重 scale 是 UE8M0 格式，而 SGLang 的设备判断没包含 SM110，不改的话会直接 trap（`CUDA_ERROR_LAUNCH_FAILED`）：

```python
# 1. configurer.py —— 把 SM110 加进去
DEEPGEMM_SCALE_UE8M0 = ENABLE_JIT_DEEPGEMM and (
    get_platform().is_sm100 or get_device_sm() in (110, 120, 121)
)

# 2. fp8_utils.py —— 只在真的不支持时才退回 triton
elif (
    backend == "auto"
    and get_platform().is_sm110
    and not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
):
    backend = "triton"
```

改动前后对比和原理见 [docs/sm110-thor-support.md](docs/sm110-thor-support.md) 第 6 节。

### 3.3 JIT 缓存（可选）

kernel 第一次用到时会现场编译并缓存。默认缓存在 `$HOME/.dj`，想换个地方：

```bash
export DG_JIT_CACHE_DIR=/var/cache/deepgemm:/opt/deepgemm-baked
```

冒号分隔：**第一个是写入目录，后面的是只读查找目录**（适合在镜像里预烤一份缓存）。

---

## 4. 注意事项

1. **不要用 `pip install .`。**
   仓库根目录的 `setup.py` 是上游留下的老入口，在 v0.2.3 上本身就是坏的（构建会失败）。给 SGLang 用一律走 `bash build_sgl_deep_gemm.sh`。

2. **不要同时安装 `sgl-deep-gemm` 和 `deep_gemm` 两个包。**
   它们发行名不同，但会往同一个 `deep_gemm/` 目录写文件，可能互相覆盖；卸载其中一个可能破坏另一个。

3. **编译用的 Python 必须和安装用的环境一致。**
   编译产物和 torch 的 C++ ABI 绑定，torch 2.13 和 2.14 不兼容。`build_sgl_deep_gemm.sh` 会打印它用的是哪个解释器，请确认一下；最稳妥的是显式设置 `PYTHON_EXE`。

4. **不要用 md5 / 哈希去判断装的是不是补丁版。**
   编译产物里嵌了源码的绝对路径，同样的源码在不同目录构建出来的文件哈希不同。用第 2.4 节的运行时验证来判断。

5. **第一次跑会很慢，这是正常的。**
   kernel 按「形状 + 选项」逐个编译，缓存一次之后就不慢了。SGLang 场景可以预热：`python -m sglang.compile_deep_gemm`。

6. **`CUDA_HOME` 必须设置。**
   连 `import deep_gemm` 都需要它，因为运行时要用 `nvcc` 现场编译 kernel。如果报错找不到 CUDA，看 [docs/sm110-thor-support.md](docs/sm110-thor-support.md) 第 4.2 节。

7. **本仓库不提供预编译二进制。**
   需要你在自己的机器上编译。因为编译产物和 **CPU 架构 + torch 版本 + CUDA 大版本** 三者严格绑定，传给你的文件在别的机器上装不上。

### 还没验证的部分

以下路径本次**没有测试**，要用的话请自行验证：

- Mega MoE、Mega Gate、Mega mHC（它们的参考实现需要 TileKernels / TileLang，本机没装）
- dense / m-grouped 的 FP4

已充分验证的：BF16 和 FP8 的 dense / grouped GEMM、SF layout、k-grouped FP4、psum、attention、einsum、hyperconnection，以及 `Qwen3-8B-FP8` 端到端推理。

---

## 5. 测试设备

所有结果都来自下面这台机器：

| 项目 | 值 |
|---|---|
| 板卡 | NVIDIA Jetson AGX Thor Developer Kit（Tegra, aarch64） |
| GPU | `NVIDIA Thor`，compute capability **11.0**（`sm_110`），20 SM，128 GB 统一内存 |
| L4T / 内核 | R39 (release 2.0)，`6.8.12-1021-tegra` |
| 系统 | Ubuntu 24.04.4 LTS，14 核 |
| 驱动 | `595.78` |
| CUDA | **13.0**（`nvcc` V13.0.88） |
| 编译器 | GCC/G++ 13.3.0 |
| PyTorch | `2.14.1+cu130` |
| Python | conda 环境 `inscope`，Python 3.12 |

### 验证结果概览

| 测试 | 用例数 | 结果 |
|---|---:|---|
| `test_bf16.py` | 449 | 0 失败 |
| `test_fp8_fp4.py` | 353 | 0 失败 |
| `test_layout.py` | 272 | 0 失败 |
| FP4 k-grouped（单独跑） | 189 | 0 失败 |
| psum SF layout（单独跑） | 224 | 0 失败 |
| `test_attention.py` / `test_einsum.py` / `test_hyperconnection.py` | 88 | 0 失败 |
| SGLang + `Qwen3-8B-FP8` 端到端 | 4 prompts | 文本正确 |

性能（SGLang 实测，同样 41 个请求 / 144 万 token，prefill 阶段）：

| 分组 | DeepGEMM | Triton | 提升 |
|---|---:|---:|---:|
| 全部 | **5,331 tok/s** | 3,639 tok/s | **1.47x** |
| ~4k 短请求 | **12,018 tok/s** | 5,428 tok/s | **2.21x** |
| ~64k 长请求 | **5,165 tok/s** | 3,571 tok/s | **1.45x** |

> 说明：这是 prefill 阶段、24/36 层的对比（两者条件完全一致，所以比值有效）。完整数据和方法见 [docs/sm110-thor-support.md](docs/sm110-thor-support.md) 第 5.6 节。

---

## 6. 上游项目

本 fork 替换了首页，上游原始 README 原文保留在 [README.upstream.md](README.upstream.md)。
上游项目：[sgl-project/DeepGEMM](https://github.com/sgl-project/DeepGEMM)。