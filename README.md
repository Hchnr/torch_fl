# torch_fl

统一多芯片 PyTorch 扩展框架，将 CUDA / NPU（华为昇腾）/ MLU（寒武纪）整合为同一套安装包和用户 API，并集成 FlagGems Triton 算子库。

---

## 当前状态（v0.1.0）

本项目处于 **框架骨架阶段**。Python 统一层和 C++ 框架层的核心骨架已就绪，**CUDA 透传路径目前完整可用**，NPU / MLU 路径的 Python 层框架已搭好但各后端的 C++ 扩展（`_C.so`）和运行时尚未迁移。

### 已实现

| 模块 | 文件 | 状态 |
|------|------|------|
| 硬件检测 | `fl/_detect.py` | ✅ 完整 |
| 统一入口（CUDA 透传 + PrivateUse1 注册） | `fl/__init__.py` | ✅ 完整 |
| PrivateUse1 9 步注册逻辑 | `fl/_register.py` | ✅ 完整 |
| CUDA 后端配置 + 透传层 | `fl/backends/cuda/` | ✅ 完整 |
| NPU 后端配置 stub | `fl/backends/npu/__init__.py` | ⚙ 配置已有，C++ 扩展待迁移 |
| MLU 后端配置 stub | `fl/backends/mlu/__init__.py` | ⚙ 配置已有，C++ 扩展待迁移 |
| FlagGems 自主注册层 | `fl/gems/__init__.py` | ✅ 完整 |
| FlagGems rollback 机制 | `fl/gems/rollback.py` | ✅ 完整 |
| monkey-patch 调度框架 | `fl/_patches/__init__.py` | ✅ 完整 |
| nn.Module patches | `fl/_patches/module_patches.py` | ✅ 完整 |
| 分布式 patches（委托 backend_mod） | `fl/_patches/distributed_patches.py` | ✅ 框架完整 |
| 优化器 patches（委托 backend_mod） | `fl/_patches/optim_patches.py` | ✅ 框架完整 |
| Dynamo patches（委托 backend_mod） | `fl/_patches/dynamo_patches.py` | ✅ 框架完整 |
| Reductions patches（委托 backend_mod） | `fl/_patches/reductions_patches.py` | ✅ 框架完整 |
| 统一设备 API | `fl/core/device.py` | ✅ 完整（委托 `_C`） |
| 统一 Stream/Event | `fl/core/streams.py` | ✅ 完整（委托 `_C._StreamBase`） |
| 统一内存管理（14 个 API） | `fl/core/memory.py` | ✅ 完整（委托 `_C`） |
| Inductor 后端框架 | `fl/_inductor/__init__.py` | ✅ `AcceleratorDeviceOpOverrides` 完整 |
| Dynamo 设备接口框架 | `fl/_dynamo/__init__.py` | ✅ `AcceleratorInterface` 完整 |
| C++ 后端接口声明 | `fl/csrc/core/DeviceInterface.h` | ✅ 完整 |
| C++ GuardImpl | `fl/csrc/core/GuardImpl.cpp` | ⚙ 设备部分完整，Stream/Event 为 stub |
| C++ CPU/Autograd Fallback | `fl/csrc/aten/FallbackKernel.cpp` | ✅ 完整 |
| C++ TensorIterator mirror | `fl/csrc/aten/mirror/TensorIterator.h/.cpp` | ✅ `AcceleratorTensorIterator` 完整 |

### 尚未实现（TODO）

| 模块 | 说明 |
|------|------|
| `fl/_patches/tensor_patches.py` 等 4 个 | 内容为 `pass`，待从 torch_npu/torch_mlu 迁移 |
| `fl/core/amp/`, `fl/core/random.py` | 单行注释占位，待实现 |
| `fl/codegen/` | 统一 codegen 框架骨架，实现代码待补 |
| `fl/distributed/`, `fl/profiler/`, `fl/testing/`, `fl/multiprocessing/` | 空文件占位 |
| `fl/gems/bench.py` | A/B 性能对比工具，待实现 |
| `fl/backends/npu/python/` + csrc | NPU 运行时，待从 torch_npu 迁移 |
| `fl/backends/mlu/python/` + csrc | MLU 运行时，待从 torch_mlu 迁移 |
| `fl/csrc/stub.cpp` Python bindings | `_initExtension` 等绑定待补全 |
| 所有 `tests/` 目录（除 `test_cuda_smoke.py`） | 待补充 |

---

## 安装

```bash
# 依赖：Python 3.10+，PyTorch >= 2.7.1

# CUDA 后端（无 C++ 编译，纯 Python）
pip install torch>=2.7.1
FL_BACKEND=cuda pip install -e .

# 可选：从源码安装 FlagGems 以获得 Triton 算子加速
# （FlagGems 暂不支持 pip install，需从源码安装）
git clone https://github.com/FlagOpen/FlagGems && pip install -e FlagGems
```

> NPU / MLU 后端安装方式待 C++ 扩展迁移完成后补充。

---

## 可以跑通的测试和 Demo

### 1. CUDA 冒烟测试 — `tests/test_cuda_smoke.py`

```bash
FL_BACKEND=cuda FL_GEMS=0 python -m pytest tests/test_cuda_smoke.py -v
```

覆盖 5 组测试类（约 15 个 case）：`TestDetect`（环境变量和自动检测逻辑）、`TestCUDAImport`（`BACKEND_NAME`、`_C`、`dist_backend`、`fl.device()`）、`TestCUDAPassthrough`（`is_available` / `device_count` / `get_device_name` / `Stream` / `memory_allocated` 与 `torch.cuda` 一致性）、`TestCUDACompute`（GPU 上矩阵乘法和 synchronize）、`TestGemsModule` + `TestBackendConfig`（三后端 config 字段）。无 GPU 时硬件相关 case 自动跳过。

### 2. CUDA 统一 API Demo — `examples/demo_cuda_api.py`

```bash
FL_BACKEND=cuda FL_GEMS=0 python examples/demo_cuda_api.py
```

验证 `fl.BACKEND_NAME`、`fl._C` 为 `None`、`fl.device(0)` 返回 `cuda:0`、`fl.dist_backend()` 返回 `"nccl"`，有 GPU 时进一步验证矩阵乘法、`fl.synchronize()`、`fl.memory_allocated()`、`fl.Stream` 类型。

### 3. 后端配置接口 Demo — `examples/demo_backend_config.py`

```bash
python examples/demo_backend_config.py
```

无需任何加速器硬件，纯 Python 可运行。打印 CUDA / NPU / MLU 三个后端的 `get_config()` 关键字段，验证 `backend_name`、`comm_backend`、`is_native`、`has_inductor`、`gems_config` 均符合设计预期。

### 4. 硬件检测 Demo — `examples/demo_detect.py`

```bash
python examples/demo_detect.py
```

验证 `FL_BACKEND` 环境变量能强制覆盖为 cuda / npu / mlu，以及清除变量后自动检测按 NPU > MLU > CUDA 优先级工作。

### 5. FlagGems API Demo — `examples/demo_gems.py`

```bash
FL_BACKEND=cuda python examples/demo_gems.py
```

需从源码安装 FlagGems（见安装一节），否则打印提示并退出。验证 `import fl` 后 FlagGems 自动全量启用、`registered_ops()` 返回算子列表、`disable()` / `set(exclude=[...])` / `reset()` 各状态切换正确、`with fl.gems.without()` 和 `with fl.gems.use_only([...])` 两个 context manager 在作用域内外的行为。

---

## 项目结构

```
torch_fl/
├── fl/
│   ├── __init__.py          # 统一入口（CUDA 透传 / PrivateUse1 注册 + FlagGems）
│   ├── _detect.py           # 硬件自动检测
│   ├── _register.py         # PrivateUse1 注册（9 步）
│   ├── _patches/            # monkey-patch 框架（仅 NPU/MLU 使用）
│   │   ├── __init__.py
│   │   ├── module_patches.py        ✅ 实现
│   │   ├── distributed_patches.py   ✅ 框架
│   │   ├── optim_patches.py         ✅ 框架
│   │   ├── dynamo_patches.py        ✅ 框架
│   │   ├── reductions_patches.py    ✅ 框架
│   │   ├── tensor_patches.py        TODO
│   │   ├── storage_patches.py       TODO
│   │   ├── serialization_patches.py TODO
│   │   └── fsdp_patches.py          TODO
│   ├── core/                # 统一设备 API（仅 PrivateUse1 使用）
│   │   ├── device.py        ✅ 14 个设备 API
│   │   ├── streams.py       ✅ Stream/Event
│   │   ├── memory.py        ✅ 14 个内存 API
│   │   ├── amp/             TODO（autocast/GradScaler）
│   │   └── random.py        TODO（RNG 状态）
│   ├── gems/                # FlagGems 集成层（全后端共享）
│   │   ├── __init__.py      ✅ 完整（auto_enable/disable/set/reset/without/use_only）
│   │   ├── rollback.py      ✅ 完整
│   │   └── bench.py         TODO（A/B 性能对比工具）
│   ├── _inductor/           ✅ AcceleratorDeviceOpOverrides
│   ├── _dynamo/             ✅ AcceleratorInterface
│   ├── codegen/             TODO（统一 codegen 框架）
│   ├── distributed/         TODO
│   ├── profiler/            TODO
│   ├── testing/             TODO
│   ├── multiprocessing/     TODO
│   ├── backends/
│   │   ├── cuda/            ✅ 完整（纯 Python 透传）
│   │   ├── npu/             ⚙ config 已有，C++ 扩展 + python 运行时待迁移
│   │   └── mlu/             ⚙ config 已有，C++ 扩展 + python 运行时待迁移
│   └── csrc/                # 统一 C++ 框架层（仅 NPU/MLU 编译）
│       ├── core/
│       │   ├── DeviceInterface.h  ✅
│       │   └── GuardImpl.cpp      ✅（Stream/Event 部分为 stub）
│       ├── aten/
│       │   ├── FallbackKernel.cpp ✅
│       │   └── mirror/
│       │       ├── TensorIterator.h   ✅
│       │       └── TensorIterator.cpp ✅
│       └── stub.cpp               TODO（Python bindings 待补全）
├── examples/
│   ├── demo_detect.py         ✅ 硬件检测
│   ├── demo_backend_config.py ✅ 后端配置
│   ├── demo_cuda_api.py       ✅ CUDA 统一 API
│   └── demo_gems.py           ✅ FlagGems 控制 API
├── tests/
│   └── test_cuda_smoke.py   ✅ 可运行
├── setup.py                 ✅（CUDA 纯 Python 路径可用，NPU/MLU 骨架）
├── CMakeLists.txt           ✅（骨架）
└── version.txt              0.1.0
```

---

## 下一步开发计划

大致优先级：

1. **M1**：CUDA 路径补全（`fl/core/amp/`、`fl/core/random.py`、`fl/gems/bench.py`）+ 扩充测试
2. **M2**：实现剩余 4 个 patch stubs + `fl/testing/`、`fl/distributed/` 基础框架
3. **M4**：实现 `fl/codegen/` 统一算子注册 codegen
4. **M3**：补全 `fl/csrc/stub.cpp` Python bindings，完成 C++ 框架层编译
5. **M5/M6**：迁移 NPU / MLU 后端运行时，消除 MLU 42 个源码补丁
