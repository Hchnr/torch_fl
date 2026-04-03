# openreg_cuda

`openreg_cuda` 是 `third_party/openreg` 的 CUDA 后端实现。它实现了完全相同的 `or*` C API，但将所有调用委托给真实的 CUDA Runtime，使得 `device="openreg"` 的张量实际运行在 CUDA 设备上。

## API 映射

| openreg API | CUDA API |
|---|---|
| `orMalloc` | `cudaMalloc` |
| `orFree` | `cudaFree` |
| `orMallocHost` | `cudaMallocHost` |
| `orFreeHost` | `cudaFreeHost` |
| `orMemcpy` | `cudaMemcpy` |
| `orMemcpyAsync` | `cudaMemcpyAsync` |
| `orPointerGetAttributes` | `cudaPointerGetAttributes` |
| `orMemoryProtect` | 无操作（CUDA 不使用 mprotect）|
| `orMemoryUnprotect` | 无操作 |
| `orGetDeviceCount` | `cudaGetDeviceCount` |
| `orGetDevice` | `cudaGetDevice` |
| `orSetDevice` | `cudaSetDevice` |
| `orDeviceSynchronize` | `cudaDeviceSynchronize` |
| `orDeviceGetStreamPriorityRange` | `cudaDeviceGetStreamPriorityRange` |
| `orStreamCreate` | `cudaStreamCreate` |
| `orStreamCreateWithPriority` | `cudaStreamCreateWithPriority` |
| `orStreamDestroy` | `cudaStreamDestroy` |
| `orStreamSynchronize` | `cudaStreamSynchronize` |
| `orStreamQuery` | `cudaStreamQuery` |
| `orStreamWaitEvent` | `cudaStreamWaitEvent` |
| `orEventCreate` | `cudaEventCreate` |
| `orEventCreateWithFlags` | `cudaEventCreateWithFlags` |
| `orEventDestroy` | `cudaEventDestroy` |
| `orEventRecord` | `cudaEventRecord` |
| `orEventSynchronize` | `cudaEventSynchronize` |
| `orEventQuery` | `cudaEventQuery` |
| `orEventElapsedTime` | `cudaEventElapsedTime` |

## 目录结构

```
openreg_cuda/
├── CMakeLists.txt
├── README.md
└── csrc/
    ├── device.cpp   # 设备管理
    ├── memory.cpp   # 内存分配与拷贝
    └── stream.cpp   # 流与事件
```

## 构建

通过顶层 `setup.py` 启用：

```bash
USE_CUDA_BACKEND=1 pip install -e . -v --no-build-isolation
```

构建产物为 `libopenreg.so`，与 CPU 模拟后端同名，其余 `torch_openreg` 代码无需任何修改即可使用。
