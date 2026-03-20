"""
demo_cuda_api.py — CUDA 统一 API Demo

验证内容：
  - import fl 在 CUDA 透传模式下正常加载，_C 为 None
  - fl.device() / fl.dist_backend() 返回正确值
  - fl.is_available() / fl.device_count() 与 torch.cuda 一致
  - fl.device(0) 创建张量，完成矩阵乘法并同步
  - fl.memory_allocated() 返回合理数值
  - FlagGems 在 FL_GEMS=0 时不被启用

运行方式：
  FL_BACKEND=cuda FL_GEMS=0 python examples/demo_cuda_api.py
"""
import os
os.environ.setdefault("FL_BACKEND", "cuda")
os.environ.setdefault("FL_GEMS", "0")

import fl
import torch

print("=== CUDA 统一 API Demo ===\n")

print(f"backend      : {fl.BACKEND_NAME}")
print(f"dist_backend : {fl.dist_backend()}")
print(f"_C           : {fl._C}")
print(f"is_available : {fl.is_available()}")
print(f"device_count : {fl.device_count()}")

d = fl.device(0)
print(f"fl.device(0) : {d}")

if fl.is_available():
    print(f"device name  : {fl.get_device_name(0)}")

    x = torch.randn(4, 4, device=d)
    y = x @ x
    print(f"matmul shape : {y.shape}  device: {y.device}")

    fl.synchronize()
    print(f"memory_alloc : {fl.memory_allocated()} bytes")

    s = fl.Stream()
    print(f"Stream type  : {type(s)}")

print(f"\ngems enabled : {fl.gems.is_enabled()}")
print(f"gems ops     : {fl.gems.registered_ops()}")
