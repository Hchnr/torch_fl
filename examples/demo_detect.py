"""
demo_detect.py — 硬件检测 Demo

验证内容：
  - FL_BACKEND 环境变量强制指定后端
  - 自动检测逻辑（NPU > MLU > CUDA 优先级）

运行方式：
  python examples/demo_detect.py
  FL_BACKEND=npu  python examples/demo_detect.py
  FL_BACKEND=mlu  python examples/demo_detect.py
  FL_BACKEND=cuda python examples/demo_detect.py
"""
import os
from fl._detect import detect_backend

print("=== 硬件检测 Demo ===\n")

# 通过环境变量强制指定
for backend in ["cuda", "npu", "mlu"]:
    os.environ["FL_BACKEND"] = backend
    result = detect_backend()
    print(f"FL_BACKEND={backend} -> detect_backend() = '{result}'")

# 清除环境变量，走自动检测
del os.environ["FL_BACKEND"]
try:
    auto = detect_backend()
    print(f"\n自动检测结果: '{auto}'")
except RuntimeError as e:
    print(f"\n自动检测: 未找到加速器 ({e})")
