import importlib
import os


def detect_backend() -> str:
    """
    检测当前环境安装了哪个后端的 SDK。

    检测顺序：
      1. 环境变量 FL_BACKEND 显式指定
      2. NPU / MLU 已编译的 C++ 扩展（PrivateUse1 后端）
      3. CUDA（PyTorch 内置，作为兜底）

    NPU/MLU 优先于 CUDA 的原因：某些加速卡驱动可能同时暴露 CUDA 兼容层，
    如果用户安装了 NPU/MLU 的 C++ 扩展，应优先使用。
    """
    env = os.environ.get("FL_BACKEND")
    if env:
        return env.lower()

    for name in ["npu", "mlu"]:
        try:
            importlib.import_module(f"fl.backends.{name}._C")
            return name
        except ImportError:
            continue

    import torch
    if torch.cuda.is_available():
        return "cuda"

    raise RuntimeError(
        "No accelerator backend found. "
        "Install with: FL_BACKEND=npu pip install torch-fl, "
        "or ensure CUDA is available."
    )
