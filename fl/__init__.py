"""torch_fl 统一入口"""
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch  # noqa: E402
from fl._detect import detect_backend  # noqa: E402

BACKEND_NAME = detect_backend()

if BACKEND_NAME == "cuda":
    # ========== CUDA 透传模式 ==========
    _C = None

    from fl.backends.cuda import _setup_cuda_passthrough
    _setup_cuda_passthrough()

    from importlib import import_module
    _backend_mod = import_module(f"fl.backends.{BACKEND_NAME}")
    _backend_config = _backend_mod.get_config()

    from torch.cuda import (
        is_available, device_count, current_device, set_device, synchronize,
        get_device_name, get_device_properties, get_device_capability,
    )
    from torch.cuda import Stream, Event, current_stream, default_stream
    from torch.cuda import (
        memory_allocated, max_memory_allocated, memory_reserved,
        empty_cache, memory_stats, memory_summary,
    )

else:
    # ========== PrivateUse1 模式（NPU / MLU） ==========
    from fl._register import register_backend  # noqa: F401

    from importlib import import_module
    _backend_mod = import_module(f"fl.backends.{BACKEND_NAME}")
    _C = import_module(f"fl.backends.{BACKEND_NAME}._C")

    from fl import core as _device_module

    _backend_config = _backend_mod.get_config()

    register_backend(BACKEND_NAME, _device_module, _backend_config)

    from fl._patches import apply_all_patches
    apply_all_patches(BACKEND_NAME, _backend_mod, _C)

    _backend_mod.post_register()

    _C._initExtension()

    if _backend_config.get("has_inductor"):
        from fl._inductor import register_inductor_backend
        register_inductor_backend(BACKEND_NAME, _backend_mod)

    from fl.core.device import (  # noqa: F811
        is_available, device_count, current_device, set_device, synchronize,
        get_device_name, get_device_properties, get_device_capability,
    )
    from fl.core.streams import Stream, Event, current_stream, default_stream  # noqa: F811
    from fl.core.memory import (  # noqa: F811
        memory_allocated, max_memory_allocated, memory_reserved,
        empty_cache, memory_stats, memory_summary,
    )

# ========== 以下逻辑 CUDA 和 PrivateUse1 共享 ==========


def device(index: int = 0):
    """
    统一的设备构造函数。
    fl.device(0) 在不同后端返回：
      CUDA: torch.device("cuda", 0)
      NPU:  torch.device("npu", 0)
      MLU:  torch.device("mlu", 0)
    """
    return torch.device(BACKEND_NAME, index)


def dist_backend() -> str:
    """
    统一的分布式后端名称。
    CUDA: "nccl",  NPU: "hccl",  MLU: "cncl"
    """
    return {"cuda": "nccl", "npu": "hccl", "mlu": "cncl"}[BACKEND_NAME]


# FlagGems 集成
_FL_GEMS = os.environ.get("FL_GEMS", "1")
if _FL_GEMS == "1":
    from fl import gems as _gems_module
    if _gems_module.is_available():
        _gems_module._auto_enable()
elif _FL_GEMS != "0":
    raise ValueError(f"Invalid FL_GEMS value: {_FL_GEMS}. Use '0' or '1'.")

# 注册退出钩子（仅 PrivateUse1 后端需要）
if BACKEND_NAME != "cuda":
    import atexit
    if hasattr(_backend_mod, "shutdown"):
        atexit.register(_backend_mod.shutdown)


def _autoload():
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = os.getenv(
        "TORCH_DEVICE_BACKEND_AUTOLOAD", "1"
    )
