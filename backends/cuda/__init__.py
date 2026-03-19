"""
CUDA 后端 — 纯透传层。
与 NPU/MLU 的关键区别：不编译 C++ 扩展，不做 PrivateUse1 注册，不做 monkey-patch。
"""
import torch


def get_config():
    return {
        "backend_name": "cuda",
        "comm_backend": "nccl",
        "sdk_home_env": "CUDA_HOME",
        "version": torch.version.cuda,
        "unsupported_dtype": [],
        "has_inductor": True,
        "has_graph_capture": True,
        "is_native": True,
        "gems_config": {
            "vendor_name": "nvidia",
            "dispatch_key": "CUDA",
            "exclude_ops": [],
        },
    }


def post_register():
    """CUDA 不需要额外注册"""
    pass


def _setup_cuda_passthrough():
    """
    设置 CUDA 透传：将 fl.* 统一 API 指向 torch.cuda。
    无需注册 — CUDA 已经是 PyTorch 内置的。
    无需 monkey-patch — PyTorch 对 CUDA 的支持是完备的。
    无需 C++ 扩展 — c10/cuda + aten/native/cuda 已内置在 PyTorch 中。
    """
    pass
