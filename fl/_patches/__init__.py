"""
统一 monkey-patch 框架。
仅 NPU/MLU 使用，CUDA 跳过。
"""
from fl._patches import (
    module_patches,
    tensor_patches,
    storage_patches,
    distributed_patches,
    optim_patches,
    serialization_patches,
    dynamo_patches,
    fsdp_patches,
    reductions_patches,
)


def apply_all_patches(backend_name: str, backend_mod, _C):
    """统一应用所有 monkey-patch。"""
    module_patches.apply(backend_name, backend_mod, _C)
    tensor_patches.apply(backend_name, backend_mod, _C)
    storage_patches.apply(backend_name, backend_mod, _C)
    distributed_patches.apply(backend_name, backend_mod, _C)
    optim_patches.apply(backend_name, backend_mod, _C)
    serialization_patches.apply(backend_name, backend_mod, _C)
    dynamo_patches.apply(backend_name, backend_mod, _C)
    fsdp_patches.apply(backend_name, backend_mod, _C)
    reductions_patches.apply(backend_name, backend_mod, _C)

    if hasattr(backend_mod, "apply_extra_patches"):
        backend_mod.apply_extra_patches()
