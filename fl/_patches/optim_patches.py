"""统一的优化器补丁。"""


def apply(backend_name: str, backend_mod, _C):
    if hasattr(backend_mod, "apply_optim_patches"):
        backend_mod.apply_optim_patches()
