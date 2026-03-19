"""统一的 multiprocessing reductions 补丁。"""
import torch
from multiprocessing.reduction import ForkingPickler


def apply(backend_name: str, backend_mod, _C):
    if not hasattr(backend_mod, "get_reductions_module"):
        return

    reductions = backend_mod.get_reductions_module()

    ForkingPickler.register(reductions.EventClass, reductions.reduce_event)
    torch.multiprocessing.reductions.reduce_storage = reductions.reduce_storage
    torch.multiprocessing.reductions.reduce_tensor = reductions.reduce_tensor
    torch.multiprocessing.reductions.init_reductions()
