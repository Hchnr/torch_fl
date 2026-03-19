"""统一的分布式补丁。"""
import torch


def apply(backend_name: str, backend_mod, _C):
    # NPU 特有的分布式方法替换
    if backend_name == "npu" and hasattr(backend_mod, "get_distributed_module"):
        dist_mod = backend_mod.get_distributed_module()
        torch._C._distributed_c10d._verify_params_across_processes = (
            dist_mod._verify_params_across_processes
        )
        torch.distributed.batch_isend_irecv = dist_mod._batch_isend_irecv
        torch.distributed.distributed_c10d.batch_isend_irecv = (
            dist_mod._batch_isend_irecv
        )
        torch.distributed.gather = dist_mod._gather
        torch.distributed.distributed_c10d.gather = dist_mod._gather
        torch.distributed.gather_object = dist_mod._gather_object
        torch.distributed.distributed_c10d.gather_object = dist_mod._gather_object
        torch.distributed.distributed_c10d.rendezvous = (
            dist_mod._trigger_rendezvous_decorator(
                torch.distributed.distributed_c10d.rendezvous
            )
        )

    # MLU 特有的分布式补丁
    if backend_name == "mlu" and hasattr(backend_mod, "get_distributed_module"):
        mlu_dist = backend_mod.get_distributed_module()
        mlu_dist.apply_all_distributed_patches()
