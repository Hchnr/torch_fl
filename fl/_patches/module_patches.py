"""统一的 nn.Module 补丁。"""
import torch


def apply(backend_name: str, backend_mod, _C):
    # nn.Module.to 增强（来自 NPU：cast_weight）
    original_to = torch.nn.Module.to

    def patched_to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        if hasattr(self, "cast_weight") and device is not None:
            import fl
            if fl.is_available():
                with torch.no_grad():
                    self.cast_weight(device)
        return original_to(self, *args, **kwargs)

    torch.nn.Module.to = patched_to

    # SyncBatchNorm.forward 补丁
    if hasattr(backend_mod, "get_syncbn_forward"):
        torch.nn.modules.batchnorm.SyncBatchNorm.forward = (
            backend_mod.get_syncbn_forward()
        )

    # DataLoader worker 补丁
    if hasattr(backend_mod, "get_dataloader_init_patch"):
        from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
        _MultiProcessingDataLoaderIter.__init__ = (
            backend_mod.get_dataloader_init_patch()
        )

    # PackedSequence 补丁（来自 MLU）
    @property
    def is_device(self):
        return self.data.device.type == backend_name

    setattr(
        torch.nn.utils.rnn.PackedSequence, f"is_{backend_name}", is_device
    )

    def to_device(self, *args, **kwargs):
        ex = torch.tensor(
            (), dtype=self.data.dtype, device=self.data.device
        ).to(*args, **kwargs)
        if ex.device.type == backend_name:
            return self.to(*args, **kwargs)
        return self.to(*args, device=backend_name, **kwargs)

    setattr(torch.nn.utils.rnn.PackedSequence, backend_name, to_device)

    # ShardedGradScaler 补丁
    if hasattr(backend_mod, "ShardedGradScaler"):
        torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler = (
            backend_mod.ShardedGradScaler
        )
