"""
统一 Inductor 后端框架。
提供参数化的 DeviceOpOverrides 基类。
"""
from torch._inductor.codegen.common import (
    DeviceOpOverrides,
    register_device_op_overrides,
)


class AcceleratorDeviceOpOverrides(DeviceOpOverrides):
    """参数化的 DeviceOpOverrides，后端只需填写名称。"""

    def __init__(self, backend_name, get_raw_stream_fn=None):
        self._backend_name = backend_name
        self._get_raw_stream_fn = get_raw_stream_fn

    def import_get_raw_stream_as(self, name):
        return (
            f"from fl.backends.{self._backend_name}._C "
            f"import _getCurrentRawStream as {name}"
        )

    def set_device(self, device_idx):
        return f"fl.set_device({device_idx})"

    def synchronize(self):
        return "fl.synchronize()"

    def device_guard(self, device_idx):
        return f"fl.core.device._DeviceGuard({device_idx})"


def register_inductor_backend(backend_name, backend_mod):
    if hasattr(backend_mod, "get_device_op_overrides"):
        overrides = backend_mod.get_device_op_overrides()
    else:
        overrides = AcceleratorDeviceOpOverrides(backend_name, None)
    register_device_op_overrides(backend_name, overrides)

    if hasattr(backend_mod, "patch_rng_prims"):
        backend_mod.patch_rng_prims()
