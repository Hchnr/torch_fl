import sys

import torch


def register_backend(backend_name: str, device_module, backend_config: dict):
    """
    统一执行 NPU/MLU 共同的 PrivateUse1 注册步骤。

    当前各项目中的对应代码：
      NPU: torch_npu/__init__.py L210-L216
      MLU: torch_mlu/__init__.py L48-L75
    """
    acc = torch._C._get_accelerator()
    if acc.type != "cpu":
        raise RuntimeError(
            f"Two accelerators cannot be used at the same time: "
            f"{backend_name} and {acc.type}"
        )

    # Step 1: 重命名 PrivateUse1
    torch.utils.rename_privateuse1_backend(backend_name)

    # Step 2: 注册设备模块
    torch._register_device_module(backend_name, device_module)

    # Step 3: 生成 Tensor/Module/Storage 方法
    unsupported_dtype = backend_config.get("unsupported_dtype", [])
    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True, for_module=True, for_storage=True,
        unsupported_dtype=unsupported_dtype,
    )

    # Step 4: UninitializedTensorMixin
    to_device_fn = getattr(torch.Tensor, backend_name)
    torch.nn.parameter.UninitializedTensorMixin._allowed_methods.append(to_device_fn)

    # Step 5: 注册 torch.backends.xxx
    torch.backends.__setattr__(backend_name, device_module)
    sys.modules[f"torch.backends.{backend_name}"] = device_module

    # Step 6: 注册 torch.version.xxx
    setattr(torch.version, backend_name, backend_config.get("version"))

    # Step 7: 设置 gradient checkpointing 默认设备
    from torch.utils.checkpoint import DefaultDeviceType
    DefaultDeviceType.set_device_type(backend_name)

    # Step 8: 注册分布式后端
    if "process_group_factory" in backend_config:
        comm_backend = backend_config["comm_backend"]
        pg_factory = backend_config["process_group_factory"]
        torch.distributed.Backend.register_backend(
            comm_backend, pg_factory, extended_api=True, devices=[backend_name]
        )

    # Step 9: 注册 Dynamo 设备接口
    if "dynamo_interface" in backend_config and backend_config["dynamo_interface"]:
        from torch._dynamo.device_interface import register_interface_for_device
        dynamo_iface = backend_config["dynamo_interface"]
        register_interface_for_device(backend_name, dynamo_iface)
        for i in range(32):
            register_interface_for_device(f"{backend_name}:{i}", dynamo_iface)
