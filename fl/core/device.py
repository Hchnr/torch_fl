"""
统一设备 API — 仅 PrivateUse1 模式（NPU/MLU）下使用。
CUDA 模式下 __init__.py 直接 from torch.cuda import ... 透传。
"""
import torch


def _parse_device(device) -> int:
    if device is None:
        return current_device()
    if isinstance(device, torch.device):
        return device.index or 0
    if isinstance(device, str):
        return torch.device(device).index or 0
    return int(device)


def is_available() -> bool:
    from fl import _C
    return _C._device_count() > 0


def device_count() -> int:
    from fl import _C
    return _C._device_count()


def current_device() -> int:
    from fl import _C
    return _C._current_device()


def set_device(device) -> None:
    from fl import _C
    idx = _parse_device(device)
    _C._set_device(idx)


def synchronize(device=None) -> None:
    from fl import _C
    if device is None:
        device = current_device()
    _C._synchronize(_parse_device(device))


def get_device_name(device=None) -> str:
    from fl import _C
    return _C._get_device_name(_parse_device(device))


def get_device_properties(device=None):
    from fl import _C
    return _C._get_device_properties(_parse_device(device))


def get_device_capability(device=None):
    from fl import _C
    return _C._get_device_capability(_parse_device(device))


def can_device_access_peer(device, peer_device) -> bool:
    from fl import _C
    return _C._can_device_access_peer(
        _parse_device(device), _parse_device(peer_device)
    )


def is_bf16_supported() -> bool:
    from fl import _C
    return _C._is_bf16_supported()


def mem_get_info(device=None):
    from fl import _C
    return _C._mem_get_info(_parse_device(device))
