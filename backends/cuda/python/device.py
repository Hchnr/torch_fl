"""CUDA device API — 透传到 torch.cuda"""
from torch.cuda import (
    is_available,
    device_count,
    current_device,
    set_device,
    synchronize,
    get_device_name,
    get_device_properties,
    get_device_capability,
    can_device_access_peer,
    mem_get_info,
)

__all__ = [
    "is_available",
    "device_count",
    "current_device",
    "set_device",
    "synchronize",
    "get_device_name",
    "get_device_properties",
    "get_device_capability",
    "can_device_access_peer",
    "mem_get_info",
]
