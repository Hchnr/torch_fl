"""
统一内存管理 API — 仅 PrivateUse1 模式（NPU/MLU）下使用。
CUDA 模式下直接 from torch.cuda import ... 透传。
"""
from fl.core.device import _parse_device


def _get_C():
    from fl import _C
    return _C


def memory_allocated(device=None) -> int:
    return _get_C()._memory_allocated(_parse_device(device))


def max_memory_allocated(device=None) -> int:
    return _get_C()._max_memory_allocated(_parse_device(device))


def memory_reserved(device=None) -> int:
    return _get_C()._memory_reserved(_parse_device(device))


def max_memory_reserved(device=None) -> int:
    return _get_C()._max_memory_reserved(_parse_device(device))


def memory_cached(device=None) -> int:
    return memory_reserved(device)


def max_memory_cached(device=None) -> int:
    return max_memory_reserved(device)


def empty_cache() -> None:
    _get_C()._empty_cache()


def memory_stats(device=None) -> dict:
    return _get_C()._memory_stats(_parse_device(device))


def memory_stats_as_nested_dict(device=None) -> dict:
    return _get_C()._memory_stats_as_nested_dict(_parse_device(device))


def memory_summary(device=None, abbreviated=False) -> str:
    return _get_C()._memory_summary(_parse_device(device), abbreviated)


def reset_peak_memory_stats(device=None) -> None:
    _get_C()._reset_peak_memory_stats(_parse_device(device))


def reset_accumulated_memory_stats(device=None) -> None:
    _get_C()._reset_accumulated_memory_stats(_parse_device(device))


def set_per_process_memory_fraction(fraction, device=None) -> None:
    _get_C()._set_per_process_memory_fraction(fraction, _parse_device(device))


def memory_snapshot():
    return _get_C()._memory_snapshot()
