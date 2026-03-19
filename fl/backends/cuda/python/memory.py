"""CUDA memory — 透传到 torch.cuda"""
from torch.cuda import (
    memory_allocated,
    max_memory_allocated,
    memory_reserved,
    max_memory_reserved,
    memory_cached,
    max_memory_cached,
    empty_cache,
    memory_stats,
    memory_stats_as_nested_dict,
    memory_summary,
    reset_peak_memory_stats,
    reset_accumulated_memory_stats,
    set_per_process_memory_fraction,
    memory_snapshot,
)

__all__ = [
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "memory_cached",
    "max_memory_cached",
    "empty_cache",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "memory_summary",
    "reset_peak_memory_stats",
    "reset_accumulated_memory_stats",
    "set_per_process_memory_fraction",
    "memory_snapshot",
]
