"""CUDA streams — 透传到 torch.cuda"""
from torch.cuda import Stream, Event, current_stream, default_stream

__all__ = ["Stream", "Event", "current_stream", "default_stream"]
