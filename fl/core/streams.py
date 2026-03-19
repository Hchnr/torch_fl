"""
统一 Stream/Event — 仅 PrivateUse1 模式（NPU/MLU）下使用。
CUDA 模式下直接 from torch.cuda import Stream, Event 透传。
"""
from fl.core.device import _parse_device, current_device


def _get_C():
    from fl import _C
    return _C


class Stream(_get_C()._StreamBase if False else object):
    """
    统一的 Stream 封装。
    实际基类在运行时由后端 _C._StreamBase 提供。
    """

    def __init__(self, device=None, priority=0, stream_id=0,
                 device_index=0, stream_ptr=0):
        if device is not None:
            device_index = _parse_device(device)
        super().__init__(
            priority=priority, stream_id=stream_id,
            device_index=device_index, stream_ptr=stream_ptr,
        )

    def synchronize(self):
        super().synchronize()

    def wait_event(self, event):
        super().wait_event(event)

    def wait_stream(self, stream):
        event = Event()
        stream.record_event(event)
        self.wait_event(event)

    def record_event(self, event=None):
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self) -> bool:
        return super().query()


class Event:
    """
    统一的 Event 封装。
    实际实现由后端 _C._EventBase 提供。
    """

    def __init__(self, enable_timing=False, blocking=False):
        _C = _get_C()
        self._event = _C._EventBase(enable_timing=enable_timing, blocking=blocking)

    def record(self, stream=None):
        if stream is None:
            stream = current_stream()
        self._event.record(stream)

    def wait(self, stream=None):
        if stream is None:
            stream = current_stream()
        self._event.wait(stream)

    def query(self) -> bool:
        return self._event.query()

    def synchronize(self):
        self._event.synchronize()

    def elapsed_time(self, end_event) -> float:
        return self._event.elapsed_time(end_event._event)


def current_stream(device=None):
    _C = _get_C()
    idx = _parse_device(device) if device is not None else current_device()
    return _C._getCurrentStream(idx)


def default_stream(device=None):
    _C = _get_C()
    idx = _parse_device(device) if device is not None else current_device()
    return _C._getDefaultStream(idx)
