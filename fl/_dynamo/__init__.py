"""
统一 Dynamo 设备接口。
提供参数化的 AcceleratorInterface 基类。
"""
from torch._dynamo.device_interface import DeviceInterface


class AcceleratorInterface(DeviceInterface):
    """参数化的 DeviceInterface 基类。"""

    _backend_name = None

    class Worker:
        @staticmethod
        def set_device(device: int):
            from fl.core.device import set_device
            set_device(device)

        @staticmethod
        def current_device() -> int:
            from fl.core.device import current_device
            return current_device()

    @staticmethod
    def is_available() -> bool:
        from fl.core.device import device_count
        return device_count() > 0

    @staticmethod
    def current_device() -> int:
        from fl.core.device import current_device
        return current_device()

    @staticmethod
    def set_device(device: int):
        from fl.core.device import set_device
        set_device(device)

    @staticmethod
    def device_count() -> int:
        from fl.core.device import device_count
        return device_count()

    @staticmethod
    def synchronize():
        from fl.core.device import synchronize
        synchronize()
