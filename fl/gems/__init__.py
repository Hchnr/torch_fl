"""
FlagGems 集成层 — torch-fl 自主注册模式。

torch-fl 不调用 flag_gems.enable()，而是运行时读取 flag_gems._FULL_CONFIG
获取算子列表，由 torch-fl 自行完成注册。

默认行为：import fl 时，若 FlagGems 已安装，自动启用全量算子。
"""
import logging

import torch

logger = logging.getLogger(__name__)

_gems_enabled = False
_gems_lib = None
_registered_ops = {}
_full_config_cache = None


def is_available() -> bool:
    """检查 FlagGems 是否已安装"""
    try:
        import flag_gems  # noqa: F401
        return True
    except ImportError:
        return False


def _load_full_config():
    """运行时读取 FlagGems 的算子配置表。"""
    global _full_config_cache
    if _full_config_cache is not None:
        return _full_config_cache

    import flag_gems
    _full_config_cache = flag_gems._FULL_CONFIG
    return _full_config_cache


def _get_backend_exclude_ops() -> list:
    """获取当前后端建议排除的 FlagGems 算子。"""
    try:
        import flag_gems
        device = flag_gems.runtime.device
        from flag_gems.runtime.backend import get_curent_device_unused_op
        return list(get_curent_device_unused_op(device.vendor_name))
    except Exception:
        return []


def _resolve_dispatch_key() -> str:
    """获取当前后端的 dispatch key。"""
    try:
        import flag_gems
        return flag_gems.runtime.device.dispatch_key
    except Exception:
        from fl import BACKEND_NAME
        if BACKEND_NAME == "cuda":
            return "CUDA"
        return "PrivateUse1"


def _auto_enable():
    """内部方法：import fl 时自动调用。"""
    _register_impl(exclude=None, include=None, rollback_on_error=True)


def _register_impl(exclude=None, include=None, rollback_on_error=True):
    """
    核心注册实现：遍历 FlagGems 算子并注册到 dispatch table。

    Args:
        exclude: 排除的算子函数名列表
        include: 仅注册的算子函数名列表
        rollback_on_error: 是否为每个算子包装 try-except rollback
    """
    global _gems_enabled, _gems_lib, _registered_ops

    if not is_available():
        return

    full_config = _load_full_config()
    backend_exclude = set(_get_backend_exclude_ops())
    user_exclude = set(exclude or [])
    user_include = set(include) if include is not None else None
    dispatch_key = _resolve_dispatch_key()

    _gems_lib = torch.library.Library("aten", "IMPL")
    _registered_ops = {}
    errors = []

    for item in full_config:
        if not item or len(item) < 2:
            continue

        op_name, impl_fn = item[0], item[1]
        func_name = impl_fn.__name__ if hasattr(impl_fn, "__name__") else str(impl_fn)

        # 检查条件函数
        if len(item) > 2:
            condition_fn = item[2]
            if not condition_fn():
                continue

        # 后端不兼容算子排除
        if func_name in backend_exclude:
            continue

        # 用户指定的 include/exclude 过滤
        if user_include is not None:
            if func_name not in user_include and op_name not in user_include:
                continue
        else:
            if func_name in user_exclude or op_name in user_exclude:
                continue

        # 注册
        try:
            if rollback_on_error:
                from fl.gems.rollback import wrap_with_rollback
                wrapped_fn = wrap_with_rollback(op_name, impl_fn)
                _gems_lib.impl(op_name, wrapped_fn, dispatch_key)
            else:
                _gems_lib.impl(op_name, impl_fn, dispatch_key)

            _registered_ops[op_name] = impl_fn
        except Exception as e:
            errors.append((op_name, e))
            logger.debug(f"Failed to register FlagGems op '{op_name}': {e}")

    if errors:
        logger.warning(
            f"FlagGems: {len(errors)} ops failed to register, "
            f"{len(_registered_ops)} ops registered successfully."
        )

    _gems_enabled = len(_registered_ops) > 0


def disable():
    """禁用 FlagGems，恢复到厂商原生算子。"""
    global _gems_enabled, _gems_lib, _registered_ops

    if _gems_lib is not None:
        if hasattr(_gems_lib, "_destroy"):
            _gems_lib._destroy()
        _gems_lib = None

    _registered_ops = {}
    _gems_enabled = False


def set(exclude=None, include=None):
    """
    调整 FlagGems 的使用范围。

    Args:
        exclude: 不使用 FlagGems 的算子列表
        include: 仅使用 FlagGems 的算子列表
    """
    if exclude is not None and include is not None:
        raise ValueError(
            "Cannot specify both 'exclude' and 'include'. Use one of them."
        )

    disable()
    _register_impl(exclude=exclude, include=include)


def reset():
    """恢复为默认状态：全量启用 FlagGems。"""
    disable()
    _auto_enable()


def is_enabled() -> bool:
    return _gems_enabled


def registered_ops() -> list:
    """返回当前 FlagGems 已注册的算子名列表"""
    return list(_registered_ops.keys())


def op_source(op_name: str) -> str:
    """
    查询指定算子当前的实现来源。

    Returns:
        "gems"   — 当前由 FlagGems Triton kernel 处理
        "vendor" — 当前由厂商原生实现处理
        "cpu"    — 未注册，将 fallback 到 CPU
    """
    if op_name in _registered_ops:
        return "gems"
    return "vendor"


class without:
    """Context manager：在作用域内临时关闭 FlagGems。"""

    def __init__(self, ops=None):
        self.ops = ops
        self._prev_enabled = False

    def __enter__(self):
        self._prev_enabled = _gems_enabled
        if self.ops is None:
            disable()
        else:
            disable()
            _register_impl(exclude=self.ops)
        return self

    def __exit__(self, *args):
        disable()
        if self._prev_enabled:
            _auto_enable()


class use_only:
    """Context manager：在作用域内仅使用指定的 FlagGems 算子。"""

    def __init__(self, ops):
        self.ops = ops

    def __enter__(self):
        disable()
        _register_impl(include=self.ops)
        return self

    def __exit__(self, *args):
        disable()
        _auto_enable()
