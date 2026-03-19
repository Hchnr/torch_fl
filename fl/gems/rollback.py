"""
FlagGems 算子异常时自动回退到厂商原生实现。
"""
import functools
import logging
import warnings
from collections import defaultdict

logger = logging.getLogger(__name__)

_rollback_count = defaultdict(int)
_failed_ops = set()
_max_rollback_warnings = 3


def wrap_with_rollback(op_name, gems_fn):
    """
    包装 FlagGems 算子，失败时自动回退到厂商实现。

    回退机制：
      1. 首次执行 gems_fn 失败 → 打印 warning
      2. 将 op_name 加入 _failed_ops
      3. 触发 fl.gems 重新注册（排除 _failed_ops 中的算子）
      4. 后续调用自动走厂商原生实现
    """

    @functools.wraps(gems_fn)
    def wrapped(*args, **kwargs):
        try:
            return gems_fn(*args, **kwargs)
        except Exception as e:
            _rollback_count[op_name] += 1
            _failed_ops.add(op_name)

            if _rollback_count[op_name] <= _max_rollback_warnings:
                warnings.warn(
                    f"[torch-fl] FlagGems op '{op_name}' failed: {e}. "
                    f"Falling back to vendor implementation. "
                    f"This op will use vendor impl for the rest of the session."
                )

            from fl import gems
            logger.info(f"Re-registering FlagGems without failed op: {op_name}")
            gems.disable()
            gems._register_impl(exclude=list(_failed_ops), rollback_on_error=True)

            raise

    return wrapped


def get_failed_ops() -> set:
    """返回已 rollback 到厂商实现的算子集合"""
    return _failed_ops.copy()


def clear_failed_ops():
    """清除 rollback 记录"""
    _failed_ops.clear()
    _rollback_count.clear()
