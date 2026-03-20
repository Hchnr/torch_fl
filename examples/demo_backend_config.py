"""
demo_backend_config.py — 后端配置接口 Demo

验证内容：
  - 三个后端（CUDA / NPU / MLU）的 get_config() 返回正确字段
  - backend_name、comm_backend、is_native、has_inductor、gems_config 均符合预期
  - 无需任何加速器硬件，纯 Python 可运行

运行方式：
  python examples/demo_backend_config.py
"""
from fl.backends.cuda import get_config as cuda_cfg
from fl.backends.npu  import get_config as npu_cfg
from fl.backends.mlu  import get_config as mlu_cfg

print("=== 后端配置接口 Demo ===\n")

for name, cfg in [("CUDA", cuda_cfg()), ("NPU", npu_cfg()), ("MLU", mlu_cfg())]:
    print(f"--- {name} ---")
    print(f"  backend_name  : {cfg['backend_name']}")
    print(f"  comm_backend  : {cfg['comm_backend']}")
    print(f"  is_native     : {cfg['is_native']}")
    print(f"  has_inductor  : {cfg['has_inductor']}")
    print(f"  gems key      : {cfg['gems_config']['dispatch_key']}")
    print(f"  gems exclude  : {cfg['gems_config']['exclude_ops']}")
    print()
