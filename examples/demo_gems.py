"""
demo_gems.py — FlagGems 集成 Demo

验证内容：
  - import fl 后 FlagGems 自动全量启用（FL_GEMS=1 默认）
  - fl.gems.disable() 关闭后算子退回原生实现
  - fl.gems.set(exclude=[...]) 排除指定算子
  - fl.gems.reset() 恢复全量
  - fl.gems.without() context manager 在作用域内临时关闭
  - fl.gems.use_only([...]) context manager 仅保留指定算子
  - fl.gems.op_source() 查询单个算子的当前来源

依赖：pip install flag-gems
      （未安装时脚本打印提示并退出，不报错）

运行方式：
  FL_BACKEND=cuda python examples/demo_gems.py
"""
import os
os.environ.setdefault("FL_BACKEND", "cuda")
# 不设 FL_GEMS=0，让 FlagGems 自动启用

import fl
import torch

print("=== FlagGems 集成 Demo ===\n")

if not fl.gems.is_available():
    print("flag-gems 未安装，跳过 Demo。")
    print("安装方式: pip install flag-gems")
    raise SystemExit(0)

print(f"gems available : {fl.gems.is_available()}")
print(f"gems enabled   : {fl.gems.is_enabled()}")
print(f"registered ops : {len(fl.gems.registered_ops())} 个")
print(f"  示例: {fl.gems.registered_ops()[:5]}")

if not fl.is_available():
    print("\n（无 CUDA 设备，跳过算子调用部分）")
    raise SystemExit(0)

x = torch.randn(512, 512, device=fl.device(0))

print(f"\nop_source('softmax') = {fl.gems.op_source('softmax')}")

# disable / reset
fl.gems.disable()
print(f"\n[disable] enabled = {fl.gems.is_enabled()}")
fl.gems.reset()
print(f"[reset]   enabled = {fl.gems.is_enabled()}")

# set(exclude)
fl.gems.set(exclude=["mm", "addmm"])
print(f"\n[set exclude mm,addmm] op_source('mm') = {fl.gems.op_source('mm')}")
fl.gems.reset()

# without() context manager
with fl.gems.without():
    y = torch.softmax(x, dim=-1)
    print(f"\n[without] enabled inside  = {fl.gems.is_enabled()}")
print(f"[without] enabled outside = {fl.gems.is_enabled()}")

# use_only() context manager
with fl.gems.use_only(["softmax", "layer_norm"]):
    y = torch.softmax(x, dim=-1)
    active = fl.gems.registered_ops()
    print(f"\n[use_only softmax,layer_norm] ops = {active}")
print(f"[use_only] after exit, ops count = {len(fl.gems.registered_ops())}")
