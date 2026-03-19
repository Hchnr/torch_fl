"""
NPU 后端（华为 Ascend）— 骨架实现。
实际 C++ 扩展和算子实现需从 torch_npu 迁移。
"""
import torch


def get_config():
    return {
        "backend_name": "npu",
        "comm_backend": "hccl",
        "sdk_home_env": "ASCEND_HOME_PATH",
        "version": None,
        "unsupported_dtype": [
            torch.quint8, torch.quint4x2, torch.quint2x4,
            torch.qint32, torch.qint8,
        ],
        "has_inductor": True,
        "has_graph_capture": True,
        "is_native": False,
        "dynamo_interface": None,  # TODO: 从 torch_npu 迁移 NpuInterface
        "process_group_factory": None,  # TODO: 从 torch_npu 迁移 HCCL factory
        "gems_config": {
            "vendor_name": "ascend",
            "dispatch_key": "PrivateUse1",
            "exclude_ops": [
                "contiguous", "sort", "sort_stable", "copy_", "_to_copy",
            ],
        },
    }


def post_register():
    """NPU 特有的额外注册 — 骨架，待迁移"""
    # TODO: contrib 模块注入
    # TODO: 自定义算子注册
    # TODO: ASD（自动故障检测）
    # TODO: RPC 后端
    # TODO: DTensor 规则
    # TODO: LCCL 后端注册
    pass


def shutdown():
    """NPU 退出清理 — 骨架，待迁移"""
    pass
