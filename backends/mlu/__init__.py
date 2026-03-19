"""
MLU 后端（寒武纪）— 骨架实现。
实际 C++ 扩展和算子实现需从 torch_mlu 迁移。
"""
import torch


def get_config():
    return {
        "backend_name": "mlu",
        "comm_backend": "cncl",
        "sdk_home_env": "NEUWARE_HOME",
        "version": None,
        "unsupported_dtype": [
            torch.quint8, torch.quint4x2, torch.quint2x4,
            torch.qint32, torch.qint8,
        ],
        "has_inductor": False,
        "has_graph_capture": False,
        "is_native": False,
        "dynamo_interface": None,  # TODO: MLU 尚未实现
        "process_group_factory": None,  # TODO: 从 torch_mlu 迁移 CNCL factory
        "gems_config": {
            "vendor_name": "cambricon",
            "dispatch_key": "PrivateUse1",
            "exclude_ops": [
                "scatter_add_", "masked_scatter", "masked_scatter_",
            ],
        },
    }


def post_register():
    """MLU 特有的额外注册 — 骨架，待迁移"""
    # TODO: 注册 cnnl / mlufusion 到 torch.backends
    # TODO: TF32 控制
    # TODO: gencase 检查
    pass


def shutdown():
    """MLU 退出清理 — 骨架，待迁移"""
    pass
