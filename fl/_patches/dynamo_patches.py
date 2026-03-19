"""统一的 Dynamo 补丁。"""


def apply(backend_name: str, backend_mod, _C):
    if hasattr(backend_mod, "get_dynamo_patches"):
        backend_mod.get_dynamo_patches()()
