import os
from setuptools import setup, find_packages


def _auto_detect_sdk():
    """自动检测 SDK 环境"""
    ascend_path = os.environ.get(
        "ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"
    )
    if os.path.exists(ascend_path):
        return "npu"
    neuware_path = os.environ.get("NEUWARE_HOME", "/usr/local/neuware")
    if os.path.exists(neuware_path):
        return "mlu"
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    if os.path.exists(os.path.join(cuda_home, "bin/nvcc")):
        return "cuda"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    raise RuntimeError("Cannot detect SDK. Set FL_BACKEND=npu|mlu|cuda")


BACKEND = os.environ.get("FL_BACKEND", _auto_detect_sdk())

packages = find_packages(include=["fl", "fl.*"])

with open("version.txt") as f:
    version = f.read().strip()

if BACKEND == "cuda":
    setup(
        name="torch-fl",
        version=version,
        packages=packages,
        install_requires=["torch>=2.7.1"],
        extras_require={"gems": ["flag-gems"]},
        python_requires=">=3.9",
    )

else:
    from torch.utils.cpp_extension import CppExtension, BuildExtension
    from glob import glob

    common_sources = glob("fl/csrc/**/*.cpp", recursive=True)
    backend_sources = glob(f"fl/backends/{BACKEND}/csrc/**/*.cpp", recursive=True)

    SDK_CONFIG = {
        "npu": {
            "home": "ASCEND_HOME_PATH",
            "libs": ["ascendcl", "hccl", "opapi"],
            "include_extra": ["third_party/acl/inc"],
        },
        "mlu": {
            "home": "NEUWARE_HOME",
            "libs": ["cnnl", "cnrt", "cncl", "cndrv"],
            "include_extra": [],
        },
    }

    cfg = SDK_CONFIG[BACKEND]
    sdk_home = os.environ.get(cfg["home"], "")

    ext_modules = [
        CppExtension(
            name=f"fl.backends.{BACKEND}._C",
            sources=common_sources + backend_sources,
            include_dirs=[
                "fl/csrc",
                f"fl/backends/{BACKEND}/csrc",
                f"{sdk_home}/include",
            ] + cfg["include_extra"],
            library_dirs=[f"{sdk_home}/lib64"],
            libraries=cfg["libs"] + ["torch", "torch_python", "c10"],
        )
    ]

    setup(
        name="torch-fl",
        version=version,
        packages=packages,
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
        install_requires=["torch>=2.7.1"],
        extras_require={"gems": ["flag-gems"]},
        python_requires=">=3.9",
    )
