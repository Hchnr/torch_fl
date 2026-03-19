/**
 * PyInit 入口 — 骨架。
 * 实际 Python bindings 需从 torch_npu/torch_mlu 迁移。
 */
#include <torch/extension.h>

PYBIND11_MODULE(_C, m) {
    m.doc() = "torch-fl unified C++ extension";
    // TODO: 注册 _initExtension, _device_count, _current_device 等
}
