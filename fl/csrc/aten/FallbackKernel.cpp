/**
 * 统一 CPU/Autograd fallback。
 *
 * 合并自：
 *   NPU: torch_npu/csrc/aten/VariableFallbackKernel.cpp
 *   MLU: torch_mlu/csrc/aten/MLUFallback.cpp
 */
#include <ATen/native/CPUFallback.h>
#include <torch/library.h>
#include "fl/csrc/core/DeviceInterface.h"

#include <unordered_set>

namespace {

static std::unordered_set<std::string> warned_ops;

void fl_cpu_fallback(
    const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    auto op_name = c10::toString(op.schema().operator_name());
    if (warned_ops.find(op_name) == warned_ops.end()) {
        warned_ops.insert(op_name);
        TORCH_WARN(
            "The operator '", op.schema().operator_name(),
            "' is not currently supported on the ",
            fl::GetBackendName(),
            " backend and will fall back to run on the CPU.");
    }
    at::native::cpu_fallback(op, stack);
}

void fl_sparse_fallback(
    const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    TORCH_CHECK(
        false,
        "The operator '", op.schema().operator_name(),
        "' is not currently supported on the ",
        fl::GetBackendName(), " backend.");
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(
        torch::CppFunction::makeFromBoxedFunction<&fl_cpu_fallback>());
}

TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
    m.fallback(
        torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(_, SparsePrivateUse1, m) {
    m.fallback(
        torch::CppFunction::makeFromBoxedFunction<&fl_sparse_fallback>());
}

}  // namespace
