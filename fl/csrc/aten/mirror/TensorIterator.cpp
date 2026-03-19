/**
 * AcceleratorTensorIterator mirror 实现 — 骨架。
 * 实际实现需从 torch_npu 的 NPUTensorIterator 迁移。
 */
#include "fl/csrc/aten/mirror/TensorIterator.h"

namespace fl {

std::tuple<at::ScalarType, c10::IntArrayRef> AcceleratorTensorIterator::binary_op(
    at::Tensor& out, const at::Tensor& a, const at::Tensor& b,
    bool check_mem_overlap) {
    // TODO: 从 NPU NPUTensorIterator::binary_op 迁移
    auto dtype = at::result_type(a, b);
    auto shape = at::infer_size(a.sizes(), b.sizes());
    return {dtype, shape};
}

std::tuple<at::ScalarType, c10::IntArrayRef> AcceleratorTensorIterator::unary_op(
    at::Tensor& out, const at::Tensor& a, bool check_mem_overlap) {
    // TODO: 从 NPU NPUTensorIterator::unary_op 迁移
    return {a.scalar_type(), a.sizes()};
}

std::tuple<at::ScalarType, c10::IntArrayRef> AcceleratorTensorIterator::comparison_op(
    at::Tensor& out, const at::Tensor& a, const at::Tensor& b,
    bool check_mem_overlap) {
    // TODO: 从 NPU NPUTensorIterator::comparison_op 迁移
    auto shape = at::infer_size(a.sizes(), b.sizes());
    return {at::kBool, shape};
}

std::tuple<at::ScalarType, c10::IntArrayRef> AcceleratorTensorIterator::reduce_op(
    at::Tensor& out, const at::Tensor& a) {
    // TODO: 从 NPU NPUTensorIterator::reduce_op 迁移
    return {a.scalar_type(), out.sizes()};
}

void AcceleratorTensorIterator::nullary_op(at::Tensor& out) {
    // TODO: 从 NPU NPUTensorIterator::nullary_op 迁移
}

}  // namespace fl
