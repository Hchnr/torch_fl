#pragma once

/**
 * 统一 AcceleratorTensorIterator mirror。
 * 直接采用 NPU 的 NPUTensorIterator 实现，重命名为 AcceleratorTensorIterator。
 * 消除 MLU 的 support_tensor_iterator_bridge.diff 补丁。
 */

#include <ATen/ATen.h>
#include <tuple>

namespace fl {

struct AcceleratorTensorIterator {
    static std::tuple<at::ScalarType, c10::IntArrayRef> binary_op(
        at::Tensor& out, const at::Tensor& a, const at::Tensor& b,
        bool check_mem_overlap = false);

    static std::tuple<at::ScalarType, c10::IntArrayRef> unary_op(
        at::Tensor& out, const at::Tensor& a,
        bool check_mem_overlap = false);

    static std::tuple<at::ScalarType, c10::IntArrayRef> comparison_op(
        at::Tensor& out, const at::Tensor& a, const at::Tensor& b,
        bool check_mem_overlap = false);

    static std::tuple<at::ScalarType, c10::IntArrayRef> reduce_op(
        at::Tensor& out, const at::Tensor& a);

    static void nullary_op(at::Tensor& out);
};

}  // namespace fl
