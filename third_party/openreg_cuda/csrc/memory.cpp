#include <include/openreg.h>
#include <cuda_runtime.h>

static orError_t from_cuda(cudaError_t err) {
  if (err == cudaSuccess) return orSuccess;
  if (err == cudaErrorNotReady) return orErrorNotReady;
  return orErrorUnknown;
}

orError_t orMalloc(void** devPtr, size_t size) {
  if (!devPtr || size == 0) return orErrorUnknown;
  return from_cuda(cudaMalloc(devPtr, size));
}

orError_t orFree(void* devPtr) {
  return from_cuda(cudaFree(devPtr));
}

orError_t orMallocHost(void** hostPtr, size_t size) {
  if (!hostPtr || size == 0) return orErrorUnknown;
  return from_cuda(cudaMallocHost(hostPtr, size));
}

orError_t orFreeHost(void* hostPtr) {
  return from_cuda(cudaFreeHost(hostPtr));
}

static cudaMemcpyKind to_cuda_kind(orMemcpyKind kind) {
  switch (kind) {
    case orMemcpyHostToHost:     return cudaMemcpyHostToHost;
    case orMemcpyHostToDevice:   return cudaMemcpyHostToDevice;
    case orMemcpyDeviceToHost:   return cudaMemcpyDeviceToHost;
    case orMemcpyDeviceToDevice: return cudaMemcpyDeviceToDevice;
    default:                     return cudaMemcpyDefault;
  }
}

orError_t orMemcpy(void* dst, const void* src, size_t count, orMemcpyKind kind) {
  return from_cuda(cudaMemcpy(dst, src, count, to_cuda_kind(kind)));
}

orError_t orMemcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    orMemcpyKind kind,
    orStream_t stream) {
  if (!stream) return orErrorUnknown;
  return from_cuda(cudaMemcpyAsync(
      dst, src, count, to_cuda_kind(kind),
      reinterpret_cast<cudaStream_t>(stream)));
}

orError_t orPointerGetAttributes(orPointerAttributes* attributes, const void* ptr) {
  if (!attributes || !ptr) return orErrorUnknown;

  cudaPointerAttributes cuda_attr{};
  cudaError_t err = cudaPointerGetAttributes(&cuda_attr, ptr);
  if (err != cudaSuccess) return orErrorUnknown;

  attributes->pointer = const_cast<void*>(ptr);
  attributes->device = cuda_attr.device;

  switch (cuda_attr.type) {
    case cudaMemoryTypeHost:
      attributes->type = orMemoryTypeHost;
      break;
    case cudaMemoryTypeDevice:
      attributes->type = orMemoryTypeDevice;
      break;
    default:
      attributes->type = orMemoryTypeUnmanaged;
      break;
  }
  return orSuccess;
}

// No-ops for CUDA: memory protection is not used
orError_t orMemoryUnprotect(void* devPtr) {
  (void)devPtr;
  return orSuccess;
}

orError_t orMemoryProtect(void* devPtr) {
  (void)devPtr;
  return orSuccess;
}
