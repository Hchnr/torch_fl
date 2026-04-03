#include <include/openreg.h>
#include <cuda_runtime.h>

// Map orError_t from cudaError_t
static orError_t from_cuda(cudaError_t err) {
  if (err == cudaSuccess) return orSuccess;
  if (err == cudaErrorNotReady) return orErrorNotReady;
  return orErrorUnknown;
}

orError_t orGetDeviceCount(int* count) {
  if (!count) return orErrorUnknown;
  return from_cuda(cudaGetDeviceCount(count));
}

orError_t orGetDevice(int* device) {
  if (!device) return orErrorUnknown;
  return from_cuda(cudaGetDevice(device));
}

orError_t orSetDevice(int device) {
  return from_cuda(cudaSetDevice(device));
}

orError_t orDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
  return from_cuda(cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority));
}

orError_t orDeviceSynchronize(void) {
  return from_cuda(cudaDeviceSynchronize());
}
