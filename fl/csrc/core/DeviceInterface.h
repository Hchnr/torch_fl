#pragma once
#include <c10/core/Device.h>

namespace fl {

// 后端在编译时实现这些函数（编译时多态，无虚函数开销）
// NPU 实现: backends/npu/csrc/core/NPUFunctions.cpp
// MLU 实现: backends/mlu/csrc/core/MLUFunctions.cpp

// === Device ===
int GetDevice(int* device);
int SetDevice(int device);
int DeviceCount();
int Synchronize(int device);
int ExchangeDevice(int device);

// === Memory ===
int Malloc(void** ptr, size_t size);
int Free(void* ptr);
int MemGetInfo(size_t* free, size_t* total, int device);

// === 后端信息 ===
const char* GetBackendName();
const char* GetDeviceName(int device);

}  // namespace fl
