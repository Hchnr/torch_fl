#include <include/openreg.h>
#include <cuda_runtime.h>

#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <atomic>
#include <condition_variable>

static orError_t from_cuda(cudaError_t err) {
  if (err == cudaSuccess) return orSuccess;
  if (err == cudaErrorNotReady) return orErrorNotReady;
  return orErrorUnknown;
}

// orStream wraps a real cudaStream_t plus a host-side task queue for
// orLaunchKernel (used by orMemcpyAsync in the CPU openreg backend).
// For the CUDA backend we execute tasks directly on the CUDA stream thread.
struct orStream {
  cudaStream_t cuda_stream{nullptr};
  int device_index{-1};

  // Host worker for orLaunchKernel tasks
  std::queue<std::function<void()>> tasks;
  std::mutex mtx;
  std::condition_variable cv;
  std::thread worker;
  std::atomic<bool> stop_flag{false};

  orStream() {
    worker = std::thread([this] {
      while (true) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(mtx);
          cv.wait(lock, [this] { return stop_flag.load() || !tasks.empty(); });
          if (stop_flag.load() && tasks.empty()) return;
          task = std::move(tasks.front());
          tasks.pop();
        }
        task();
      }
    });
  }

  ~orStream() {
    {
      std::lock_guard<std::mutex> lock(mtx);
      stop_flag.store(true);
    }
    cv.notify_one();
    worker.join();
    if (cuda_stream) cudaStreamDestroy(cuda_stream);
  }
};

struct orEvent {
  cudaEvent_t cuda_event{nullptr};
};

namespace openreg {
orError_t addTaskToStream(orStream* stream, std::function<void()> task) {
  if (!stream) return orErrorUnknown;
  {
    std::lock_guard<std::mutex> lock(stream->mtx);
    stream->tasks.push(std::move(task));
  }
  stream->cv.notify_one();
  return orSuccess;
}
} // namespace openreg

// Stream
orError_t orStreamCreateWithPriority(orStream_t* stream, unsigned int flags, int priority) {
  if (!stream) return orErrorUnknown;
  auto* s = new orStream();
  int cur = 0;
  cudaGetDevice(&cur);
  s->device_index = cur;
  cudaError_t err = cudaStreamCreateWithPriority(&s->cuda_stream, flags, priority);
  if (err != cudaSuccess) { delete s; return orErrorUnknown; }
  *stream = s;
  return orSuccess;
}

orError_t orStreamCreate(orStream_t* stream) {
  return orStreamCreateWithPriority(stream, cudaStreamDefault, 0);
}

orError_t orStreamGetPriority(orStream_t stream, int* priority) {
  if (!stream || !priority) return orErrorUnknown;
  return from_cuda(cudaStreamGetPriority(stream->cuda_stream, priority));
}

orError_t orStreamDestroy(orStream_t stream) {
  if (!stream) return orErrorUnknown;
  delete stream;
  return orSuccess;
}

orError_t orStreamQuery(orStream_t stream) {
  if (!stream) return orErrorUnknown;
  return from_cuda(cudaStreamQuery(stream->cuda_stream));
}

orError_t orStreamSynchronize(orStream_t stream) {
  if (!stream) return orErrorUnknown;
  return from_cuda(cudaStreamSynchronize(stream->cuda_stream));
}

orError_t orStreamWaitEvent(orStream_t stream, orEvent_t event, unsigned int flags) {
  if (!stream || !event) return orErrorUnknown;
  return from_cuda(cudaStreamWaitEvent(stream->cuda_stream, event->cuda_event, flags));
}

// Event
orError_t orEventCreateWithFlags(orEvent_t* event, unsigned int flags) {
  if (!event) return orErrorUnknown;
  auto* e = new orEvent();
  // orEventEnableTiming == 0x1; cudaEventDisableTiming == 0x2
  unsigned int cuda_flags = (flags & orEventEnableTiming) ? cudaEventDefault : cudaEventDisableTiming;
  cudaError_t err = cudaEventCreateWithFlags(&e->cuda_event, cuda_flags);
  if (err != cudaSuccess) { delete e; return orErrorUnknown; }
  *event = e;
  return orSuccess;
}

orError_t orEventCreate(orEvent_t* event) {
  return orEventCreateWithFlags(event, orEventDisableTiming);
}

orError_t orEventDestroy(orEvent_t event) {
  if (!event) return orErrorUnknown;
  cudaError_t err = cudaEventDestroy(event->cuda_event);
  delete event;
  return from_cuda(err);
}

orError_t orEventRecord(orEvent_t event, orStream_t stream) {
  if (!event || !stream) return orErrorUnknown;
  return from_cuda(cudaEventRecord(event->cuda_event, stream->cuda_stream));
}

orError_t orEventSynchronize(orEvent_t event) {
  if (!event) return orErrorUnknown;
  return from_cuda(cudaEventSynchronize(event->cuda_event));
}

orError_t orEventQuery(orEvent_t event) {
  if (!event) return orErrorUnknown;
  return from_cuda(cudaEventQuery(event->cuda_event));
}

orError_t orEventElapsedTime(float* ms, orEvent_t start, orEvent_t end) {
  if (!ms || !start || !end) return orErrorUnknown;
  return from_cuda(cudaEventElapsedTime(ms, start->cuda_event, end->cuda_event));
}
