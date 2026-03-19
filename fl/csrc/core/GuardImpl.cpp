#include <c10/core/impl/DeviceGuardImplInterface.h>
#include "fl/csrc/core/DeviceInterface.h"

namespace fl {

struct AcceleratorGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

    c10::DeviceType type() const override {
        return c10::DeviceType::PrivateUse1;
    }

    c10::Device exchangeDevice(c10::Device d) const override {
        auto old = ExchangeDevice(d.index());
        return c10::Device(c10::DeviceType::PrivateUse1, old);
    }

    c10::Device getDevice() const override {
        int device = -1;
        GetDevice(&device);
        return c10::Device(c10::DeviceType::PrivateUse1, device);
    }

    void setDevice(c10::Device d) const override {
        SetDevice(d.index());
    }

    void uncheckedSetDevice(c10::Device d) const noexcept override {
        SetDevice(d.index());
    }

    c10::Stream getStream(c10::Device d) const override {
        // TODO: 委托到后端 Stream 实现
        return c10::Stream(c10::Stream::DEFAULT, d);
    }

    c10::Stream getDefaultStream(c10::Device d) const override {
        return c10::Stream(c10::Stream::DEFAULT, d);
    }

    c10::Stream exchangeStream(c10::Stream s) const override {
        // TODO: 委托到后端 Stream 实现
        return s;
    }

    c10::DeviceIndex deviceCount() const noexcept override {
        return static_cast<c10::DeviceIndex>(DeviceCount());
    }

    // Event 方法 — 后续从各后端迁移
    void destroyEvent(void* event, const c10::DeviceIndex device_index)
        const noexcept override {
        // TODO: 委托到后端 Event 实现
    }

    void record(
        void** event,
        const c10::Stream& stream,
        const c10::DeviceIndex device_index,
        const c10::EventFlag flag) const override {
        // TODO: 委托到后端 Event 实现
    }

    void block(void* event, const c10::Stream& stream) const override {
        // TODO: 委托到后端 Event 实现
    }

    bool queryEvent(void* event) const override {
        // TODO: 委托到后端 Event 实现
        return true;
    }

    bool queryStream(const c10::Stream& stream) const override {
        return true;
    }

    void synchronizeStream(const c10::Stream& stream) const override {
        // TODO: 委托到后端 Stream 实现
    }

    void synchronizeEvent(void* event) const override {
        // TODO: 委托到后端 Event 实现
    }
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, AcceleratorGuardImpl);

}  // namespace fl
