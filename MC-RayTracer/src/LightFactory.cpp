#include "LightFactory.h"

#include <mutex>
#include <unordered_map>

namespace {
using Registry = std::unordered_map<std::string, LightFactory::Creator>;

Registry& GetRegistry() {
    static Registry registry;
    return registry;
}

std::mutex& GetRegistryMutex() {
    static std::mutex m;
    return m;
}
} // namespace

bool LightFactory::Register(const std::string& type, Creator creator) {
    if (type.empty() || !creator) return false;
    std::lock_guard<std::mutex> lock(GetRegistryMutex());
    GetRegistry()[type] = std::move(creator);
    return true;
}

std::unique_ptr<ILight> LightFactory::Create(const std::string& type, const LightParams& params) {
    std::lock_guard<std::mutex> lock(GetRegistryMutex());
    const auto it = GetRegistry().find(type);
    if (it == GetRegistry().end()) return nullptr;
    return it->second(params);
}

