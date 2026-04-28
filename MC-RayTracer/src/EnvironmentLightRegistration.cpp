#include "EnvironmentLight.h"
#include "LightFactory.h"

#include <memory>
#include <variant>

namespace {
const bool kRegisteredEnvironmentLight = LightFactory::Register(
    "environment_light",
    [](const LightParams& params) -> std::unique_ptr<ILight> {
        if (const auto* p = std::get_if<EnvironmentLightParams>(&params)) {
            return std::make_unique<EnvironmentLight>(*p);
        }
        return nullptr;
    });
} // namespace

