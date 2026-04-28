#pragma once

#include <functional>
#include <memory>
#include <string>
#include <variant>

#include "EnvironmentLight.h"
#include "ILight.h"

using LightParams = std::variant<EnvironmentLightParams>;

class LightFactory {
public:
    using Creator = std::function<std::unique_ptr<ILight>(const LightParams&)>;

    static bool Register(const std::string& type, Creator creator);
    static std::unique_ptr<ILight> Create(const std::string& type, const LightParams& params);
};

