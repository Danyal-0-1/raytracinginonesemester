#pragma once

#include <memory>
#include <vector>

#include "ILight.h"
#include "scene.h"

struct LightParseSummary {
    int loaded = 0;
    int skipped = 0;
    int failed = 0;
};

// Reads top-level "lights": [ ... ] from SceneIO::JsonValue root and appends
// polymorphic lights to out_lights. Non-throwing by contract.
LightParseSummary ParseTopLevelLights(
    const SceneIO::JsonValue& root,
    std::vector<std::unique_ptr<ILight>>& out_lights) noexcept;

