#include "LightJsonParser.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <optional>
#include <string>
#include <variant>

#include "EnvironmentLight.h"
#include "LightFactory.h"

namespace {
std::optional<float> read_number(const SceneIO::JsonValue& obj, const char* key) {
    const SceneIO::JsonValue* v = nullptr;
    if (!SceneIO::json_get(obj, key, &v) || v->type != SceneIO::JsonValue::Type::Number) return std::nullopt;
    return static_cast<float>(v->num);
}

std::optional<bool> read_bool(const SceneIO::JsonValue& obj, const char* key) {
    const SceneIO::JsonValue* v = nullptr;
    if (!SceneIO::json_get(obj, key, &v) || v->type != SceneIO::JsonValue::Type::Bool) return std::nullopt;
    return v->b;
}

std::optional<std::string> read_string(const SceneIO::JsonValue& obj, const char* key) {
    const SceneIO::JsonValue* v = nullptr;
    if (!SceneIO::json_get(obj, key, &v) || v->type != SceneIO::JsonValue::Type::String) return std::nullopt;
    return v->str;
}

std::optional<Vec3f> read_vec3(const SceneIO::JsonValue& obj, const char* key) {
    const SceneIO::JsonValue* v = nullptr;
    Vec3f out{};
    if (!SceneIO::json_get(obj, key, &v)) return std::nullopt;
    if (!SceneIO::json_as_vec3(*v, out)) return std::nullopt;
    return out;
}

std::optional<std::array<int, 2>> read_res2(const SceneIO::JsonValue& obj, const char* key) {
    const SceneIO::JsonValue* v = nullptr;
    if (!SceneIO::json_get(obj, key, &v)) return std::nullopt;

    if (v->type == SceneIO::JsonValue::Type::Number) {
        const int s = std::max(1, static_cast<int>(v->num));
        return std::array<int, 2>{s, std::max(1, s / 2)};
    }

    if (v->type == SceneIO::JsonValue::Type::Array && v->arr.size() == 2 &&
        v->arr[0].type == SceneIO::JsonValue::Type::Number &&
        v->arr[1].type == SceneIO::JsonValue::Type::Number) {
        const int w = std::max(1, static_cast<int>(v->arr[0].num));
        const int h = std::max(1, static_cast<int>(v->arr[1].num));
        return std::array<int, 2>{w, h};
    }
    return std::nullopt;
}

bool parse_environment_light_params(const SceneIO::JsonValue& item,
                                    EnvironmentLightParams& out,
                                    std::string& err) {
    if (item.type != SceneIO::JsonValue::Type::Object) {
        err = "light entry must be an object";
        return false;
    }

    if (const auto enabled = read_bool(item, "enabled")) out.enabled = *enabled;
    if (const auto path = read_string(item, "hdri_path")) out.hdri_path = *path;
    if (const auto intensity = read_number(item, "intensity_scale")) out.intensity_scale = *intensity;
    if (const auto tint = read_vec3(item, "tint")) out.tint = *tint;
    if (const auto rot = read_vec3(item, "rotation_euler_deg")) out.rotation_euler_deg = *rot;
    if (const auto res = read_res2(item, "importance_sampling_resolution")) out.importance_sampling_resolution = *res;

    if (out.enabled && out.hdri_path.empty()) {
        err = "environment_light requires non-empty 'hdri_path' when enabled=true";
        return false;
    }
    return true;
}
} // namespace

LightParseSummary ParseTopLevelLights(const SceneIO::JsonValue& root,
                                      std::vector<std::unique_ptr<ILight>>& out_lights) noexcept {
    LightParseSummary summary{};
    try {
        if (root.type != SceneIO::JsonValue::Type::Object) return summary;

        const SceneIO::JsonValue* lights = nullptr;
        if (!SceneIO::json_get(root, "lights", &lights)) return summary;
        if (lights->type != SceneIO::JsonValue::Type::Array) {
            std::fprintf(stderr, "[LightJsonParser] top-level 'lights' exists but is not an array\n");
            summary.failed += 1;
            return summary;
        }

        for (size_t i = 0; i < lights->arr.size(); ++i) {
            const SceneIO::JsonValue& item = lights->arr[i];
            if (item.type != SceneIO::JsonValue::Type::Object) {
                summary.skipped += 1;
                continue;
            }

            const auto type = read_string(item, "type");
            if (!type.has_value()) {
                std::fprintf(stderr, "[LightJsonParser] lights[%zu] missing string 'type'\n", i);
                summary.failed += 1;
                continue;
            }

            if (*type != "environment_light") {
                summary.skipped += 1;
                continue;
            }

            EnvironmentLightParams params{};
            std::string parse_err;
            if (!parse_environment_light_params(item, params, parse_err)) {
                std::fprintf(stderr, "[LightJsonParser] lights[%zu] parse error: %s\n", i, parse_err.c_str());
                summary.failed += 1;
                continue;
            }

            auto light = LightFactory::Create(*type, LightParams{params});
            if (!light) {
                std::fprintf(stderr, "[LightJsonParser] lights[%zu] factory create failed for type=%s\n", i, type->c_str());
                summary.failed += 1;
                continue;
            }

            out_lights.emplace_back(std::move(light));
            summary.loaded += 1;
        }
    } catch (...) {
        std::fprintf(stderr, "[LightJsonParser] unexpected failure while parsing top-level lights\n");
        summary.failed += 1;
    }

    return summary;
}
