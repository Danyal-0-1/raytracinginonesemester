#pragma once

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "vec3.h"
#include "camera.h"
#include "material.h"
#include "environment.h"

struct SceneSettings {
    int max_depth = 1;
    int spp = 1;
    bool diffuse_bounce = true;
};

struct Light {
    Vec3 position = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3 color = make_vec3(1.0f, 1.0f, 1.0f);
    int intensity = 1;
};

struct SceneObject {
    std::string name;
    std::string type;
    std::string path;
    Vec3 position = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3 rotation = make_vec3(0.0f, 0.0f, 0.0f); // degrees
    Vec3 scale = make_vec3(1.0f, 1.0f, 1.0f);
    Material material;
};

struct Scene {
    SceneSettings settings;
    Camera camera;
    Vec3 miss_color = make_vec3(0.0f, 0.0f, 0.0f);
    Environment environment;
    std::vector<Light> lights;
    std::vector<SceneObject> objects;
};

namespace SceneIO {

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;
};

struct JsonParser {
    const std::string* s = nullptr;
    size_t i = 0;
    std::string err;

    void skip_ws() {
        while (i < s->size() && std::isspace(static_cast<unsigned char>((*s)[i]))) ++i;
    }

    bool parse_value(JsonValue& out) {
        skip_ws();
        if (i >= s->size()) return fail("Unexpected end of input");
        char c = (*s)[i];
        if (c == '{') return parse_object(out);
        if (c == '[') return parse_array(out);
        if (c == '"') return parse_string(out);
        if (c == 't' || c == 'f') return parse_bool(out);
        if (c == 'n') return parse_null(out);
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number(out);
        return fail("Unexpected character");
    }

    bool parse_object(JsonValue& out) {
        out.type = JsonValue::Type::Object;
        out.obj.clear();
        ++i; // skip '{'
        skip_ws();
        if (i < s->size() && (*s)[i] == '}') { ++i; return true; }
        while (i < s->size()) {
            JsonValue key;
            if (!parse_string(key)) return false;
            skip_ws();
            if (i >= s->size() || (*s)[i] != ':') return fail("Expected ':'");
            ++i;
            JsonValue val;
            if (!parse_value(val)) return false;
        out.obj.emplace_back(key.str, std::move(val));
            skip_ws();
            if (i < s->size() && (*s)[i] == ',') { ++i; skip_ws(); continue; }
            if (i < s->size() && (*s)[i] == '}') { ++i; return true; }
            return fail("Expected ',' or '}'");
        }
        return fail("Unterminated object");
    }

    bool parse_array(JsonValue& out) {
        out.type = JsonValue::Type::Array;
        out.arr.clear();
        ++i; // skip '['
        skip_ws();
        if (i < s->size() && (*s)[i] == ']') { ++i; return true; }
        while (i < s->size()) {
            JsonValue val;
            if (!parse_value(val)) return false;
            out.arr.push_back(std::move(val));
            skip_ws();
            if (i < s->size() && (*s)[i] == ',') { ++i; skip_ws(); continue; }
            if (i < s->size() && (*s)[i] == ']') { ++i; return true; }
            return fail("Expected ',' or ']'");
        }
        return fail("Unterminated array");
    }

    bool parse_string(JsonValue& out) {
        out.type = JsonValue::Type::String;
        out.str.clear();
        if ((*s)[i] != '"') return fail("Expected '\"'");
        ++i;
        while (i < s->size()) {
            char c = (*s)[i++];
            if (c == '"') return true;
            if (c == '\\') {
                if (i >= s->size()) return fail("Bad escape");
                char e = (*s)[i++];
                switch (e) {
                    case '"': out.str.push_back('"'); break;
                    case '\\': out.str.push_back('\\'); break;
                    case '/': out.str.push_back('/'); break;
                    case 'b': out.str.push_back('\b'); break;
                    case 'f': out.str.push_back('\f'); break;
                    case 'n': out.str.push_back('\n'); break;
                    case 'r': out.str.push_back('\r'); break;
                    case 't': out.str.push_back('\t'); break;
                    case 'u': {
                        if (i + 4 > s->size()) return fail("Bad unicode escape");
                        unsigned code = 0;
                        for (int k = 0; k < 4; ++k) {
                            char h = (*s)[i++];
                            code <<= 4;
                            if (h >= '0' && h <= '9') code |= (h - '0');
                            else if (h >= 'a' && h <= 'f') code |= (h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') code |= (h - 'A' + 10);
                            else return fail("Bad unicode escape");
                        }
                        if (code <= 0x7F) out.str.push_back(static_cast<char>(code));
                        else out.str.push_back('?');
                    } break;
                    default: return fail("Bad escape");
                }
            } else {
                out.str.push_back(c);
            }
        }
        return fail("Unterminated string");
    }

    bool parse_number(JsonValue& out) {
        out.type = JsonValue::Type::Number;
        const char* start = s->c_str() + i;
        char* end = nullptr;
        out.num = std::strtod(start, &end);
        if (end == start) return fail("Bad number");
        i = static_cast<size_t>(end - s->c_str());
        return true;
    }

    bool parse_bool(JsonValue& out) {
        if (s->compare(i, 4, "true") == 0) {
            out.type = JsonValue::Type::Bool;
            out.b = true;
            i += 4;
            return true;
        }
        if (s->compare(i, 5, "false") == 0) {
            out.type = JsonValue::Type::Bool;
            out.b = false;
            i += 5;
            return true;
        }
        return fail("Bad boolean");
    }

    bool parse_null(JsonValue& out) {
        if (s->compare(i, 4, "null") == 0) {
            out.type = JsonValue::Type::Null;
            i += 4;
            return true;
        }
        return fail("Bad null");
    }

    bool fail(const std::string& msg) {
        err = msg;
        return false;
    }
};

inline bool parse_json(const std::string& text, JsonValue& out, std::string* err) {
    JsonParser p;
    p.s = &text;
    if (!p.parse_value(out)) {
        if (err) *err = p.err;
        return false;
    }
    p.skip_ws();
    if (p.i != text.size()) {
        if (err) *err = "Trailing characters";
        return false;
    }
    return true;
}

inline bool json_get(const JsonValue& obj, const char* key, const JsonValue** out) {
    if (obj.type != JsonValue::Type::Object) return false;
    for (auto& kv : obj.obj) {
        if (kv.first == key) {
            *out = &kv.second;
            return true;
        }
    }
    return false;
}

inline bool json_as_vec3(const JsonValue& v, Vec3& out) {
    if (v.type != JsonValue::Type::Array || v.arr.size() != 3) return false;
    if (v.arr[0].type != JsonValue::Type::Number) return false;
    if (v.arr[1].type != JsonValue::Type::Number) return false;
    if (v.arr[2].type != JsonValue::Type::Number) return false;
    out = make_vec3(
        static_cast<float>(v.arr[0].num),
        static_cast<float>(v.arr[1].num),
        static_cast<float>(v.arr[2].num));
    return true;
}

inline std::string to_lower_ascii(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

inline EnvironmentPreset preset_from_string(const std::string& preset_str) {
    const std::string v = to_lower_ascii(preset_str);
    if (v == "custom") return EnvironmentPreset::Custom;
    if (v == "sunrise" || v == "morning") return EnvironmentPreset::Sunrise;
    if (v == "midday" || v == "noon") return EnvironmentPreset::Midday;
    if (v == "sunset" || v == "evening") return EnvironmentPreset::Sunset;
    if (v == "night" || v == "night_moon" || v == "moonlight") return EnvironmentPreset::NightMoon;
    if (v == "night_nomoon" || v == "starlight") return EnvironmentPreset::NightNoMoon;
    return EnvironmentPreset::Custom;
}

inline bool parse_clock_hhmm(const std::string& time_str, float& hour_out) {
    if (time_str.size() < 4) return false;
    size_t colon = time_str.find(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= time_str.size()) return false;
    const std::string h = time_str.substr(0, colon);
    const std::string m = time_str.substr(colon + 1);
    const int hour = std::atoi(h.c_str());
    const int minute = std::atoi(m.c_str());
    if (hour < 0 || hour > 23 || minute < 0 || minute > 59) return false;
    hour_out = static_cast<float>(hour) + static_cast<float>(minute) / 60.0f;
    return true;
}

inline bool is_tempe_az(const std::string& location) {
    const std::string v = to_lower_ascii(location);
    return v.find("tempe") != std::string::npos && v.find("az") != std::string::npos;
}

inline EnvironmentPreset tempe_preset_from_time(float hour_local) {
    if (hour_local >= 5.0f && hour_local < 10.0f) return EnvironmentPreset::Sunrise;
    if (hour_local >= 10.0f && hour_local < 16.0f) return EnvironmentPreset::Midday;
    if (hour_local >= 16.0f && hour_local < 21.0f) return EnvironmentPreset::Sunset;
    return EnvironmentPreset::NightMoon;
}

inline bool parse_scene(const JsonValue& root, Scene& scene, std::string* err) {
    if (root.type != JsonValue::Type::Object) {
        if (err) *err = "Root is not an object";
        return false;
    }

    const JsonValue* settings = nullptr;
    if (json_get(root, "settings", &settings)) {
        const JsonValue* max_bounces = nullptr;
        if (json_get(*settings, "max_bounces", &max_bounces) &&
            max_bounces->type == JsonValue::Type::Number) {
            scene.settings.max_depth = static_cast<int>(max_bounces->num);
        }
        const JsonValue* spp_val = nullptr;
        if (json_get(*settings, "spp", &spp_val) &&
            spp_val->type == JsonValue::Type::Number) {
            scene.settings.spp = static_cast<int>(spp_val->num);
            if (scene.settings.spp < 1) scene.settings.spp = 1;
        }
        const JsonValue* diffuse_bounce = nullptr;
        if (json_get(*settings, "diffuse_bounce", &diffuse_bounce) &&
            diffuse_bounce->type == JsonValue::Type::Bool) {
            scene.settings.diffuse_bounce = diffuse_bounce->b;
        }
            
    }

    const JsonValue* miss_color = nullptr;
    if (json_get(root, "miss_color", &miss_color)) {
        json_as_vec3(*miss_color, scene.miss_color);
    }

    scene.environment = Environment();
    const JsonValue* environment = nullptr;
    if (json_get(root, "environment", &environment) && environment->type == JsonValue::Type::Object) {
        Environment env = scene.environment;
        const JsonValue* v = nullptr;

        if (json_get(*environment, "enabled", &v) && v->type == JsonValue::Type::Bool) {
            env.enabled = v->b;
        }

        bool has_preset = false;
        if (json_get(*environment, "preset", &v) && v->type == JsonValue::Type::String) {
            const EnvironmentPreset preset = preset_from_string(v->str);
            ApplyEnvironmentPreset(env, preset);
            has_preset = true;
        }

        std::string location_str;
        std::string time_str;
        if (json_get(*environment, "location", &v) && v->type == JsonValue::Type::String) {
            location_str = v->str;
        }
        if (json_get(*environment, "time", &v) && v->type == JsonValue::Type::String) {
            time_str = v->str;
        }

        // Tempe, AZ heuristic when preset is omitted.
        if (!has_preset && !location_str.empty() && !time_str.empty() && is_tempe_az(location_str)) {
            float hour = 0.0f;
            if (parse_clock_hhmm(time_str, hour)) {
                ApplyEnvironmentPreset(env, tempe_preset_from_time(hour));
            }
        }

        const JsonValue* sun_elev = nullptr;
        const JsonValue* sun_azim = nullptr;
        if (json_get(*environment, "sun_elevation_deg", &sun_elev) &&
            json_get(*environment, "sun_azimuth_deg", &sun_azim) &&
            sun_elev->type == JsonValue::Type::Number &&
            sun_azim->type == JsonValue::Type::Number) {
            env.sun_dir = DirectionFromElevationAzimuthDeg(
                static_cast<float>(sun_elev->num),
                static_cast<float>(sun_azim->num));
            env.preset = EnvironmentPreset::Custom;
        }

        const JsonValue* moon_elev = nullptr;
        const JsonValue* moon_azim = nullptr;
        if (json_get(*environment, "moon_elevation_deg", &moon_elev) &&
            json_get(*environment, "moon_azimuth_deg", &moon_azim) &&
            moon_elev->type == JsonValue::Type::Number &&
            moon_azim->type == JsonValue::Type::Number) {
            env.moon_dir = DirectionFromElevationAzimuthDeg(
                static_cast<float>(moon_elev->num),
                static_cast<float>(moon_azim->num));
        }

        if (json_get(*environment, "sun_intensity", &v) && v->type == JsonValue::Type::Number) {
            env.sun_intensity = static_cast<float>(v->num);
        }
        if (json_get(*environment, "sky_intensity", &v) && v->type == JsonValue::Type::Number) {
            env.sky_intensity = static_cast<float>(v->num);
        }
        if (json_get(*environment, "turbidity", &v) && v->type == JsonValue::Type::Number) {
            env.turbidity = static_cast<float>(v->num);
        }
        if (json_get(*environment, "moon_intensity", &v) && v->type == JsonValue::Type::Number) {
            env.moon_intensity = static_cast<float>(v->num);
            if (env.moon_intensity <= 0.0f) env.moon_enabled = false;
        }
        if (json_get(*environment, "moon_enabled", &v) && v->type == JsonValue::Type::Bool) {
            env.moon_enabled = v->b;
            if (!env.moon_enabled) env.moon_intensity = 0.0f;
        }
        if (json_get(*environment, "star_intensity", &v) && v->type == JsonValue::Type::Number) {
            env.star_intensity = static_cast<float>(v->num);
        }
        if (json_get(*environment, "star_density", &v) && v->type == JsonValue::Type::Number) {
            env.star_density = static_cast<float>(v->num);
        }
        if (json_get(*environment, "star_threshold", &v) && v->type == JsonValue::Type::Number) {
            env.star_threshold = static_cast<float>(v->num);
        }
        if (json_get(*environment, "exposure", &v) && v->type == JsonValue::Type::Number) {
            env.exposure = static_cast<float>(v->num);
        }
        if (json_get(*environment, "sun_tint", &v)) {
            json_as_vec3(*v, env.sun_tint);
        }
        if (json_get(*environment, "horizon_tint", &v)) {
            json_as_vec3(*v, env.horizon_tint);
        }
        if (json_get(*environment, "tint", &v)) {
            json_as_vec3(*v, env.tint);
        }
        if (json_get(*environment, "sky_tint", &v)) {
            json_as_vec3(*v, env.tint);
        }
        if (json_get(*environment, "warm_scatter", &v) && v->type == JsonValue::Type::Number) {
            env.warm_scatter = static_cast<float>(v->num);
        }
        if (json_get(*environment, "ground_color", &v)) {
            json_as_vec3(*v, env.ground_color);
        }
        if (json_get(*environment, "ground_intensity", &v) && v->type == JsonValue::Type::Number) {
            env.ground_intensity = static_cast<float>(v->num);
        }

        scene.environment = env;
    }

    const JsonValue* camera = nullptr;
    if (json_get(root, "camera", &camera)) {
        Vec3 cam_pos = scene.camera.get_center();
        Vec3 cam_look_at = scene.camera.get_look_at();
        Vec3 cam_up = scene.camera.get_up_vector();
        double focal_length_mm = scene.camera.get_focal_length_mm();
        double sensor_height_mm = scene.camera.get_sensor_height_mm();
        int pixel_width = scene.camera.pixel_width;
        int pixel_height = scene.camera.pixel_height;

        const JsonValue* v = nullptr;
        if (json_get(*camera, "focal_length_mm", &v) && v->type == JsonValue::Type::Number)
            focal_length_mm = static_cast<double>(v->num);
        if (json_get(*camera, "sensor_height_mm", &v) && v->type == JsonValue::Type::Number)
            sensor_height_mm = static_cast<double>(v->num);
        if (json_get(*camera, "pixel_width", &v) && v->type == JsonValue::Type::Number)
            pixel_width = static_cast<int>(v->num);
        if (json_get(*camera, "pixel_height", &v) && v->type == JsonValue::Type::Number)
            pixel_height = static_cast<int>(v->num);
        if (json_get(*camera, "position", &v)) json_as_vec3(*v, cam_pos);
        if (json_get(*camera, "look_at", &v)) json_as_vec3(*v, cam_look_at);
        if (json_get(*camera, "up", &v)) json_as_vec3(*v, cam_up);

        scene.camera = Camera(
            cam_pos, cam_look_at, cam_up,
            focal_length_mm, sensor_height_mm,
            pixel_width, pixel_height
        );
    }

    scene.lights.clear();
    const JsonValue* lights = nullptr;
    if (json_get(root, "lights", &lights) && lights->type == JsonValue::Type::Array) {
        for (const auto& item : lights->arr) {
            if (item.type != JsonValue::Type::Object) continue;
            Light lc;
            const JsonValue* v = nullptr;
            if (json_get(item, "position", &v)) json_as_vec3(*v, lc.position);
            if (json_get(item, "color", &v)) json_as_vec3(*v, lc.color);
            if (json_get(item, "intensity", &v) && v->type == JsonValue::Type::Number) {
                lc.intensity = static_cast<int>(v->num);
            }
            scene.lights.push_back(lc);
        }
    }
    // Backward compatibility: allow single "light": { ... } in scene json.
    if (scene.lights.empty()) {
        const JsonValue* light = nullptr;
        if (json_get(root, "light", &light) && light->type == JsonValue::Type::Object) {
            Light lc;
            const JsonValue* v = nullptr;
            if (json_get(*light, "position", &v)) json_as_vec3(*v, lc.position);
            if (json_get(*light, "color", &v)) json_as_vec3(*v, lc.color);
            if (json_get(*light, "intensity", &v) && v->type == JsonValue::Type::Number) {
                lc.intensity = static_cast<int>(v->num);
            }
            scene.lights.push_back(lc);
        }
    }

    const JsonValue* scene_arr = nullptr;
    if (!json_get(root, "scene", &scene_arr) || scene_arr->type != JsonValue::Type::Array) {
        if (err) *err = "Missing 'scene' array";
        return false;
    }

    scene.objects.clear();
    for (const auto& item : scene_arr->arr) {
        if (item.type != JsonValue::Type::Object) continue;
        SceneObject obj;
        const JsonValue* v = nullptr;
        if (json_get(item, "name", &v) && v->type == JsonValue::Type::String) obj.name = v->str;
        if (json_get(item, "type", &v) && v->type == JsonValue::Type::String) obj.type = v->str;
        if (json_get(item, "path", &v) && v->type == JsonValue::Type::String) obj.path = v->str;
        const JsonValue* transform = nullptr;
        if (json_get(item, "transform", &transform) && transform->type == JsonValue::Type::Object) {
            if (json_get(*transform, "position", &v)) json_as_vec3(*v, obj.position);
            if (json_get(*transform, "rotation", &v)) json_as_vec3(*v, obj.rotation);
            if (json_get(*transform, "scale", &v)) json_as_vec3(*v, obj.scale);
        }
        const JsonValue* material = nullptr;
        if (json_get(item, "material", &material) && material->type == JsonValue::Type::Object) {
            if (json_get(*material, "albedo", &v)) json_as_vec3(*v, obj.material.albedo);
            if (json_get(*material, "specular_color", &v)) json_as_vec3(*v, obj.material.specularColor);
            if (json_get(*material, "emission", &v)) json_as_vec3(*v, obj.material.emission);
            if (json_get(*material, "kd", &v) && v->type == JsonValue::Type::Number) {
                obj.material.kd = static_cast<float>(v->num);
            }
            if (json_get(*material, "ks", &v) && v->type == JsonValue::Type::Number) {
                obj.material.ks = static_cast<float>(v->num);
            }
            if (json_get(*material, "shininess", &v) && v->type == JsonValue::Type::Number) {
                obj.material.shininess = static_cast<float>(v->num);
            }
            if (json_get(*material, "kr", &v) && v->type == JsonValue::Type::Number) {
                obj.material.kr = static_cast<float>(v->num);
            }
        }
        if (!obj.path.empty()) scene.objects.push_back(obj);
    }

    if (scene.objects.empty()) {
        if (err) *err = "Scene contains no valid objects";
        return false;
    }
    return true;
}

inline bool LoadSceneFromFile(const std::string& path, Scene& scene, std::string* err) {
    std::ifstream f(path);
    if (!f) {
        if (err) *err = "Failed to open scene file: " + path;
        return false;
    }
    std::stringstream buffer;
    buffer << f.rdbuf();
    JsonValue root;
    if (!parse_json(buffer.str(), root, err)) return false;
    return parse_scene(root, scene, err);
}

inline std::string dirname(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return ".";
    return path.substr(0, pos);
}

inline bool is_abs_path(const std::string& path) {
    if (path.empty()) return false;
    if (path[0] == '/' || path[0] == '\\') return true;
    if (path.size() >= 2 && std::isalpha(static_cast<unsigned char>(path[0])) && path[1] == ':') return true;
    return false;
}

inline std::string join_path(const std::string& base, const std::string& rel) {
    if (base.empty() || base == ".") return rel;
    if (!base.empty() && (base.back() == '/' || base.back() == '\\')) return base + rel;
    return base + "/" + rel;
}

} // namespace SceneIO
