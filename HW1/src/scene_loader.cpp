#include "scene_loader.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

namespace {

camera::point3 parse_vec3(const json& j) {
    if (!j.is_array() || j.size() < 3)
        throw std::runtime_error("Expected array of 3 numbers for vec3");
    return camera::point3{
        static_cast<float>(j[0].get<double>()),
        static_cast<float>(j[1].get<double>()),
        static_cast<float>(j[2].get<double>())
    };
}

Vec3 parse_vec3_v(const json& j) {
    camera::point3 p = parse_vec3(j);
    return make_vec3(p.x, p.y, p.z);
}

Transform parse_transform(const json& node) {
    Transform t{};
    if (!node.contains("transform")) return t;

    const auto& tr = node["transform"];
    if (!tr.is_object()) return t;

    if (tr.contains("position")) {
        t.position = parse_vec3_v(tr["position"]);
    }
    if (tr.contains("rotation")) {
        t.rotation_deg = parse_vec3_v(tr["rotation"]);
    }
    if (tr.contains("scale")) {
        const auto& sc = tr["scale"];
        if (sc.is_number()) {
            const float s = static_cast<float>(sc.get<double>());
            t.scale = make_vec3(s, s, s);
        } else {
            t.scale = parse_vec3_v(sc);
        }
    }
    return t;
}

std::string resolve_path(const std::string& base_dir, const std::string& path) {
    if (path.empty()) return path;
    if (path.size() >= 2 && path[0] == '.' && (path[1] == '/' || path[1] == '\\'))
        return base_dir + "/" + path.substr(2);
    if (path[0] == '/' || (path.size() >= 2 && path[1] == ':'))
        return path; // absolute
    return base_dir + "/" + path;
}

} // namespace

SceneConfig SceneLoader::load(const std::string& json_path,
                              const std::string& base_dir) {
    std::ifstream f(json_path);
    if (!f.is_open()) {
        throw std::runtime_error("SceneLoader: cannot open file: " + json_path);
    }
    json j;
    try {
        f >> j;
    } catch (const json::exception& e) {
        throw std::runtime_error(std::string("SceneLoader: JSON parse error: ") + e.what());
    }

    SceneConfig config;

    // settings
    if (j.contains("settings")) {
        const auto& s = j["settings"];
        if (s.contains("max_bounces"))
            config.settings.max_bounces = s["max_bounces"].get<int>();
    }

    // camera
    if (j.contains("camera")) {
        const auto& c = j["camera"];
        if (c.contains("focal_length_mm"))
            config.camera.focal_length_mm = c["focal_length_mm"].get<double>();
        if (c.contains("sensor_height_mm"))
            config.camera.sensor_height_mm = c["sensor_height_mm"].get<double>();
        if (c.contains("pixel_width"))
            config.camera.pixel_width = c["pixel_width"].get<int>();
        if (c.contains("pixel_height"))
            config.camera.pixel_height = c["pixel_height"].get<int>();
        if (c.contains("position"))
            config.camera.position = parse_vec3(c["position"]);
        if (c.contains("look_at"))
            config.camera.look_at = parse_vec3(c["look_at"]);
        if (c.contains("up"))
            config.camera.up = parse_vec3(c["up"]);
    }

    // light
    if (j.contains("light")) {
        const auto& l = j["light"];
        if (l.contains("position"))
            config.light.position = parse_vec3(l["position"]);
        if (l.contains("color"))
            config.light.color = parse_vec3(l["color"]);
    }

    // scene
    if (j.contains("scene") && j["scene"].is_array()) {
        for (const auto& node : j["scene"]) {
            SceneNode sn;
            if (node.contains("name")) sn.name = node["name"].get<std::string>();
            if (node.contains("type")) sn.type = node["type"].get<std::string>();
            if (node.contains("path")) {
                std::string raw = node["path"].get<std::string>();
                sn.path = resolve_path(base_dir, raw);
            }
            sn.transform = parse_transform(node);
            config.scene.push_back(std::move(sn));
        }
    }

    return config;
}

camera SceneLoader::make_camera(const CameraParams& p) {
    return camera(p.position, p.look_at, p.up,
                  p.focal_length_mm, p.sensor_height_mm,
                  p.pixel_width, p.pixel_height);
}