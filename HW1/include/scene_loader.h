#ifndef SCENE_LOADER_H
#define SCENE_LOADER_H

#include "camera.h"
#include "transform.h"
#include "vec3.h"

#include <string>
#include <vector>

// Global render settings (corresponds to JSON "settings")
struct RenderSettings {
    int max_bounces = 8;
};

// Light parameters (corresponds to JSON "light")
struct LightParams {
    camera::point3 position{-3.0f, 0.0f, 1.0f};
    camera::vec3 color{1.0f, 1.0f, 1.0f};
};

// Camera parameters (corresponds to JSON "camera")
struct CameraParams {
    double focal_length_mm = 50.0;
    double sensor_height_mm = 24.0;
    int pixel_width = 320;
    int pixel_height = 180;
    camera::point3 position{0.0f, 0.0f, 0.0f};
    camera::point3 look_at{0.0f, 0.0f, 0.0f};
    camera::vec3 up{0.0f, 0.0f, 1.0f};
};

// A node in the scene graph (one entry in the JSON "scene" array)
struct SceneNode {
    std::string name;
    std::string type;  // e.g. "mesh"
    std::string path;  // e.g. "./assets/meshes/frog.obj"
    Transform transform; // per-mesh transform (optional in JSON)
};

// Full scene config: settings + camera + light + scene
struct SceneConfig {
    RenderSettings settings;
    CameraParams camera;
    LightParams light;
    std::vector<SceneNode> scene;
};

// Load scene definition from JSON file
class SceneLoader {
public:
    // Read JSON from json_path and parse into SceneConfig.
    // base_dir: base directory for resolving relative paths (e.g. config dir or project root), default ".".
    static SceneConfig load(const std::string& json_path,
                            const std::string& base_dir = ".");

    // Construct camera instance from CameraParams
    static camera make_camera(const CameraParams& p);
};

#endif