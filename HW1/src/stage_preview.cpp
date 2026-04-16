#include <array>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "camera.h"
#include "MeshOBJ.h"
#include "scene_loader.h"
#include "transform.h"
#include "vec3.h"
#include "visualization.h"

#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"

using vec3f = glm::vec3;

// Build a list of pixel world positions (as glm::vec3 for Polyscope)
static std::vector<vec3f> build_pixel_points(const camera& cam) {
    std::vector<vec3f> pixels;
    pixels.reserve(static_cast<size_t>(cam.pixel_width) * static_cast<size_t>(cam.pixel_height));

    for (int j = 0; j < cam.pixel_height; ++j) {
        for (int i = 0; i < cam.pixel_width; ++i) {
            camera::point3 p = cam.get_pixel_position(i, j);
            pixels.emplace_back(viz::to_glm_vec3(p));
        }
    }
    return pixels;
}

// Build the ray network: one camera node + one pixel node per pixel, edges from camera -> pixel
static void build_ray_network(const camera& cam,
                              std::vector<vec3f>& out_nodes,
                              std::vector<std::array<size_t, 2>>& out_edges)
{
    const camera::point3 cam_center = cam.get_center();
    out_nodes.clear();
    out_edges.clear();

    const size_t cam_node_idx = 0;
    out_nodes.reserve(1 + static_cast<size_t>(cam.pixel_width) * static_cast<size_t>(cam.pixel_height));
    out_nodes.emplace_back(viz::to_glm_vec3(cam_center)); // single camera node

    // Add pixel nodes and edges (camera -> pixel)
    for (int j = 0; j < cam.pixel_height; ++j) {
        for (int i = 0; i < cam.pixel_width; ++i) {
            camera::point3 p = cam.get_pixel_position(i, j);
            size_t pixel_node_idx = out_nodes.size();
            out_nodes.emplace_back(viz::to_glm_vec3(p));
            out_edges.push_back({cam_node_idx, pixel_node_idx});
        }
    }
}

// Register data with Polyscope (point clouds and curve network)
static void register_with_polyscope(const std::vector<vec3f>& camera_point,
                                    const std::vector<vec3f>& pixel_points,
                                    const std::vector<vec3f>& ray_nodes,
                                    const std::vector<std::array<size_t,2>>& ray_edges)
{
    constexpr bool radius_is_relative = true;

    auto* cam_cloud = polyscope::registerPointCloud("Camera Position", camera_point);
    cam_cloud->setPointRadius(0.01, /*isRelative=*/radius_is_relative);

    auto* pixel_cloud = polyscope::registerPointCloud("Pixel Positions", pixel_points);
    pixel_cloud->setPointRadius(0.003, /*isRelative=*/radius_is_relative);

    auto* ray_network = polyscope::registerCurveNetwork("Camera Rays", ray_nodes, ray_edges);
    ray_network->setRadius(0.001, /*isRelative=*/radius_is_relative);
}

// Load mesh from OBJ and register to Polyscope
static void register_mesh_from_obj(const std::string& objPath, const std::string& meshName, const Transform& xf) {
    MeshSOA mesh;
    if (!LoadOBJ_ToMeshSOA(objPath, mesh)) {
        std::cerr << "Failed to load OBJ: " << objPath << "\n";
        return;
    }
    ApplyTransformToMeshSOA(mesh, xf);

    // Vertices: MeshSOA.positions (Vec3) -> glm::vec3
    std::vector<vec3f> vertices;
    vertices.reserve(mesh.positions.size());
    for (const auto& p : mesh.positions) {
        vertices.emplace_back(p.x, p.y, p.z);
    }

    // Faces: MeshSOA.indices are 3*uint32_t per triangle -> Fx3
    const size_t numTris = mesh.indices.size() / 3u;
    std::vector<std::array<size_t, 3>> faces;
    faces.reserve(numTris);
    for (size_t i = 0; i < numTris; ++i) {
        faces.push_back({
            static_cast<size_t>(mesh.indices[3u * i + 0u]),
            static_cast<size_t>(mesh.indices[3u * i + 1u]),
            static_cast<size_t>(mesh.indices[3u * i + 2u])
        });
    }

    polyscope::registerSurfaceMesh(meshName, vertices, faces);
}

static void print_pixel_positions(const camera& cam) {
    std::ostringstream oss;
    for (int j = 0; j < cam.pixel_height; ++j) {
        for (int i = 0; i < cam.pixel_width; ++i) {
            camera::point3 p = cam.get_pixel_position(i, j);
            oss << "Pixel (" << i << ", " << j << "): ("
                << p.x << ", " << p.y << ", " << p.z << ")\n";
        }
    }
    std::cout << oss.str();
}

int main(int argc, char** argv) {
    // Load scene config from JSON, default config/frog.json
    std::string config_path = "config/frog.json";
    if (argc >= 2) config_path = argv[1];

    std::string base_dir = ".";
    SceneConfig config;
    try {
        config = SceneLoader::load(config_path, base_dir);
    } catch (const std::exception& e) {
        std::cerr << "SceneLoader error: " << e.what() << "\n";
        return 1;
    }

    camera cam = SceneLoader::make_camera(config.camera);

    // Initializing Polyscope using visualization helper
    viz::init_polyscope_zup();

    // Camera point cloud (single point)
    const camera::point3 cam_center = cam.get_center();
    std::vector<vec3f> camera_point;
    camera_point.emplace_back(viz::to_glm_vec3(cam_center));

    // Pixel points
    const std::vector<vec3f> pixel_points = build_pixel_points(cam);

    // Ray network
    std::vector<vec3f> ray_nodes;
    std::vector<std::array<size_t,2>> ray_edges;
    build_ray_network(cam, ray_nodes, ray_edges);

    // // Print pixel coordinates to console
    // print_pixel_positions(cam);

    // Register with Polyscope
    register_with_polyscope(camera_point, pixel_points, ray_nodes, ray_edges);

    // Import meshes from scene config and visualize
    for (const auto& node : config.scene) {
        if (node.type == "mesh" && !node.path.empty()) {
            register_mesh_from_obj(node.path, node.name.empty() ? "mesh" : node.name, node.transform);
        }
    }

    // Light as a single colored point
    std::vector<vec3f> light_point{viz::to_glm_vec3(config.light.position)};
    auto* light_cloud = polyscope::registerPointCloud("Light", light_point);
    light_cloud->setPointRadius(0.02, /*isRelative=*/true);
    light_cloud->setPointColor(glm::vec3(
        static_cast<float>(config.light.color.x),
        static_cast<float>(config.light.color.y),
        static_cast<float>(config.light.color.z)));

    // Add axes and show UI
    constexpr double axis_length = 0.5;
    viz::AxesVizOptions axes_opt;
    axes_opt.radius_is_relative = true;
    axes_opt.axis_radius = 0.005;
    viz::register_axes(axis_length, glm::vec3(0.0f, 0.0f, 0.0f), axes_opt);

    viz::show();

    return 0;
}