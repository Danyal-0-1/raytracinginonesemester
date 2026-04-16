// src/render.cpp
#include "MeshOBJ.h"
#include "camera.h"
#include "ray.h"
#include "raytracer.h"
#include "vec3.h"
#include "material.h"
#include "scene_loader.h"
#include "transform.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <limits>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// -------- Material presets for BRDF testing --------

// Matte diffuse (pure Lambert)
static Material matte_red() {
    Material m;
    m.type = MAT_LAMBERTIAN;
    m.albedo = make_vec3(0.8f, 0.2f, 0.2f);
    m.kd = 1.0f;
    m.ks = 0.0f;
    m.shininess = 1.0f;
    m.kr = 0.0f;
    return m;
}

// Plastic (diffuse + soft specular highlight)
static Material plastic_red() {
    Material m;
    m.type = MAT_LAMBERTIAN;
    m.albedo = make_vec3(0.8f, 0.2f, 0.2f);
    m.kd = 1.0f;
    m.ks = 0.25f;
    m.specularColor = make_vec3(0.04f, 0.04f, 0.04f);
    m.shininess = 64.0f;
    m.kr = 0.0f; // no mirror bounce yet (still gets BRDF specular highlight)
    return m;
}

// Glossy plastic (tight highlight)
static Material glossy_plastic() {
    Material m;
    m.type = MAT_LAMBERTIAN;
    m.albedo = make_vec3(0.8f, 0.2f, 0.2f);
    m.kd = 1.0f;
    m.ks = 0.5f;
    m.specularColor = make_vec3(0.04f, 0.04f, 0.04f);
    m.shininess = 128.0f;
    m.kr = 0.0f;
    return m;
}

// Brushed metal (tinted specular, no diffuse) + some mirror bounce
static Material metal_red() {
    Material m;
    m.type = MAT_METAL;
    m.albedo = make_vec3(0.8f, 0.2f, 0.2f);
    m.kd = 0.0f;
    m.ks = 1.0f;
    m.specularColor = m.albedo; // metal = colored specular
    m.shininess = 64.0f;
    m.kr = 0.35f; // recursive reflection weight
    return m;
}

// Mirror-like metal (very sharp) + strong recursion
static Material mirror_metal() {
    Material m;
    m.type = MAT_METAL;
    m.albedo = make_vec3(1.0f, 1.0f, 1.0f);
    m.kd = 0.0f;
    m.ks = 1.0f;
    m.specularColor = m.albedo;
    m.shininess = 512.0f;
    m.kr = 0.95f; // recursive reflection weight
    return m;
}

// Build triangles once per render (instead of per-pixel)
static std::vector<Triangle> build_triangles(const MeshSOA& mesh, const Material& mat) {
    std::vector<Triangle> tris;
    tris.reserve(mesh.indices.size() / 3);

    const size_t indexCount = mesh.indices.size();

    for (size_t k = 0; k < indexCount; k += 3) {
        Triangle tri;
        tri.mat = mat;

        tri.v0 = mesh.positions[mesh.indices[k]];
        tri.v1 = mesh.positions[mesh.indices[k + 1]];
        tri.v2 = mesh.positions[mesh.indices[k + 2]];

        if (mesh.hasNormals()) {
            tri.n0 = mesh.normals[mesh.indices[k]];
            tri.n1 = mesh.normals[mesh.indices[k + 1]];
            tri.n2 = mesh.normals[mesh.indices[k + 2]];
        } else {
            Vec3 faceN = unit_vector(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
            tri.n0 = tri.n1 = tri.n2 = faceN;
        }

        tris.push_back(tri);
    }

    return tris;
}

// Render + write one PNG
static void render_and_write_png(const MeshSOA& mesh,
                                 const camera& cam,
                                 const std::vector<Light>& lights,
                                 const Material& mat,
                                 const std::string& output_filename)
{
    const int pixel_width  = cam.pixel_width;
    const int pixel_height = cam.pixel_height;

    std::vector<Vec3> image(static_cast<size_t>(pixel_width) * static_cast<size_t>(pixel_height));

    const Vec3 center = cam.get_center();

    // Prebuild triangles once
    std::vector<Triangle> tris = build_triangles(mesh, mat);

    // Recursion depth: 1 = direct only, 2+ = reflections/shadow rays still work either way
    const int maxDepth = 4; // conservative for now

    for (int j = 0; j < pixel_height; ++j) {
        if (j % 10 == 0) {
            std::cout << "\r  scanlines remaining: " << (pixel_height - j) << "   " << std::flush;
        }

        for (int i = 0; i < pixel_width; ++i) {
            Ray r(center, cam.get_pixel_position(i, j) - center);

            Vec3 color = TraceRay(r, tris, lights, maxDepth);

            image[static_cast<size_t>(j) * static_cast<size_t>(pixel_width) + static_cast<size_t>(i)] = color;
        }
    }

    std::cout << "\r  writing: " << output_filename << "                         \n";

    std::vector<unsigned char> png_data(static_cast<size_t>(pixel_width) * static_cast<size_t>(pixel_height) * 3);

    for (int k = 0; k < pixel_width * pixel_height; ++k) {
        Vec3 c = clamp(image[static_cast<size_t>(k)]);
        png_data[static_cast<size_t>(k) * 3 + 0] = static_cast<unsigned char>(255.99f * c.x);
        png_data[static_cast<size_t>(k) * 3 + 1] = static_cast<unsigned char>(255.99f * c.y);
        png_data[static_cast<size_t>(k) * 3 + 2] = static_cast<unsigned char>(255.99f * c.z);
    }

    stbi_write_png(output_filename.c_str(),
                   pixel_width, pixel_height,
                   3, png_data.data(),
                   pixel_width * 3);
}

int main(int argc, char** argv)
{
    // 1) Choose scene JSON
    std::string config_path = "config/sphere.json";
    if (argc >= 2) config_path = argv[1];

    // Run from project root
    std::string base_dir = ".";

    SceneConfig config;
    try {
        config = SceneLoader::load(config_path, base_dir);
    } catch (const std::exception& e) {
        std::cerr << "SceneLoader error: " << e.what() << "\n";
        return 1;
    }

    // 2) Camera from config
    camera cam = SceneLoader::make_camera(config.camera);

    // 3) Lights from config (should hold for multiple lights)
    std::vector<Light> lights;
    {
        Light L;
        L.position = make_vec3(config.light.position.x, config.light.position.y, config.light.position.z);
        L.color    = make_vec3(config.light.color.x,    config.light.color.y,    config.light.color.z);
        lights.push_back(L);
    }

    // 4) Pick a material to apply to everything for now (simple test)
    //    Later we should add materials to JSON nodes and assign per-node.
    Material mat = plastic_red();

    // 5) Load all scene meshes and bake transforms into triangles
    std::vector<Triangle> tris;
    for (const auto& node : config.scene) {
        if (node.type != "mesh" || node.path.empty()) continue;

        std::cout << "Loading mesh node '" << node.name << "': " << node.path << "\n";

        MeshSOA mesh;
        if (!LoadOBJ_ToMeshSOA(node.path, mesh)) {
            std::cerr << "Failed to load OBJ: " << node.path << "\n";
            return 1;
        }

        // Apply node transform to mesh in-place (positions + normals)
        ApplyTransformToMeshSOA(mesh, node.transform);

        // Convert mesh to triangles once
        const size_t indexCount = mesh.indices.size();
        tris.reserve(tris.size() + indexCount / 3);

        for (size_t k = 0; k < indexCount; k += 3) {
            Triangle tri;
            tri.mat = mat;

            tri.v0 = mesh.positions[mesh.indices[k]];
            tri.v1 = mesh.positions[mesh.indices[k + 1]];
            tri.v2 = mesh.positions[mesh.indices[k + 2]];

            if (mesh.hasNormals()) {
                tri.n0 = mesh.normals[mesh.indices[k]];
                tri.n1 = mesh.normals[mesh.indices[k + 1]];
                tri.n2 = mesh.normals[mesh.indices[k + 2]];
            } else {
                Vec3 faceN = unit_vector(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                tri.n0 = tri.n1 = tri.n2 = faceN;
            }

            tris.push_back(tri);
        }
    }

    if (tris.empty()) {
        std::cerr << "Scene contains no triangles.\n";
        return 1;
    }

    // 6) Render the image
    const int pixel_width  = cam.pixel_width;
    const int pixel_height = cam.pixel_height;

    std::vector<Vec3> image(static_cast<size_t>(pixel_width) * static_cast<size_t>(pixel_height));
    const Vec3 center = cam.get_center();

    const int maxDepth = std::max(1, config.settings.max_bounces);

    for (int j = 0; j < pixel_height; ++j) {
        if (j % 10 == 0) {
            std::cout << "\r  scanlines remaining: " << (pixel_height - j) << "   " << std::flush;
        }

        for (int i = 0; i < pixel_width; ++i) {
            Ray r(center, cam.get_pixel_position(i, j) - center);
            Vec3 color = TraceRay(r, tris, lights, maxDepth);
            image[static_cast<size_t>(j) * static_cast<size_t>(pixel_width) + static_cast<size_t>(i)] = color;
        }
    }

    // 7) Write PNG
    std::string out = "scene_render.png";
    std::cout << "\r  writing: " << out << "                         \n";

    std::vector<unsigned char> png_data(static_cast<size_t>(pixel_width) * static_cast<size_t>(pixel_height) * 3);
    for (int k = 0; k < pixel_width * pixel_height; ++k) {
        Vec3 c = clamp(image[static_cast<size_t>(k)]);
        png_data[static_cast<size_t>(k) * 3 + 0] = static_cast<unsigned char>(255.99f * c.x);
        png_data[static_cast<size_t>(k) * 3 + 1] = static_cast<unsigned char>(255.99f * c.y);
        png_data[static_cast<size_t>(k) * 3 + 2] = static_cast<unsigned char>(255.99f * c.z);
    }

    stbi_write_png(out.c_str(), pixel_width, pixel_height, 3, png_data.data(), pixel_width * 3);

    std::cout << "Done.\n";
    return 0;
}

