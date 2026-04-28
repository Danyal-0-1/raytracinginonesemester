// src/main.cu

#include "MeshOBJ.h"
#include "buffers.h"
#include "bvh.h"
#include "visualizer.h"
#include "warmup.h"
#include "scene.h"
#include "camera.h"
#include "query.h"
#include "texture.h"
#include "medium.h"
#include "vec2.h"
#include "LightJsonParser.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <numeric>
#include <chrono>
#include <fstream>
#include <cmath>
#include <algorithm>

#ifdef __CUDACC__
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#define OPTIX_CHECK(call)                                                         \
    do {                                                                          \
        OptixResult res = call;                                                   \
        if (res != OPTIX_SUCCESS) {                                               \
            std::fprintf(stderr, "OptiX error: %s at %s:%d\n",                   \
                         optixGetErrorString(res), __FILE__, __LINE__);           \
        }                                                                         \
    } while (0)
#endif

// Global texture cache definition
std::unordered_map<std::string, std::unique_ptr<Texture>> g_textureCache;

#ifdef __CUDACC__
__global__ void buildTrianglesKernel(const MeshView mesh, Triangle* out, int numTriangles) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles) return;

    const uint32_t i0 = mesh.indices[idx * 3 + 0];
    const uint32_t i1 = mesh.indices[idx * 3 + 1];
    const uint32_t i2 = mesh.indices[idx * 3 + 2];

    const Vec3 v0 = mesh.positions[i0];
    const Vec3 v1 = mesh.positions[i1];
    const Vec3 v2 = mesh.positions[i2];

    Vec3 n0 = make_vec3(0,0,0), n1 = make_vec3(0,0,0), n2 = make_vec3(0,0,0);
    if (mesh.normals != nullptr) {
        n0 = mesh.normals[i0];
        n1 = mesh.normals[i1];
        n2 = mesh.normals[i2];
    }

    Vec2 uv0 = make_vec2(0.0f, 0.0f);
    Vec2 uv1 = make_vec2(0.0f, 0.0f);
    Vec2 uv2 = make_vec2(0.0f, 0.0f);
    if (mesh.uvs != nullptr) {
        uv0 = mesh.uvs[i0];
        uv1 = mesh.uvs[i1];
        uv2 = mesh.uvs[i2];
    }

    out[idx] = Triangle(v0, v1, v2, n0, n1, n2, uv0, uv1, uv2);
}
#endif

RayTracer::BVHState RayTracer::BVHState::fromChunk(char*& chunk, size_t P)
{
    BVHState state;
    obtain(chunk, state.Nodes, 2 * P - 1, 128);
    obtain(chunk, state.AABBs, 2 * P - 1, 128);
    return state;
}

static inline float deg2rad(const float d) {
    return d * 0.01745329251994329577f;
}

static inline Vec3 rotateXYZ(Vec3 v, const Vec3& rotationDeg) {
    const float rx = deg2rad(rotationDeg.x);
    const float ry = deg2rad(rotationDeg.y);
    const float rz = deg2rad(rotationDeg.z);

    const float cx = cosf(rx), sx = sinf(rx);
    const float cy = cosf(ry), sy = sinf(ry);
    const float cz = cosf(rz), sz = sinf(rz);

    v = make_vec3(v.x, cx * v.y - sx * v.z, sx * v.y + cx * v.z);
    v = make_vec3(cy * v.x + sy * v.z, v.y, -sy * v.x + cy * v.z);
    v = make_vec3(cz * v.x - sz * v.y, sz * v.x + cz * v.y, v.z);
    return v;
}

static inline void applyObjectTransform(Mesh& mesh, const SceneObject& obj) {
    for (auto& p : mesh.positions) {
        Vec3 scaled = make_vec3(p.x * obj.scale.x, p.y * obj.scale.y, p.z * obj.scale.z);
        Vec3 rotated = rotateXYZ(scaled, obj.rotation);
        p = rotated + obj.position;
    }

    for (auto& n : mesh.normals) {
        Vec3 nScaled = n;
        if (fabsf(obj.scale.x) > 1e-8f) nScaled.x /= obj.scale.x;
        if (fabsf(obj.scale.y) > 1e-8f) nScaled.y /= obj.scale.y;
        if (fabsf(obj.scale.z) > 1e-8f) nScaled.z /= obj.scale.z;
        Vec3 nRot = rotateXYZ(nScaled, obj.rotation);
        const float len2 = dot(nRot, nRot);
        if (len2 > 1e-12f) {
            n = nRot * (1.0f / sqrtf(len2));
        } else {
            n = make_vec3(0.0f, 0.0f, 1.0f);
        }
    }
}

static Vec3 lerp_vec3(const Vec3& a, const Vec3& b, float t) {
    return make_vec3(a.x + t*(b.x-a.x), a.y + t*(b.y-a.y), a.z + t*(b.z-a.z));
}

static Vec3 sample_keyframe_vec3(const std::vector<Vec3>& keys, int frame, int total_frames) {
    if (keys.size() == 1) return keys[0];
    float t = (total_frames <= 1) ? 0.0f : static_cast<float>(frame) / static_cast<float>(total_frames - 1);
    float pos_f = t * static_cast<float>(keys.size() - 1);
    int idx = static_cast<int>(pos_f);
    if (idx >= static_cast<int>(keys.size()) - 1) return keys.back();
    return lerp_vec3(keys[idx], keys[idx + 1], pos_f - static_cast<float>(idx));
}

static std::string frame_filename(const std::string& base, int frame, int total_frames) {
    if (total_frames <= 1) return base;
    size_t dot = base.rfind('.');
    std::string stem = (dot != std::string::npos) ? base.substr(0, dot) : base;
    std::string ext  = (dot != std::string::npos) ? base.substr(dot)    : "";
    char buf[16];
    std::snprintf(buf, sizeof(buf), "_%04d", frame);
    return stem + buf + ext;
}

static int animated_path_index(int frame, int total_frames, int num_paths) {
    if (num_paths <= 1) return 0;
    int interval = total_frames / num_paths;
    if (interval < 1) interval = 1;
    return std::min(frame / interval, num_paths - 1);
}

static bool loadRawScalarField(const std::string& path,
                               int nx, int ny, int nz,
                               int format,
                               std::vector<float>& out_values,
                               std::string* err)
{
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        if (err) *err = "raw grid resolution must be positive in all dimensions";
        return false;
    }

    const size_t voxel_count =
        static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (err) *err = "Failed to open raw grid file: " + path;
        return false;
    }

    out_values.clear();
    out_values.resize(voxel_count, 0.0f);

    if (format == VOLUME_DENSITY_U8) {
        std::vector<unsigned char> raw_bytes(voxel_count, 0u);
        file.read(reinterpret_cast<char*>(raw_bytes.data()), static_cast<std::streamsize>(voxel_count));
        if (file.gcount() != static_cast<std::streamsize>(voxel_count)) {
            if (err) *err = "Raw grid size does not match expected voxel count for u8 data";
            return false;
        }
        for (size_t i = 0; i < voxel_count; ++i) {
            out_values[i] = static_cast<float>(raw_bytes[i]) / 255.0f;
        }
        return true;
    }

    if (format == VOLUME_DENSITY_F32) {
        file.read(reinterpret_cast<char*>(out_values.data()),
                  static_cast<std::streamsize>(voxel_count * sizeof(float)));
        if (file.gcount() != static_cast<std::streamsize>(voxel_count * sizeof(float))) {
            if (err) *err = "Raw grid size does not match expected voxel count for f32 data";
            return false;
        }
        return true;
    }

    if (err) *err = "Unsupported raw grid format; use 'u8' or 'f32'";
    return false;
}

// ============================================================
// EnvImportanceMaps — host-side owner for the 2D CDF used by
// env_light_sample_dir / env_light_pdf in query.h.
// Built from the float HDR data using sampleHDRI's UV convention
// (u = atan2(y,x), sin-linear z → same texels as sampleHDRI).
// ============================================================
struct EnvImportanceMaps {
    std::vector<float> row_cdf_data;            // [H+1]
    std::vector<float> col_cdf_data;            // [H*(W+1)]
    std::vector<float> pmf_data;                // [W*H]
    EnvImportanceData  view;                    // POD view into the above
};

static EnvImportanceMaps BuildEnvImportance(const HDRTextureData& hdri, float az_rot) {
    EnvImportanceMaps result;
    if (!hdri.data || hdri.width <= 0 || hdri.height <= 0) return result;

    const int W = hdri.width, H = hdri.height;
    result.pmf_data.assign(static_cast<size_t>(W) * static_cast<size_t>(H), 0.0f);
    result.row_cdf_data.assign(static_cast<size_t>(H) + 1u, 0.0f);
    result.col_cdf_data.assign(static_cast<size_t>(H) * static_cast<size_t>(W + 1), 0.0f);

    // For sampleHDRI's UV convention:
    //   image row y → v_img = (y+0.5)/H (image-space, row 0 = zenith)
    //   v_in = 1 - v_img
    //   z    = sin(π·(v_in − 0.5))
    //   sin_theta = |cos(π·(v_in − 0.5))| = sqrt(1−z²)
    // The sin_theta factor makes the CDF importance proportional to solid angle.
    for (int y = 0; y < H; ++y) {
        const float v_in      = 1.0f - (static_cast<float>(y) + 0.5f) / static_cast<float>(H);
        const float sin_theta = std::fabsf(std::cosf(3.14159265359f * (v_in - 0.5f)));

        const int row_off = y * (W + 1);
        result.col_cdf_data[static_cast<size_t>(row_off)] = 0.0f;
        float row_sum = 0.0f;

        for (int x = 0; x < W; ++x) {
            const int idx = (y * W + x) * 3;
            const float r = hdri.data[idx];
            const float g = hdri.data[idx + 1];
            const float b = hdri.data[idx + 2];
            // BT.709 luminance of the linear-HDR texel
            const float lum    = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            const float weight = std::fmaxf(0.0f, lum) * sin_theta;

            result.pmf_data[static_cast<size_t>(y) * static_cast<size_t>(W)
                            + static_cast<size_t>(x)] = weight;
            row_sum += weight;
            result.col_cdf_data[static_cast<size_t>(row_off + x + 1)] = row_sum;
        }

        // Normalise conditional column CDF for this row
        if (row_sum > 0.0f) {
            const float inv = 1.0f / row_sum;
            for (int x = 1; x <= W; ++x)
                result.col_cdf_data[static_cast<size_t>(row_off + x)] *= inv;
        } else {
            for (int x = 0; x <= W; ++x)
                result.col_cdf_data[static_cast<size_t>(row_off + x)] =
                    static_cast<float>(x) / static_cast<float>(W);
        }

        result.row_cdf_data[static_cast<size_t>(y + 1)] =
            result.row_cdf_data[static_cast<size_t>(y)] + row_sum;
    }

    // Normalise marginal row CDF and PMF
    const float total = result.row_cdf_data[static_cast<size_t>(H)];
    if (total > 0.0f) {
        const float inv = 1.0f / total;
        for (float& v : result.pmf_data)    v *= inv;
        for (float& v : result.row_cdf_data) v *= inv;
        result.row_cdf_data[0]                         = 0.0f;
        result.row_cdf_data[static_cast<size_t>(H)]    = 1.0f;
    } else {
        // All-black HDR: uniform fallback
        const float uni = 1.0f / static_cast<float>(W * H);
        std::fill(result.pmf_data.begin(), result.pmf_data.end(), uni);
        for (int y2 = 0; y2 <= H; ++y2)
            result.row_cdf_data[static_cast<size_t>(y2)] =
                static_cast<float>(y2) / static_cast<float>(H);
        for (int y2 = 0; y2 < H; ++y2) {
            const int ro = y2 * (W + 1);
            for (int x = 0; x <= W; ++x)
                result.col_cdf_data[static_cast<size_t>(ro + x)] =
                    static_cast<float>(x) / static_cast<float>(W);
        }
    }

    result.view.width   = W;
    result.view.height  = H;
    result.view.valid   = true;
    result.view.az_rot  = az_rot;
    result.view.row_cdf = result.row_cdf_data.data();
    result.view.col_cdf = result.col_cdf_data.data();
    result.view.pmf     = result.pmf_data.data();
    return result;
}

int main(int argc, char** argv)
{
    using vec3 = Vec3;
    using point3 = Vec3;

    // Parse optional flags
    bool use_denoiser = false;
    std::string output_filename = "render.png";
    int nee_mode = 2;
    bool use_bdpt = false;     // BDPT integrator opt-in: --integrator bdpt
    std::vector<char*> positional_args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--denoise" || arg == "-d") {
            use_denoiser = true;
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            output_filename = argv[++i];
        } else if (arg == "--nee-mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "area")      nee_mode = 0;
            else if (mode == "brdf") nee_mode = 1;
            else if (mode == "mis")  nee_mode = 2;
            else { printf("Unknown --nee-mode '%s'. Use: area, brdf, mis\n", mode.c_str()); return 1; }
        } else if (arg == "--integrator" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "pt")        use_bdpt = false;
            else if (mode == "bdpt") use_bdpt = true;
            else { printf("Unknown --integrator '%s'. Use: pt, bdpt\n", mode.c_str()); return 1; }
        } else {
            positional_args.push_back(argv[i]);
        }
    }
    const char* nee_names[] = {"area", "brdf", "mis"};
    printf("NEE mode: %s\n", nee_names[nee_mode]);
    printf("Integrator: %s\n", use_bdpt ? "bdpt" : "pt");
    int pos_argc = static_cast<int>(positional_args.size()) + 1;

    std::vector<SceneObject> load_objects;
    Scene scene;
    bool has_scene = false;
    std::string scene_base_dir = ".";
    std::string scene_project_dir = ".";
    std::string scene_json_path;   // saved for ParseTopLevelLights re-parse
    auto file_exists = [](const std::string& p) {
        std::ifstream f(p);
        return static_cast<bool>(f);
    };
    auto resolve_scene_path = [&](const std::string& path) -> std::string {
        if (path.empty() || SceneIO::is_abs_path(path)) return path;
        const std::string scene_relative = SceneIO::join_path(scene_base_dir, path);
        std::string project_relative = path;
        if (project_relative.rfind("./", 0) == 0)
            project_relative = project_relative.substr(2);
        project_relative = SceneIO::join_path(scene_project_dir, project_relative);

        if (file_exists(scene_relative)) return scene_relative;
        if (file_exists(path)) return path;
        if (file_exists(project_relative)) return project_relative;
        return scene_relative;
    };
    if (pos_argc >= 2) {
        std::string first = positional_args[0];
        const bool is_scene =
            (first.size() >= 5 && first.substr(first.size() - 5) == ".json") ||
            (first.size() >= 6 && first.substr(first.size() - 6) == ".scene");
        if (is_scene) {
            std::string err;
            if (!SceneIO::LoadSceneFromFile(first, scene, &err)) {
                std::cerr << "Failed to load scene: " << err << "\n";
                return 1;
            }
            has_scene = true;
            scene_json_path = first;
            scene_base_dir = SceneIO::dirname(first);
            scene_project_dir = SceneIO::dirname(SceneIO::dirname(scene_base_dir));
            for (const auto& obj : scene.objects) {
                if (!obj.type.empty() && obj.type != "mesh" && obj.type != "volume") continue;
                SceneObject resolved = obj;
                resolved.path = resolve_scene_path(resolved.path);
                load_objects.push_back(resolved);
            }
        } else {
            for (const auto& parg : positional_args) {
                SceneObject obj;
                obj.path = parg;
                load_objects.push_back(obj);
            }
        }
    } else {
        SceneObject obj;
        obj.path = "../assets/meshes/frog.obj";
        load_objects.push_back(obj);
    }

    // Load meshes, MTL materials, and textures
    Mesh globalMesh;
    std::vector<Material> objectMaterials;
    std::vector<HomogeneousMedium> objectMediaList;  // per-object medium
    int nextObjectId = 0;

    // Texture management: collect all loaded TextureData for upload
    std::vector<TextureData> allTextureData;  
    // Helper to register a texture and return its index
    auto registerTexture = [&](const std::string& path) -> int {
        if (path.empty()) return -1;
        // Check if already registered
        for (int i = 0; i < (int)allTextureData.size(); ++i) {
            // Compare by pointer, same cached Texture will have same data ptr
            Texture* existing = nullptr;
            for (auto& kv : g_textureCache) {
                if (kv.second->sampled.data == allTextureData[i].data) {
                    if (kv.first == path) return i;
                }
            }
        }
        Texture* tex = LoadTexture(path);
        if (!tex) return -1;
        int idx = (int)allTextureData.size();
        allTextureData.push_back(tex->sampled);
        printf("  -> Registered texture [%d]: %s (%dx%d, %d ch)\n",
               idx, path.c_str(), tex->width, tex->height, tex->channels);
        return idx;
    };

    for (const auto& obj : load_objects)
    {
        // Handle volume-only objects (no geometry, just medium)
        if (obj.is_volume) {
            std::printf("Registering volume (no geometry): %s\n", obj.name.c_str());
            // Ensure objectMediaList is large enough
            if (objectMediaList.size() < static_cast<size_t>(nextObjectId + 1))
                objectMediaList.resize(nextObjectId + 1, HomogeneousMedium());
            // Register the medium for this "object"
            objectMediaList[nextObjectId] = obj.medium;
            if (objectMaterials.size() < static_cast<size_t>(nextObjectId + 1))
                objectMaterials.resize(nextObjectId + 1, Material());
            objectMaterials[nextObjectId] = obj.material;
            nextObjectId++;
            continue;  // Skip OBJ loading for volumes
        }
        
        if (obj.path.empty()) {
            std::printf("Skipping object '%s' with empty path (not a volume)\n", obj.name.c_str());
            continue;
        }

        const std::string& path = obj.path;
        std::printf("Loading OBJ: %s\n", path.c_str());
        Mesh tempMesh;
        const int objIdBegin = nextObjectId;

        // Load with MTL support
        std::vector<ParsedMTLMaterial> mtlMats;
        std::string mtlLibName;
        if (!LoadOBJ_ToMesh(path, tempMesh, nextObjectId,
                            obj.use_mtl ? &mtlMats : nullptr,
                            &mtlLibName))
        {
            std::cerr << "Failed to load OBJ: " << path << "\n";
            continue;
        }
        applyObjectTransform(tempMesh, obj);

        // Ensure objectMaterials and objectMediaList are large enough
        if (objectMaterials.size() < static_cast<size_t>(nextObjectId))
            objectMaterials.resize(nextObjectId, Material());
        if (objectMediaList.size() < static_cast<size_t>(nextObjectId))
            objectMediaList.resize(nextObjectId, HomogeneousMedium());

        // Apply materials: MTL first, then scene JSON overrides
        for (int oid = objIdBegin; oid < nextObjectId; ++oid) {
            Material mat = obj.material;  // start with scene JSON material
            int localIdx = oid - objIdBegin;

            // If MTL materials were loaded, merge them in
            if (obj.use_mtl && localIdx < (int)mtlMats.size() &&
                !mtlMats[localIdx].name.empty()) {
                const ParsedMTLMaterial& mtl = mtlMats[localIdx];
                mat.albedo       = mtl.Kd;
                mat.specularColor = mtl.Ks;
                mat.emission     = mtl.Ke;
                mat.shininess    = mtl.Ns;

                // Auto-detect specular weight from Ks magnitude
                float ksMax = fmaxf(mtl.Ks.x, fmaxf(mtl.Ks.y, mtl.Ks.z));
                if (ksMax > 1e-4f && mat.ks <= 0.0f) {
                    mat.ks = 0.5f;  // enable specular if MTL has Ks
                }

                // Register diffuse texture from MTL
                if (!mtl.map_Kd.empty()) {
                    mat.diffuseTexIdx = registerTexture(mtl.map_Kd);
                }
                // Register alpha/dissolve mask from MTL (leaf cards, cutouts)
                if (!mtl.map_d.empty()) {
                    mat.alphaTexIdx = registerTexture(mtl.map_d);
                }
                // Register normal map from MTL
                if (!mtl.map_Bump.empty()) {
                    mat.normalTexIdx = registerTexture(mtl.map_Bump);
                }

                // Scene JSON overrides (if the scene also specified material props,
                // those were already in obj.material, we only override textures here
                // if the scene JSON explicitly provided paths)
                if (obj.material.emission.x > 0 || obj.material.emission.y > 0 ||
                    obj.material.emission.z > 0) {
                    mat.emission = obj.material.emission;
                }
            }

            // Scene JSON texture overrides (take priority over MTL)
            if (!obj.diffuse_tex_path.empty()) {
                mat.diffuseTexIdx = registerTexture(obj.diffuse_tex_path);
            }
            if (!obj.normal_tex_path.empty()) {
                mat.normalTexIdx = registerTexture(obj.normal_tex_path);
            }
            if (!obj.alpha_tex_path.empty()) {
                mat.alphaTexIdx = registerTexture(obj.alpha_tex_path);
            }

            objectMaterials[oid] = mat;

            // Per-object medium from scene JSON
            objectMediaList[oid] = obj.medium;
        }

        std::printf("  -> Loaded %zu triangles.\n", tempMesh.indices.size() / 3);
        AppendMesh(globalMesh, tempMesh);
    }

    if (globalMesh.positions.empty()) {
        std::cerr << "No valid geometry loaded.\n";
        return 1;
    }

    const int numObjectMedia = static_cast<int>(objectMediaList.size());
    const int numTextures = static_cast<int>(allTextureData.size());
    printf("Total textures loaded: %d\n", numTextures);
    printf("Objects with media: ");
    int mediaCount = 0;
    for (int i = 0; i < numObjectMedia; ++i) {
        if (objectMediaList[i].enabled) mediaCount++;
    }
    printf("%d\n", mediaCount);

    // Extract volume regions from scene
    std::vector<VolumeRegionGPU> volumeRegionsList;
    std::vector<std::vector<float>> hostVolumeDensityBuffers;
    std::vector<std::vector<float>> hostVolumeTemperatureBuffers;
    std::vector<std::vector<float>> hostVolumeFlameBuffers;
    if (has_scene && !scene.volumes.empty()) {
        for (const auto& vol : scene.volumes) {
            VolumeRegionGPU gpuVol;
            gpuVol.min_bounds = vol.min_bounds;
            gpuVol.max_bounds = vol.max_bounds;
            gpuVol.medium = vol.medium;
            gpuVol.density_scale = vol.density_scale;
            gpuVol.density_majorant = vol.density_majorant;
            gpuVol.temperature_scale = vol.temperature_scale;
            gpuVol.flame_scale = vol.flame_scale;
            gpuVol.emission_scale = vol.emission_scale;
            gpuVol.emission_temp_min = vol.emission_temp_min;
            gpuVol.emission_temp_max = vol.emission_temp_max;

            hostVolumeDensityBuffers.emplace_back();
            hostVolumeTemperatureBuffers.emplace_back();
            hostVolumeFlameBuffers.emplace_back();
            const int volume_idx = static_cast<int>(volumeRegionsList.size());

            if (vol.has_density_grid()) {
                std::vector<float> density_values;
                std::string density_err;
                const std::string density_path = resolve_scene_path(vol.density_file);
                if (!loadRawScalarField(density_path, vol.density_nx, vol.density_ny, vol.density_nz,
                                        vol.density_format, density_values, &density_err)) {
                    std::cerr << "Failed to load raw volume density: " << density_err << "\n";
                    return 1;
                }

                const float max_density = density_values.empty()
                    ? 0.0f
                    : *std::max_element(density_values.begin(), density_values.end());
                const float sigma_t_max = fmaxf(vol.medium.sigma_t.x,
                    fmaxf(vol.medium.sigma_t.y, vol.medium.sigma_t.z));

                hostVolumeDensityBuffers[volume_idx] = std::move(density_values);
                gpuVol.density_data = hostVolumeDensityBuffers[volume_idx].data();
                gpuVol.density_nx = vol.density_nx;
                gpuVol.density_ny = vol.density_ny;
                gpuVol.density_nz = vol.density_nz;
                if (gpuVol.density_majorant <= 0.0f) {
                    gpuVol.density_majorant = max_density * gpuVol.density_scale * sigma_t_max;
                }

                printf("Loaded raw density grid: %s (%dx%dx%d, majorant %.4f)\n",
                       density_path.c_str(),
                       gpuVol.density_nx, gpuVol.density_ny, gpuVol.density_nz,
                       gpuVol.density_majorant);
                if (gpuVol.emission_scale > 0.0f) {
                    printf("  Volume emission: scale=%.2f, temp=[%.0f, %.0f] K\n",
                           gpuVol.emission_scale, gpuVol.emission_temp_min, gpuVol.emission_temp_max);
                }
            }

            if (vol.has_temperature_grid()) {
                std::string temperature_err;
                const std::string temperature_path = resolve_scene_path(vol.temperature_file);
                if (!loadRawScalarField(temperature_path,
                                        vol.temperature_nx, vol.temperature_ny, vol.temperature_nz,
                                        vol.temperature_format,
                                        hostVolumeTemperatureBuffers[volume_idx],
                                        &temperature_err)) {
                    std::cerr << "Failed to load raw volume temperature: " << temperature_err << "\n";
                    return 1;
                }

                gpuVol.temperature_data = hostVolumeTemperatureBuffers[volume_idx].data();
                gpuVol.temperature_nx = vol.temperature_nx;
                gpuVol.temperature_ny = vol.temperature_ny;
                gpuVol.temperature_nz = vol.temperature_nz;

                printf("Loaded raw temperature grid: %s (%dx%dx%d, scale %.4f)\n",
                       temperature_path.c_str(),
                       gpuVol.temperature_nx, gpuVol.temperature_ny, gpuVol.temperature_nz,
                       gpuVol.temperature_scale);
            }

            if (vol.has_flame_grid()) {
                std::string flame_err;
                const std::string flame_path = resolve_scene_path(vol.flame_file);
                if (!loadRawScalarField(flame_path,
                                        vol.flame_nx, vol.flame_ny, vol.flame_nz,
                                        vol.flame_format,
                                        hostVolumeFlameBuffers[volume_idx],
                                        &flame_err)) {
                    std::cerr << "Failed to load raw volume flames: " << flame_err << "\n";
                    return 1;
                }

                gpuVol.flame_data = hostVolumeFlameBuffers[volume_idx].data();
                gpuVol.flame_nx = vol.flame_nx;
                gpuVol.flame_ny = vol.flame_ny;
                gpuVol.flame_nz = vol.flame_nz;

                printf("Loaded raw flame grid: %s (%dx%dx%d, scale %.4f)\n",
                       flame_path.c_str(),
                       gpuVol.flame_nx, gpuVol.flame_ny, gpuVol.flame_nz,
                       gpuVol.flame_scale);
            }

            volumeRegionsList.push_back(gpuVol);
        }
    }
    const int numVolumeRegions = static_cast<int>(volumeRegionsList.size());
    if (numVolumeRegions > 0) {
        printf("Volume regions loaded: %d\n", numVolumeRegions);
    }

    AABB sceneAABB;

    size_t P = globalMesh.indices.size() / 3;
    size_t bvh_chunk_size = required<RayTracer::BVHState>(P);
    char* bvh_chunk = nullptr;
#ifdef __CUDACC__
    cudaError_t alloc_err = cudaMalloc(&bvh_chunk, bvh_chunk_size);
    if (alloc_err != cudaSuccess) {
        std::fprintf(stderr, "Failed to allocate device memory for BVH: %s\n", cudaGetErrorString(alloc_err));
        return 1;
    }
#else
    bvh_chunk = new char[bvh_chunk_size];
#endif
    RayTracer::BVHState bvhState = RayTracer::BVHState::fromChunk(bvh_chunk, P);

    AccStruct::BVH bvh;

#ifdef __CUDACC__
    Vec3* d_positions = nullptr;
    Vec3* d_normals = nullptr;
    Vec2* d_uvs = nullptr;
    uint32_t* d_indices = nullptr;
    int32_t* d_triangle_obj_ids = nullptr;
    Material* d_object_materials = nullptr;

    const size_t bytesPos    = globalMesh.positions.size() * sizeof(Vec3);
    const size_t bytesIdx    = globalMesh.indices.size() * sizeof(uint32_t);
    const size_t bytesNrm    = globalMesh.normals.size() * sizeof(Vec3);
    const size_t bytesUV     = globalMesh.uvs.size() * sizeof(Vec2);
    const size_t bytesTriObj = globalMesh.triangleObjIds.size() * sizeof(int32_t);
    const size_t bytesObjMat = objectMaterials.size() * sizeof(Material);

    CHECK_CUDA((cudaMalloc(&d_positions, bytesPos)), true);
    CHECK_CUDA((cudaMalloc(&d_indices, bytesIdx)), true);
    CHECK_CUDA((cudaMalloc(&d_triangle_obj_ids, bytesTriObj)), true);
    CHECK_CUDA((cudaMalloc(&d_object_materials, bytesObjMat)), true);
    if (!globalMesh.normals.empty()) {
        CHECK_CUDA((cudaMalloc(&d_normals, bytesNrm)), true);
    }
    if (!globalMesh.uvs.empty()) {
        CHECK_CUDA((cudaMalloc(&d_uvs, bytesUV)), true);
    }

    CHECK_CUDA((cudaMemcpy(d_positions, globalMesh.positions.data(), bytesPos, cudaMemcpyHostToDevice)), true);
    CHECK_CUDA((cudaMemcpy(d_indices, globalMesh.indices.data(), bytesIdx, cudaMemcpyHostToDevice)), true);
    CHECK_CUDA((cudaMemcpy(d_triangle_obj_ids, globalMesh.triangleObjIds.data(), bytesTriObj, cudaMemcpyHostToDevice)), true);
    CHECK_CUDA((cudaMemcpy(d_object_materials, objectMaterials.data(), bytesObjMat, cudaMemcpyHostToDevice)), true);
    if (!globalMesh.normals.empty()) {
        CHECK_CUDA((cudaMemcpy(d_normals, globalMesh.normals.data(), bytesNrm, cudaMemcpyHostToDevice)), true);
    }
    if (!globalMesh.uvs.empty()) {
        CHECK_CUDA((cudaMemcpy(d_uvs, globalMesh.uvs.data(), bytesUV, cudaMemcpyHostToDevice)), true);
    }

    MeshView d_mesh{};
    d_mesh.positions = d_positions;
    d_mesh.normals = d_normals;
    d_mesh.uvs = d_uvs;
    d_mesh.indices = d_indices;
    d_mesh.triangleObjIds = d_triangle_obj_ids;
    d_mesh.numVertices = globalMesh.positions.size();
    d_mesh.numIndices = globalMesh.indices.size();
    d_mesh.numTriangles = P;

    CHECK_CUDA(bvh.calculateAABBs(d_mesh, bvhState.AABBs), true);

    // ---- Upload per-object media ----
    HomogeneousMedium* d_objectMedia = nullptr;
    if (numObjectMedia > 0) {
        CHECK_CUDA((cudaMalloc(&d_objectMedia, sizeof(HomogeneousMedium) * numObjectMedia)), true);
        CHECK_CUDA((cudaMemcpy(d_objectMedia, objectMediaList.data(),
                               sizeof(HomogeneousMedium) * numObjectMedia, cudaMemcpyHostToDevice)), true);
    }

    // ---- Upload volume regions to GPU ----
    VolumeRegionGPU* d_volumeRegions = nullptr;
    std::vector<float*> d_volumeDensityPtrs;
    std::vector<float*> d_volumeTemperaturePtrs;
    std::vector<float*> d_volumeFlamePtrs;
    if (numVolumeRegions > 0) {
        std::vector<VolumeRegionGPU> deviceVolumeRegions = volumeRegionsList;
        d_volumeDensityPtrs.resize(numVolumeRegions, nullptr);
        d_volumeTemperaturePtrs.resize(numVolumeRegions, nullptr);
        d_volumeFlamePtrs.resize(numVolumeRegions, nullptr);
        for (int i = 0; i < numVolumeRegions; ++i) {
            if (!volumeRegionsList[i].has_density_grid()) continue;
            const size_t voxel_count =
                static_cast<size_t>(volumeRegionsList[i].density_nx) *
                static_cast<size_t>(volumeRegionsList[i].density_ny) *
                static_cast<size_t>(volumeRegionsList[i].density_nz);
            const size_t density_bytes = voxel_count * sizeof(float);
            CHECK_CUDA((cudaMalloc(&d_volumeDensityPtrs[i], density_bytes)), true);
            CHECK_CUDA((cudaMemcpy(d_volumeDensityPtrs[i], hostVolumeDensityBuffers[i].data(),
                                   density_bytes, cudaMemcpyHostToDevice)), true);
            deviceVolumeRegions[i].density_data = d_volumeDensityPtrs[i];
        }
        for (int i = 0; i < numVolumeRegions; ++i) {
            if (!volumeRegionsList[i].has_temperature_grid()) continue;
            const size_t voxel_count =
                static_cast<size_t>(volumeRegionsList[i].temperature_nx) *
                static_cast<size_t>(volumeRegionsList[i].temperature_ny) *
                static_cast<size_t>(volumeRegionsList[i].temperature_nz);
            const size_t grid_bytes = voxel_count * sizeof(float);
            CHECK_CUDA((cudaMalloc(&d_volumeTemperaturePtrs[i], grid_bytes)), true);
            CHECK_CUDA((cudaMemcpy(d_volumeTemperaturePtrs[i], hostVolumeTemperatureBuffers[i].data(),
                                   grid_bytes, cudaMemcpyHostToDevice)), true);
            deviceVolumeRegions[i].temperature_data = d_volumeTemperaturePtrs[i];
        }
        for (int i = 0; i < numVolumeRegions; ++i) {
            if (!volumeRegionsList[i].has_flame_grid()) continue;
            const size_t voxel_count =
                static_cast<size_t>(volumeRegionsList[i].flame_nx) *
                static_cast<size_t>(volumeRegionsList[i].flame_ny) *
                static_cast<size_t>(volumeRegionsList[i].flame_nz);
            const size_t grid_bytes = voxel_count * sizeof(float);
            CHECK_CUDA((cudaMalloc(&d_volumeFlamePtrs[i], grid_bytes)), true);
            CHECK_CUDA((cudaMemcpy(d_volumeFlamePtrs[i], hostVolumeFlameBuffers[i].data(),
                                   grid_bytes, cudaMemcpyHostToDevice)), true);
            deviceVolumeRegions[i].flame_data = d_volumeFlamePtrs[i];
        }
        CHECK_CUDA((cudaMalloc(&d_volumeRegions, sizeof(VolumeRegionGPU) * numVolumeRegions)), true);
        CHECK_CUDA((cudaMemcpy(d_volumeRegions, deviceVolumeRegions.data(),
                               sizeof(VolumeRegionGPU) * numVolumeRegions, cudaMemcpyHostToDevice)), true);
    }

    // ---- Upload texture data to GPU ----
    // For GPU: we need to upload each texture's pixel data separately,
    // then create device-side TextureData structs pointing to device memory.
    TextureData* d_textures = nullptr;
    std::vector<unsigned char*> d_texPixelPtrs; // track for cleanup
    if (numTextures > 0) {
        // Upload pixel data for each texture
        std::vector<TextureData> deviceTexData(numTextures);
        d_texPixelPtrs.resize(numTextures, nullptr);

        for (int ti = 0; ti < numTextures; ++ti) {
            const TextureData& hTex = allTextureData[ti];
            if (hTex.data == nullptr || hTex.width <= 0 || hTex.height <= 0) continue;

            size_t pixelBytes = (size_t)hTex.width * hTex.height * hTex.channels;
            unsigned char* d_pixels = nullptr;
            CHECK_CUDA((cudaMalloc(&d_pixels, pixelBytes)), true);
            CHECK_CUDA((cudaMemcpy(d_pixels, hTex.data, pixelBytes, cudaMemcpyHostToDevice)), true);

            d_texPixelPtrs[ti] = d_pixels;
            deviceTexData[ti].width    = hTex.width;
            deviceTexData[ti].height   = hTex.height;
            deviceTexData[ti].channels = hTex.channels;
            deviceTexData[ti].data     = d_pixels;
        }

        CHECK_CUDA((cudaMalloc(&d_textures, sizeof(TextureData) * numTextures)), true);
        CHECK_CUDA((cudaMemcpy(d_textures, deviceTexData.data(),
                               sizeof(TextureData) * numTextures, cudaMemcpyHostToDevice)), true);
    }

    // ---- Load & upload HDR sky environment map ----
    HDRTextureData  h_hdri{};          // host-side view (data = host ptr)
    HDRTextureData* d_hdri = nullptr;  // device pointer to HDRTextureData struct
    float*          d_hdri_pixels = nullptr;
    HDRTexture*     hdri_owner = nullptr;

    if (!scene.sky_hdri_path.empty()) {
        hdri_owner = LoadHDRTexture(scene.sky_hdri_path);
        if (hdri_owner) {
            size_t floatBytes = (size_t)hdri_owner->width * hdri_owner->height * 3 * sizeof(float);
            CHECK_CUDA((cudaMalloc(&d_hdri_pixels, floatBytes)), true);
            CHECK_CUDA((cudaMemcpy(d_hdri_pixels, hdri_owner->data.data(),
                                   floatBytes, cudaMemcpyHostToDevice)), true);

            // Build a device-side HDRTextureData struct pointing to device pixels
            HDRTextureData dev_hdri;
            dev_hdri.width     = hdri_owner->width;
            dev_hdri.height    = hdri_owner->height;
            dev_hdri.data      = d_hdri_pixels;
            dev_hdri.tint      = scene.environment.tint;
            dev_hdri.intensity = scene.environment.map_intensity;
            dev_hdri.az_rot    = scene.environment.map_rotation_deg / 360.0f;

            CHECK_CUDA((cudaMalloc(&d_hdri, sizeof(HDRTextureData))), true);
            CHECK_CUDA((cudaMemcpy(d_hdri, &dev_hdri,
                                   sizeof(HDRTextureData), cudaMemcpyHostToDevice)), true);
        }
    }

#else
    MeshView h_mesh = globalMesh.getView();
    bvh.calculateAABBs(h_mesh, bvhState.AABBs);

    HDRTexture*    h_hdri_owner = nullptr;
    HDRTextureData h_hdri_data{};
    if (!scene.sky_hdri_path.empty()) {
        h_hdri_owner = LoadHDRTexture(scene.sky_hdri_path);
        if (h_hdri_owner) {
            h_hdri_data           = h_hdri_owner->sampled;
            h_hdri_data.tint      = scene.environment.tint;
            h_hdri_data.intensity = scene.environment.map_intensity;
            h_hdri_data.az_rot    = scene.environment.map_rotation_deg / 360.0f;
        }
    }

    // Wire the GPU path az_rot too (if GPU build sets dev_hdri, update it here)
    // For the GPU path this variable is unused; the GPU gets d_hdri (already uploaded).

    // Build 2D importance distribution from float HDR for map-based sampling.
    // ParseTopLevelLights (the polymorphic ILight path) is called here to wire
    // EnvironmentLight/LightFactory/LightJsonParser into the runtime pipeline.
    EnvImportanceMaps env_imp_maps;
    {
        // Wire polymorphic light path: parse top-level lights[] to get ILight objects.
        // Their parameters (enabled, tint, intensity, rotation) were already applied to
        // scene.environment during scene load; the ILight objects are used here for
        // validation and to confirm the factory/registration path is exercised at runtime.
        std::vector<std::unique_ptr<ILight>> ilights;
        if (!scene_json_path.empty()) {
            std::ifstream jf(scene_json_path);
            if (jf) {
                std::string json_text((std::istreambuf_iterator<char>(jf)),
                                       std::istreambuf_iterator<char>());
                SceneIO::JsonValue root;
                std::string perr;
                if (SceneIO::parse_json(json_text, root, &perr)) {
                    LightParseSummary summary = ParseTopLevelLights(root, ilights);
                    if (summary.loaded > 0 || summary.failed > 0) {
                        printf("[Lighting]   ILight path: %d loaded, %d failed, %d skipped\n",
                               summary.loaded, summary.failed, summary.skipped);
                    }
                }
            }
        }

        // Build CDF from float HDR (uses sampleHDRI UV convention for consistency).
        if (h_hdri_data.width > 0) {
            const float az = scene.environment.map_rotation_deg / 360.0f;
            env_imp_maps = BuildEnvImportance(h_hdri_data, az);
            printf("[Lighting]   env importance map: %dx%d pixels, az_rot=%.4f\n",
                   env_imp_maps.view.width, env_imp_maps.view.height, az);
        }
    }
    const EnvImportanceData* h_env_imp =
        (env_imp_maps.view.valid ? &env_imp_maps.view : nullptr);
#endif

    AABB SceneBoundingBox;
#ifdef __CUDACC__
    AABB default_aabb;
    SceneBoundingBox = thrust::reduce(
        thrust::device_pointer_cast(bvhState.AABBs + (P - 1)),
        thrust::device_pointer_cast(bvhState.AABBs + (2*P - 1)),
        default_aabb,
        [] __device__ __host__ (const AABB& lhs, const AABB& rhs) {
            return AABB::merge(lhs, rhs);
        });

    thrust::device_vector<unsigned int> TriangleIndices(P);
    thrust::copy(thrust::make_counting_iterator<std::uint32_t>(0),
        thrust::make_counting_iterator<std::uint32_t>(P),
        TriangleIndices.begin());

    warmupGPU();
    auto start_gpu = std::chrono::high_resolution_clock::now();
    bvh.buildBVH(bvhState.Nodes, bvhState.AABBs, SceneBoundingBox,
                 &TriangleIndices, static_cast<int>(P));
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_gpu = end_gpu - start_gpu;
    printf("GPU LBVH Build Time: %.3f ms\n", ms_gpu.count());
#else
    SceneBoundingBox = std::accumulate(
        bvhState.AABBs + (P - 1),
        bvhState.AABBs + (2 * P - 1),
        AABB(),
        [](const AABB& lhs, const AABB& rhs) { return AABB::merge(lhs, rhs); });

    std::vector<unsigned int> TriangleIndices(P);
    std::iota(TriangleIndices.begin(), TriangleIndices.end(), 0);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    bvh.buildBVH(bvhState.Nodes, bvhState.AABBs, SceneBoundingBox,
                 TriangleIndices, static_cast<int>(P));
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_cpu = end_cpu - start_cpu;
    printf("CPU LBVH Build Time: %.3f ms\n", ms_cpu.count());
#endif

    // --- Camera and Render Settings ---
    int max_depth = has_scene ? scene.settings.max_depth : 1;
    int spp = has_scene ? scene.settings.spp : 1;
    bool diffuse_bounce = has_scene ? scene.settings.diffuse_bounce : true;
    const int keyframes = has_scene ? scene.settings.keyframes : 1;

    Vec3 miss_color = has_scene ? scene.miss_color : make_vec3(0.0f, 0.0f, 0.0f);
    Camera cam = has_scene ? scene.camera : Camera();
    std::vector<Light> render_lights = scene.lights;
    {
        std::vector<Light> supported_lights;
        for (const auto& l : render_lights) {
            if (l.type == 0 || l.type == 2) supported_lights.push_back(l);
        }
        render_lights = std::move(supported_lights);
    }
    const int num_lights = static_cast<int>(render_lights.size());

    // ---- Runtime lighting summary ----
    {
        int n_dir = 0, n_point = 0;
        for (const auto& l : render_lights) {
            if (l.type == 2) ++n_dir;
            else if (l.type == 0) ++n_point;
        }
        printf("[Lighting] %d active lights  (%d directional, %d point)\n",
               num_lights, n_dir, n_point);
        for (const auto& l : render_lights) {
            if (l.type == 2) {
                printf("[Lighting]   sun  dir=(%.3f,%.3f,%.3f)  tint=(%.2f,%.2f,%.2f)  intensity=%.3f\n",
                       l.direction.x, l.direction.y, l.direction.z,
                       l.color.x, l.color.y, l.color.z, l.intensity);
            }
        }
        if (!scene.sky_hdri_path.empty()) {
            printf("[Lighting]   hdri path=%s\n", scene.sky_hdri_path.c_str());
            printf("[Lighting]   hdri tint=(%.2f,%.2f,%.2f)  intensity=%.3f\n",
                   scene.environment.tint.x, scene.environment.tint.y, scene.environment.tint.z,
                   scene.environment.map_intensity);
        } else {
            printf("[Lighting]   no HDRI — using miss color (%.2f,%.2f,%.2f)\n",
                   scene.miss_color.x, scene.miss_color.y, scene.miss_color.z);
        }
    }

    // ---- Emissive triangle list (rebuildable) ----
    std::vector<EmissiveTriInfo> h_emissiveTris;
    std::vector<float> h_emissiveCDF;
    float totalEmissiveArea = 0.0f;
    int numEmissiveTris = 0;

    auto buildEmissiveTris = [&]() {
        h_emissiveTris.clear();
        h_emissiveCDF.clear();
        totalEmissiveArea = 0.0f;
        const int nm = static_cast<int>(objectMaterials.size());
        for (size_t ti = 0; ti < P; ++ti) {
            const int objId = globalMesh.triangleObjIds[ti];
            if (objId < 0 || objId >= nm) continue;
            const Vec3& em = objectMaterials[objId].emission;
            if (em.x <= 0.0f && em.y <= 0.0f && em.z <= 0.0f) continue;

            const uint32_t i0 = globalMesh.indices[ti * 3 + 0];
            const uint32_t i1 = globalMesh.indices[ti * 3 + 1];
            const uint32_t i2 = globalMesh.indices[ti * 3 + 2];
            const Vec3& v0 = globalMesh.positions[i0];
            const Vec3& v1 = globalMesh.positions[i1];
            const Vec3& v2 = globalMesh.positions[i2];

            Vec3 e1 = v1 - v0;
            Vec3 e2 = v2 - v0;
            Vec3 cr = cross(e1, e2);
            float triArea = 0.5f * sqrtf(dot(cr, cr));
            if (triArea < 1e-12f) continue;
            float crLen = sqrtf(dot(cr, cr));
            Vec3 faceN = cr * (1.0f / crLen);
            EmissiveTriInfo info;
            info.triangleIdx = static_cast<int>(ti);
            info.emission = em;
            info.area = triArea;
            info.normal = faceN;
            h_emissiveTris.push_back(info);
            totalEmissiveArea += triArea;
        }
        h_emissiveCDF.resize(h_emissiveTris.size());
        float cumulative = 0.0f;
        for (size_t i = 0; i < h_emissiveTris.size(); ++i) {
            cumulative += h_emissiveTris[i].area;
            h_emissiveCDF[i] = cumulative / totalEmissiveArea;
        }
        if (!h_emissiveCDF.empty()) h_emissiveCDF.back() = 1.0f;
        numEmissiveTris = static_cast<int>(h_emissiveTris.size());
        printf("Emissive triangles: %zu  (total area: %.4f)\n",
               h_emissiveTris.size(), totalEmissiveArea);
    };

    buildEmissiveTris();

    const int img_w = cam.pixel_width;
    const int img_h = cam.pixel_height;
    const int num_pixels = img_w * img_h;
    std::vector<Vec3> image(num_pixels, make_vec3(0.0f, 0.0f, 0.0f));

    // ---- Animated geometry ----
    std::vector<int> cur_path_indices(load_objects.size(), 0);

    auto reloadAnimatedGeometry = [&]() {
        globalMesh = Mesh{};
        objectMaterials.clear();
        objectMediaList.clear();
        nextObjectId = 0;
        for (const auto& obj : load_objects) {
            if (obj.is_volume) {
                if (objectMediaList.size() < static_cast<size_t>(nextObjectId + 1))
                    objectMediaList.resize(nextObjectId + 1, HomogeneousMedium());
                objectMediaList[nextObjectId] = obj.medium;
                if (objectMaterials.size() < static_cast<size_t>(nextObjectId + 1))
                    objectMaterials.resize(nextObjectId + 1, Material());
                objectMaterials[nextObjectId] = obj.material;
                nextObjectId++;
                continue;
            }
            if (obj.path.empty()) continue;
            std::printf("Loading OBJ: %s\n", obj.path.c_str());
            Mesh tempMesh;
            const int objIdBegin = nextObjectId;
            std::vector<ParsedMTLMaterial> mtlMats;
            std::string mtlLibName;
            if (!LoadOBJ_ToMesh(obj.path, tempMesh, nextObjectId,
                                obj.use_mtl ? &mtlMats : nullptr, &mtlLibName)) {
                std::cerr << "Failed to load OBJ: " << obj.path << "\n";
                continue;
            }
            applyObjectTransform(tempMesh, obj);
            if (objectMaterials.size() < static_cast<size_t>(nextObjectId))
                objectMaterials.resize(nextObjectId, Material());
            if (objectMediaList.size() < static_cast<size_t>(nextObjectId))
                objectMediaList.resize(nextObjectId, HomogeneousMedium());
            for (int oid = objIdBegin; oid < nextObjectId; ++oid) {
                Material mat = obj.material;
                int localIdx = oid - objIdBegin;
                if (obj.use_mtl && localIdx < (int)mtlMats.size() &&
                    !mtlMats[localIdx].name.empty()) {
                    const ParsedMTLMaterial& mtl = mtlMats[localIdx];
                    mat.albedo        = mtl.Kd;
                    mat.specularColor = mtl.Ks;
                    mat.emission      = mtl.Ke;
                    mat.shininess     = mtl.Ns;
                    float ksMax = fmaxf(mtl.Ks.x, fmaxf(mtl.Ks.y, mtl.Ks.z));
                    if (ksMax > 1e-4f && mat.ks <= 0.0f) mat.ks = 0.5f;
                    if (!mtl.map_Kd.empty())   mat.diffuseTexIdx = registerTexture(mtl.map_Kd);
                    if (!mtl.map_d.empty())    mat.alphaTexIdx   = registerTexture(mtl.map_d);
                    if (!mtl.map_Bump.empty()) mat.normalTexIdx  = registerTexture(mtl.map_Bump);
                    if (obj.material.emission.x > 0 || obj.material.emission.y > 0 ||
                        obj.material.emission.z > 0)
                        mat.emission = obj.material.emission;
                }
                if (!obj.diffuse_tex_path.empty()) mat.diffuseTexIdx = registerTexture(obj.diffuse_tex_path);
                if (!obj.normal_tex_path.empty())  mat.normalTexIdx  = registerTexture(obj.normal_tex_path);
                if (!obj.alpha_tex_path.empty())   mat.alphaTexIdx   = registerTexture(obj.alpha_tex_path);
                objectMaterials[oid] = mat;
                objectMediaList[oid] = obj.medium;
            }
            AppendMesh(globalMesh, tempMesh);
        }
        P = globalMesh.indices.size() / 3;
    };

#ifdef __CUDACC__
    Triangle* d_tris = nullptr;
    Vec3* d_image = nullptr;
    Light* d_lights = nullptr;

    CHECK_CUDA((cudaMalloc(&d_tris, sizeof(Triangle) * P)), true);
    CHECK_CUDA((cudaMalloc(&d_image, sizeof(Vec3) * img_w * img_h)), true);
    CHECK_CUDA((cudaMalloc(&d_lights, sizeof(Light) * num_lights)), true);
    CHECK_CUDA((cudaMemcpy(d_lights, render_lights.data(), sizeof(Light) * num_lights, cudaMemcpyHostToDevice)), true);

    EmissiveTriInfo* d_emissiveTris = nullptr;
    float* d_emissiveCDF = nullptr;
    if (numEmissiveTris > 0) {
        CHECK_CUDA((cudaMalloc(&d_emissiveTris, sizeof(EmissiveTriInfo) * numEmissiveTris)), true);
        CHECK_CUDA((cudaMemcpy(d_emissiveTris, h_emissiveTris.data(), sizeof(EmissiveTriInfo) * numEmissiveTris, cudaMemcpyHostToDevice)), true);
        CHECK_CUDA((cudaMalloc(&d_emissiveCDF, sizeof(float) * numEmissiveTris)), true);
        CHECK_CUDA((cudaMemcpy(d_emissiveCDF, h_emissiveCDF.data(), sizeof(float) * numEmissiveTris, cudaMemcpyHostToDevice)), true);
    }

    Vec3* d_albedo_aov = nullptr;
    Vec3* d_normal_aov = nullptr;
    if (use_denoiser) {
        CHECK_CUDA((cudaMalloc(&d_albedo_aov, sizeof(Vec3) * img_w * img_h)), true);
        CHECK_CUDA((cudaMalloc(&d_normal_aov, sizeof(Vec3) * img_w * img_h)), true);
        CHECK_CUDA((cudaMemset(d_albedo_aov, 0, sizeof(Vec3) * img_w * img_h)), true);
        CHECK_CUDA((cudaMemset(d_normal_aov, 0, sizeof(Vec3) * img_w * img_h)), true);
    }

    {
        const int tri_blocks = (static_cast<int>(P) + 256 - 1) / 256;
        buildTrianglesKernel<<<tri_blocks, 256>>>(d_mesh, d_tris, static_cast<int>(P));
        CHECK_CUDA((cudaDeviceSynchronize()), true);
    }

    render(P, 1, 1, cam, miss_color, max_depth, 1, bvhState.Nodes, bvhState.AABBs, d_tris,
           d_triangle_obj_ids, d_object_materials, static_cast<int>(objectMaterials.size()),
           d_lights, num_lights, diffuse_bounce,
           d_emissiveTris, d_emissiveCDF, numEmissiveTris, totalEmissiveArea,
           d_image, nullptr, nullptr, nee_mode,
           d_objectMedia, numObjectMedia,
           d_textures, numTextures,
           d_volumeRegions, numVolumeRegions,
           d_hdri);

    auto rebuildGPUGeometry = [&]() {
        cudaFree(d_positions);
        if (d_normals) cudaFree(d_normals);
        if (d_uvs) cudaFree(d_uvs);
        cudaFree(d_indices);
        cudaFree(d_triangle_obj_ids);
        cudaFree(d_object_materials);
        if (d_objectMedia) cudaFree(d_objectMedia);
        cudaFree(bvh_chunk);

        d_positions = nullptr; d_normals = nullptr; d_uvs = nullptr;
        d_objectMedia = nullptr;

        const size_t newBvhSize = required<RayTracer::BVHState>(P);
        CHECK_CUDA((cudaMalloc(&bvh_chunk, newBvhSize)), true);
        bvhState = RayTracer::BVHState::fromChunk(bvh_chunk, P);

        CHECK_CUDA((cudaMalloc(&d_positions, globalMesh.positions.size() * sizeof(Vec3))), true);
        CHECK_CUDA((cudaMalloc(&d_indices, globalMesh.indices.size() * sizeof(uint32_t))), true);
        CHECK_CUDA((cudaMalloc(&d_triangle_obj_ids, globalMesh.triangleObjIds.size() * sizeof(int32_t))), true);
        CHECK_CUDA((cudaMalloc(&d_object_materials, objectMaterials.size() * sizeof(Material))), true);
        if (!globalMesh.normals.empty())
            CHECK_CUDA((cudaMalloc(&d_normals, globalMesh.normals.size() * sizeof(Vec3))), true);
        if (!globalMesh.uvs.empty())
            CHECK_CUDA((cudaMalloc(&d_uvs, globalMesh.uvs.size() * sizeof(Vec2))), true);

        CHECK_CUDA((cudaMemcpy(d_positions, globalMesh.positions.data(), globalMesh.positions.size() * sizeof(Vec3), cudaMemcpyHostToDevice)), true);
        CHECK_CUDA((cudaMemcpy(d_indices, globalMesh.indices.data(), globalMesh.indices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice)), true);
        CHECK_CUDA((cudaMemcpy(d_triangle_obj_ids, globalMesh.triangleObjIds.data(), globalMesh.triangleObjIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice)), true);
        CHECK_CUDA((cudaMemcpy(d_object_materials, objectMaterials.data(), objectMaterials.size() * sizeof(Material), cudaMemcpyHostToDevice)), true);
        if (!globalMesh.normals.empty())
            CHECK_CUDA((cudaMemcpy(d_normals, globalMesh.normals.data(), globalMesh.normals.size() * sizeof(Vec3), cudaMemcpyHostToDevice)), true);
        if (!globalMesh.uvs.empty())
            CHECK_CUDA((cudaMemcpy(d_uvs, globalMesh.uvs.data(), globalMesh.uvs.size() * sizeof(Vec2), cudaMemcpyHostToDevice)), true);

        d_mesh.positions       = d_positions;
        d_mesh.normals         = d_normals;
        d_mesh.uvs             = d_uvs;
        d_mesh.indices         = d_indices;
        d_mesh.triangleObjIds  = d_triangle_obj_ids;
        d_mesh.numVertices     = globalMesh.positions.size();
        d_mesh.numIndices      = globalMesh.indices.size();
        d_mesh.numTriangles    = P;

        const int newNumObjMedia = static_cast<int>(objectMediaList.size());
        if (newNumObjMedia > 0) {
            CHECK_CUDA((cudaMalloc(&d_objectMedia, sizeof(HomogeneousMedium) * newNumObjMedia)), true);
            CHECK_CUDA((cudaMemcpy(d_objectMedia, objectMediaList.data(), sizeof(HomogeneousMedium) * newNumObjMedia, cudaMemcpyHostToDevice)), true);
        }

        CHECK_CUDA(bvh.calculateAABBs(d_mesh, bvhState.AABBs), true);

        AABB newSceneBB;
        newSceneBB = thrust::reduce(
            thrust::device_pointer_cast(bvhState.AABBs + (P - 1)),
            thrust::device_pointer_cast(bvhState.AABBs + (2*P - 1)),
            AABB(),
            [] __device__ __host__ (const AABB& a, const AABB& b) { return AABB::merge(a, b); });

        thrust::device_vector<unsigned int> newTriIdx(P);
        thrust::copy(thrust::make_counting_iterator<std::uint32_t>(0),
            thrust::make_counting_iterator<std::uint32_t>(P), newTriIdx.begin());
        bvh.buildBVH(bvhState.Nodes, bvhState.AABBs, newSceneBB, &newTriIdx, static_cast<int>(P));
        cudaDeviceSynchronize();

        cudaFree(d_tris);
        d_tris = nullptr;
        CHECK_CUDA((cudaMalloc(&d_tris, sizeof(Triangle) * P)), true);
        const int newBlocks = (static_cast<int>(P) + 256 - 1) / 256;
        buildTrianglesKernel<<<newBlocks, 256>>>(d_mesh, d_tris, static_cast<int>(P));
        CHECK_CUDA((cudaDeviceSynchronize()), true);

        buildEmissiveTris();
        if (d_emissiveTris) { cudaFree(d_emissiveTris); d_emissiveTris = nullptr; }
        if (d_emissiveCDF)  { cudaFree(d_emissiveCDF);  d_emissiveCDF  = nullptr; }
        if (numEmissiveTris > 0) {
            CHECK_CUDA((cudaMalloc(&d_emissiveTris, sizeof(EmissiveTriInfo) * numEmissiveTris)), true);
            CHECK_CUDA((cudaMemcpy(d_emissiveTris, h_emissiveTris.data(), sizeof(EmissiveTriInfo) * numEmissiveTris, cudaMemcpyHostToDevice)), true);
            CHECK_CUDA((cudaMalloc(&d_emissiveCDF, sizeof(float) * numEmissiveTris)), true);
            CHECK_CUDA((cudaMemcpy(d_emissiveCDF, h_emissiveCDF.data(), sizeof(float) * numEmissiveTris, cudaMemcpyHostToDevice)), true);
        }
    };
#endif

    // ---- Frame loop ----
    auto reinhard = [](float c) -> unsigned char {
        float mapped = c / (1.0f + c);
        mapped = powf(fmaxf(mapped, 0.0f), 1.0f / 2.2f);
        return static_cast<unsigned char>(255.0f * fminf(mapped, 1.0f));
    };

    for (int frame = 0; frame < keyframes; ++frame) {
        if (keyframes > 1)
            printf("Rendering frame %d / %d\n", frame + 1, keyframes);

        Camera frame_cam = cam;
        if (!scene.camera_positions.empty()) {
            Vec3 pos = sample_keyframe_vec3(scene.camera_positions, frame, keyframes);
            Vec3 lat = scene.camera_look_ats.empty()
                ? scene.camera.get_look_at()
                : sample_keyframe_vec3(scene.camera_look_ats, frame, keyframes);
            frame_cam = Camera(pos, lat,
                scene.camera.get_up_vector(),
                scene.camera.get_focal_length_mm(),
                scene.camera.get_sensor_height_mm(),
                cam.pixel_width, cam.pixel_height);
        }

        bool geometry_changed = false;
        for (int oi = 0; oi < (int)load_objects.size(); ++oi) {
            if (load_objects[oi].paths.empty()) continue;
            int new_idx = animated_path_index(frame, keyframes, (int)load_objects[oi].paths.size());
            if (new_idx != cur_path_indices[oi]) {
                cur_path_indices[oi] = new_idx;
                load_objects[oi].path = resolve_scene_path(load_objects[oi].paths[new_idx]);
                geometry_changed = true;
            }
        }
        if (geometry_changed) {
            reloadAnimatedGeometry();
#ifdef __CUDACC__
            rebuildGPUGeometry();
#else
            delete[] bvh_chunk;
            const size_t newBvhSize = required<RayTracer::BVHState>(P);
            bvh_chunk = new char[newBvhSize];
            bvhState = RayTracer::BVHState::fromChunk(bvh_chunk, P);
            MeshView h_mesh_rebuild = globalMesh.getView();
            bvh.calculateAABBs(h_mesh_rebuild, bvhState.AABBs);
            AABB newSceneBB = std::accumulate(
                bvhState.AABBs + (P - 1),
                bvhState.AABBs + (2 * P - 1),
                AABB(),
                [](const AABB& a, const AABB& b) { return AABB::merge(a, b); });
            std::vector<unsigned int> newTriIdx(P);
            std::iota(newTriIdx.begin(), newTriIdx.end(), 0);
            bvh.buildBVH(bvhState.Nodes, bvhState.AABBs, newSceneBB, newTriIdx, static_cast<int>(P));
            buildEmissiveTris();
#endif
        }

        const int num_om    = static_cast<int>(objectMaterials.size());
        const int num_media = static_cast<int>(objectMediaList.size());

#ifdef __CUDACC__
        CHECK_CUDA((cudaMemset(d_image, 0, sizeof(Vec3) * img_w * img_h)), true);
        if (use_denoiser) {
            CHECK_CUDA((cudaMemset(d_albedo_aov, 0, sizeof(Vec3) * img_w * img_h)), true);
            CHECK_CUDA((cudaMemset(d_normal_aov, 0, sizeof(Vec3) * img_w * img_h)), true);
        }

        auto start_render = std::chrono::high_resolution_clock::now();
        render(P, img_w, img_h, frame_cam, miss_color, max_depth, spp, bvhState.Nodes, bvhState.AABBs, d_tris,
               d_triangle_obj_ids, d_object_materials, num_om,
               d_lights, num_lights, diffuse_bounce,
               d_emissiveTris, d_emissiveCDF, numEmissiveTris, totalEmissiveArea,
               d_image, d_albedo_aov, d_normal_aov, nee_mode,
               d_objectMedia, num_media,
               d_textures, numTextures,
               d_volumeRegions, numVolumeRegions,
               d_hdri, use_bdpt);
        auto end_render = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_render = end_render - start_render;
        printf("GPU Render Time: %.3f ms\n", ms_render.count());

        if (use_denoiser) {
            auto start_denoise = std::chrono::high_resolution_clock::now();
            OPTIX_CHECK(optixInit());

            CUcontext cuCtx = nullptr;
            OptixDeviceContext optixCtx = nullptr;
            OPTIX_CHECK(optixDeviceContextCreate(cuCtx, nullptr, &optixCtx));

            OptixDenoiserOptions denoiserOptions = {};
            denoiserOptions.guideAlbedo = 1;
            denoiserOptions.guideNormal = 1;
            denoiserOptions.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;

            OptixDenoiser denoiser = nullptr;
            OPTIX_CHECK(optixDenoiserCreate(optixCtx, OPTIX_DENOISER_MODEL_KIND_HDR,
                                            &denoiserOptions, &denoiser));

            OptixDenoiserSizes denoiserSizes = {};
            OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser,
                            static_cast<unsigned int>(img_w),
                            static_cast<unsigned int>(img_h),
                            &denoiserSizes));

            CUdeviceptr d_denoiserState = 0, d_denoiserScratch = 0;
            CHECK_CUDA((cudaMalloc(reinterpret_cast<void**>(&d_denoiserState), denoiserSizes.stateSizeInBytes)), true);
            CHECK_CUDA((cudaMalloc(reinterpret_cast<void**>(&d_denoiserScratch), denoiserSizes.withoutOverlapScratchSizeInBytes)), true);

            OPTIX_CHECK(optixDenoiserSetup(denoiser, nullptr,
                            static_cast<unsigned int>(img_w), static_cast<unsigned int>(img_h),
                            d_denoiserState, denoiserSizes.stateSizeInBytes,
                            d_denoiserScratch, denoiserSizes.withoutOverlapScratchSizeInBytes));

            CUdeviceptr d_hdrIntensity = 0, d_intensityScratch = 0;
            CHECK_CUDA((cudaMalloc(reinterpret_cast<void**>(&d_hdrIntensity), sizeof(float))), true);
            CHECK_CUDA((cudaMalloc(reinterpret_cast<void**>(&d_intensityScratch), denoiserSizes.computeIntensitySizeInBytes)), true);

            auto makeImage2D = [&](CUdeviceptr ptr) -> OptixImage2D {
                OptixImage2D img = {};
                img.data = ptr;
                img.width = static_cast<unsigned int>(img_w);
                img.height = static_cast<unsigned int>(img_h);
                img.rowStrideInBytes = static_cast<unsigned int>(img_w * sizeof(Vec3));
                img.pixelStrideInBytes = static_cast<unsigned int>(sizeof(Vec3));
                img.format = OPTIX_PIXEL_FORMAT_FLOAT3;
                return img;
            };

            OptixImage2D colorImg  = makeImage2D(reinterpret_cast<CUdeviceptr>(d_image));
            OptixImage2D albedoImg = makeImage2D(reinterpret_cast<CUdeviceptr>(d_albedo_aov));
            OptixImage2D normalImg = makeImage2D(reinterpret_cast<CUdeviceptr>(d_normal_aov));

            OPTIX_CHECK(optixDenoiserComputeIntensity(denoiser, nullptr,
                            &colorImg, d_hdrIntensity,
                            d_intensityScratch, denoiserSizes.computeIntensitySizeInBytes));

            OptixImage2D outputImg = colorImg;

            OptixDenoiserGuideLayer guideLayer = {};
            guideLayer.albedo = albedoImg;
            guideLayer.normal = normalImg;

            OptixDenoiserLayer layer = {};
            layer.input  = colorImg;
            layer.output = outputImg;

            OptixDenoiserParams params = {};
            params.hdrIntensity = d_hdrIntensity;
            params.blendFactor = 0.0f;
            params.hdrAverageColor = 0;
            params.temporalModeUsePreviousLayers = 0;

            OPTIX_CHECK(optixDenoiserInvoke(denoiser, nullptr, &params,
                            d_denoiserState, denoiserSizes.stateSizeInBytes,
                            &guideLayer, &layer, 1, 0, 0,
                            d_denoiserScratch, denoiserSizes.withoutOverlapScratchSizeInBytes));

            CHECK_CUDA((cudaDeviceSynchronize()), true);

            auto end_denoise = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> ms_denoise = end_denoise - start_denoise;
            printf("OptiX Denoise Time: %.3f ms\n", ms_denoise.count());

            cudaFree(reinterpret_cast<void*>(d_denoiserState));
            cudaFree(reinterpret_cast<void*>(d_denoiserScratch));
            cudaFree(reinterpret_cast<void*>(d_hdrIntensity));
            cudaFree(reinterpret_cast<void*>(d_intensityScratch));
            optixDenoiserDestroy(denoiser);
            optixDeviceContextDestroy(optixCtx);
        }

        CHECK_CUDA((cudaMemcpy(image.data(), d_image, sizeof(Vec3) * img_w * img_h, cudaMemcpyDeviceToHost)), true);
#else
        std::vector<Triangle> h_tris(P);
        for (size_t ti = 0; ti < P; ++ti) {
            const uint32_t i0 = globalMesh.indices[ti * 3 + 0];
            const uint32_t i1 = globalMesh.indices[ti * 3 + 1];
            const uint32_t i2 = globalMesh.indices[ti * 3 + 2];
            Vec3 n0 = make_vec3(0,0,0), n1 = make_vec3(0,0,0), n2 = make_vec3(0,0,0);
            if (!globalMesh.normals.empty()) {
                n0 = globalMesh.normals[i0]; n1 = globalMesh.normals[i1]; n2 = globalMesh.normals[i2];
            }
            Vec2 uv0 = make_vec2(0.0f, 0.0f), uv1 = make_vec2(0.0f, 0.0f), uv2 = make_vec2(0.0f, 0.0f);
            if (!globalMesh.uvs.empty()) {
                uv0 = globalMesh.uvs[i0]; uv1 = globalMesh.uvs[i1]; uv2 = globalMesh.uvs[i2];
            }
            h_tris[ti] = Triangle(globalMesh.positions[i0], globalMesh.positions[i1],
                                  globalMesh.positions[i2], n0, n1, n2, uv0, uv1, uv2);
        }

        std::fill(image.begin(), image.end(), make_vec3(0.0f, 0.0f, 0.0f));
        auto start_render = std::chrono::high_resolution_clock::now();
        render(P, img_w, img_h, frame_cam, miss_color, max_depth, spp, bvhState.Nodes, bvhState.AABBs, h_tris.data(),
               globalMesh.triangleObjIds.data(), objectMaterials.data(), num_om,
               render_lights.data(), num_lights, diffuse_bounce,
               h_emissiveTris.data(), h_emissiveCDF.data(), numEmissiveTris, totalEmissiveArea,
               image.data(), nullptr, nullptr, nee_mode,
               objectMediaList.data(), num_media,
               allTextureData.data(), numTextures,
               volumeRegionsList.data(), numVolumeRegions,
               /*hdri=*/(h_hdri_data.width > 0 ? &h_hdri_data : nullptr), use_bdpt,
               /*env_importance=*/h_env_imp);
        auto end_render = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_render = end_render - start_render;
        printf("CPU Render Time: %.3f ms\n", ms_render.count());
#endif

        const std::string fname = frame_filename(output_filename, frame, keyframes);
        std::vector<unsigned char> img_data(num_pixels * 3);
        for (size_t i = 0; i < num_pixels; ++i) {
            img_data[i * 3 + 0] = reinhard(image[i].x);
            img_data[i * 3 + 1] = reinhard(image[i].y);
            img_data[i * 3 + 2] = reinhard(image[i].z);
        }
        stbi_write_png(fname.c_str(), img_w, img_h, 3, img_data.data(), img_w * 3);
        printf("Image saved to %s\n", fname.c_str());
    }

#ifdef __CUDACC__
    cudaFree(d_tris);
    cudaFree(d_image);
    if (d_albedo_aov) cudaFree(d_albedo_aov);
    if (d_normal_aov) cudaFree(d_normal_aov);
    cudaFree(d_lights);
    if (d_emissiveTris) cudaFree(d_emissiveTris);
    if (d_emissiveCDF)  cudaFree(d_emissiveCDF);
    cudaFree(d_positions);
    cudaFree(d_normals);
    if (d_uvs) cudaFree(d_uvs);
    cudaFree(d_indices);
    cudaFree(d_triangle_obj_ids);
    cudaFree(d_object_materials);
    if (d_objectMedia) cudaFree(d_objectMedia);
    if (d_volumeRegions) cudaFree(d_volumeRegions);
    for (auto* p : d_volumeDensityPtrs) { if (p) cudaFree(p); }
    for (auto* p : d_volumeTemperaturePtrs) { if (p) cudaFree(p); }
    for (auto* p : d_volumeFlamePtrs) { if (p) cudaFree(p); }
    if (d_textures) cudaFree(d_textures);
    for (auto* p : d_texPixelPtrs) { if (p) cudaFree(p); }
    if (d_hdri_pixels) cudaFree(d_hdri_pixels);
    if (d_hdri)        cudaFree(d_hdri);
    delete hdri_owner;
#endif

    return 0;
}
