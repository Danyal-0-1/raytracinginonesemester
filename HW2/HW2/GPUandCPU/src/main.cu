#include "MeshOBJ.h"
#include "buffers.h"
#include "bvh.h"
#include "visualizer.h"
#include "warmup.h"
#include "scene.h"
#include "camera.h"
#include "query.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <numeric>
#include <chrono>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <algorithm>

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

    Vec3 n0 = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3 n1 = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3 n2 = make_vec3(0.0f, 0.0f, 0.0f);
    if (mesh.normals != nullptr) {
        n0 = mesh.normals[i0];
        n1 = mesh.normals[i1];
        n2 = mesh.normals[i2];
    }

    out[idx] = Triangle(v0, v1, v2, n0, n1, n2);
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

    // Rotate around X
    v = make_vec3(v.x, cx * v.y - sx * v.z, sx * v.y + cx * v.z);
    // Rotate around Y
    v = make_vec3(cy * v.x + sy * v.z, v.y, -sy * v.x + cy * v.z);
    // Rotate around Z
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
            const float invLen = 1.0f / sqrtf(len2);
            n = nRot * invLen;
        } else {
            n = make_vec3(0.0f, 0.0f, 1.0f);
        }
    }
}

static inline bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return static_cast<bool>(f);
}

static inline std::string basenameNoExt(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = name.find_last_of('.');
    if (dot != std::string::npos) name = name.substr(0, dot);
    return name;
}

static inline void appendTimingCsv(
    const std::string& csv_path,
    const char* backend,
    const char* scene_name,
    const char* preset_name,
    int spp,
    int width,
    int height,
    int bounces,
    double seconds)
{
    if (csv_path.empty()) return;

    bool write_header = true;
    {
        std::ifstream in(csv_path);
        if (in.good()) {
            in.seekg(0, std::ios::end);
            write_header = (in.tellg() == 0);
        }
    }

    std::ofstream out(csv_path, std::ios::app);
    if (!out) {
        std::cerr << "Warning: failed to open timing csv for append: " << csv_path << "\n";
        return;
    }
    if (write_header) {
        out << "backend,scene,preset,spp,width,height,bounces,seconds\n";
    }
    out << backend << ","
        << scene_name << ","
        << preset_name << ","
        << spp << ","
        << width << ","
        << height << ","
        << bounces << ","
        << std::fixed << std::setprecision(6) << seconds
        << "\n";
}

static inline void printUsage(const char* exe) {
    std::printf("Usage: %s [scene.json | obj1 obj2 ...] [--spp N] [--output out.png] [--timing-csv file.csv]\n", exe);
    std::printf("       [--env-preset sunrise|midday|sunset|night_moon|night_nomoon] [--env-enabled 0|1]\n");
    std::printf("       [--sun-elev deg] [--sun-azim deg] [--moon-intensity f] [--sky-intensity f]\n");
    std::printf("       [--sun-intensity f] [--turbidity f] [--star-intensity f] [--star-density f] [--exposure f]\n");
}

int main(int argc, char** argv)
{

    using vec3 = Vec3;
    using point3 = Vec3;

    std::vector<std::string> positional_args;
    std::string output_path = "render.png";
    std::string timing_csv_path;
    int cli_spp = -1;
    bool cli_has_env_preset = false;
    EnvironmentPreset cli_env_preset = EnvironmentPreset::Custom;
    bool cli_has_env_enabled = false;
    bool cli_env_enabled = false;
    bool cli_has_sun_elev = false;
    float cli_sun_elev_deg = 0.0f;
    bool cli_has_sun_azim = false;
    float cli_sun_azim_deg = 0.0f;
    bool cli_has_moon_intensity = false;
    float cli_moon_intensity = 0.0f;
    bool cli_has_sky_intensity = false;
    float cli_sky_intensity = 0.0f;
    bool cli_has_sun_intensity = false;
    float cli_sun_intensity = 0.0f;
    bool cli_has_turbidity = false;
    float cli_turbidity = 0.0f;
    bool cli_has_star_intensity = false;
    float cli_star_intensity = 0.0f;
    bool cli_has_star_density = false;
    float cli_star_density = 0.0f;
    bool cli_has_exposure = false;
    float cli_exposure = 1.0f;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        if (arg == "--spp") {
            if (i + 1 >= argc) {
                std::cerr << "--spp expects an integer value\n";
                return 1;
            }
            cli_spp = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--output") {
            if (i + 1 >= argc) {
                std::cerr << "--output expects a file path\n";
                return 1;
            }
            output_path = argv[++i];
            continue;
        }
        if (arg == "--timing-csv") {
            if (i + 1 >= argc) {
                std::cerr << "--timing-csv expects a file path\n";
                return 1;
            }
            timing_csv_path = argv[++i];
            continue;
        }
        if (arg == "--env-preset" || arg == "--preset") {
            if (i + 1 >= argc) {
                std::cerr << "--env-preset expects a preset string\n";
                return 1;
            }
            const std::string p = argv[++i];
            cli_env_preset = SceneIO::preset_from_string(p);
            cli_has_env_preset = true;
            continue;
        }
        if (arg == "--env-enabled") {
            if (i + 1 >= argc) {
                std::cerr << "--env-enabled expects 0 or 1\n";
                return 1;
            }
            cli_has_env_enabled = true;
            cli_env_enabled = (std::atoi(argv[++i]) != 0);
            continue;
        }
        if (arg == "--sun-elev") {
            if (i + 1 >= argc) { std::cerr << "--sun-elev expects a float\n"; return 1; }
            cli_has_sun_elev = true;
            cli_sun_elev_deg = std::atof(argv[++i]);
            continue;
        }
        if (arg == "--sun-azim") {
            if (i + 1 >= argc) { std::cerr << "--sun-azim expects a float\n"; return 1; }
            cli_has_sun_azim = true;
            cli_sun_azim_deg = std::atof(argv[++i]);
            continue;
        }
        if (arg == "--moon-intensity") {
            if (i + 1 >= argc) { std::cerr << "--moon-intensity expects a float\n"; return 1; }
            cli_has_moon_intensity = true;
            cli_moon_intensity = std::atof(argv[++i]);
            continue;
        }
        if (arg == "--sky-intensity") {
            if (i + 1 >= argc) { std::cerr << "--sky-intensity expects a float\n"; return 1; }
            cli_has_sky_intensity = true;
            cli_sky_intensity = std::atof(argv[++i]);
            continue;
        }
        if (arg == "--sun-intensity") {
            if (i + 1 >= argc) { std::cerr << "--sun-intensity expects a float\n"; return 1; }
            cli_has_sun_intensity = true;
            cli_sun_intensity = std::atof(argv[++i]);
            continue;
        }
        if (arg == "--turbidity") {
            if (i + 1 >= argc) { std::cerr << "--turbidity expects a float\n"; return 1; }
            cli_has_turbidity = true;
            cli_turbidity = std::atof(argv[++i]);
            continue;
        }
        if (arg == "--star-intensity") {
            if (i + 1 >= argc) { std::cerr << "--star-intensity expects a float\n"; return 1; }
            cli_has_star_intensity = true;
            cli_star_intensity = std::atof(argv[++i]);
            continue;
        }
        if (arg == "--star-density") {
            if (i + 1 >= argc) { std::cerr << "--star-density expects a float\n"; return 1; }
            cli_has_star_density = true;
            cli_star_density = std::atof(argv[++i]);
            continue;
        }
        if (arg == "--exposure") {
            if (i + 1 >= argc) { std::cerr << "--exposure expects a float\n"; return 1; }
            cli_has_exposure = true;
            cli_exposure = std::atof(argv[++i]);
            continue;
        }
        positional_args.push_back(arg);
    }

    std::vector<SceneObject> load_objects;
    Scene scene;
    bool has_scene = false;
    std::string scene_path;
    if (!positional_args.empty()) {
        const std::string first = positional_args.front();
        const bool is_scene =
            (first.size() >= 5 && first.substr(first.size() - 5) == ".json") ||
            (first.size() >= 6 && first.substr(first.size() - 6) == ".scene");
        if (is_scene) {
            scene_path = first;
            std::string err;
            if (!SceneIO::LoadSceneFromFile(scene_path, scene, &err)) {
                std::cerr << "Failed to load scene: " << err << "\n";
                return 1;
            }
            has_scene = true;
            const std::string base_dir = SceneIO::dirname(scene_path);
            const std::string project_dir = SceneIO::dirname(SceneIO::dirname(base_dir));
            for (const auto& obj : scene.objects) {
                if (!obj.type.empty() && obj.type != "mesh") continue;
                SceneObject resolved = obj;
                std::string path = resolved.path;
                if (!SceneIO::is_abs_path(path)) {
                    const std::string scene_relative = SceneIO::join_path(base_dir, path);
                    std::string project_relative = path;
                    if (project_relative.rfind("./", 0) == 0) {
                        project_relative = project_relative.substr(2);
                    }
                    project_relative = SceneIO::join_path(project_dir, project_relative);

                    if (fileExists(scene_relative)) {
                        path = scene_relative;
                    } else if (fileExists(path)) {
                        // Keep cwd-relative path as-is.
                    } else if (fileExists(project_relative)) {
                        path = project_relative;
                    } else {
                        // Fall back to scene-relative for clearer diagnostics.
                        path = scene_relative;
                    }
                }
                resolved.path = path;
                load_objects.push_back(resolved);
            }
        } else {
            for (const std::string& obj_path : positional_args) {
                SceneObject obj;
                obj.path = obj_path;
                load_objects.push_back(obj);
            }
        }
    } else {
        SceneObject obj;
        obj.path = "../assets/meshes/frog.obj";
        load_objects.push_back(obj);
    }

    Mesh globalMesh;
    std::vector<Material> objectMaterials;
    int nextObjectId = 0;

    for (const auto& obj : load_objects)
    {
        const std::string& path = obj.path;
        std::printf("Loading OBJ: %s\n", path.c_str());
        Mesh tempMesh;
        const int objIdBegin = nextObjectId;
        if (!LoadOBJ_ToMesh(path, tempMesh, nextObjectId))
        {
            std::cerr << "Failed to load OBJ: " << path << "\n";
            continue;
        }
        applyObjectTransform(tempMesh, obj);

        if (objectMaterials.size() < static_cast<size_t>(nextObjectId)) {
            objectMaterials.resize(nextObjectId, Material());
        }
        for (int oid = objIdBegin; oid < nextObjectId; ++oid) {
            objectMaterials[oid] = obj.material;
        }

        std::printf("  -> Loaded %zu triangles.\n", tempMesh.indices.size() / 3);
        AppendMesh(globalMesh, tempMesh);
    }

    if (globalMesh.positions.empty())
    {
        std::cerr << "No valid geometry loaded.\n";
        return 1;
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
    uint32_t* d_indices = nullptr;
    int32_t* d_triangle_obj_ids = nullptr;
    Material* d_object_materials = nullptr;

    const size_t bytesPos = globalMesh.positions.size() * sizeof(Vec3);
    const size_t bytesIdx = globalMesh.indices.size() * sizeof(uint32_t);
    const size_t bytesNrm = globalMesh.normals.size() * sizeof(Vec3);
    const size_t bytesTriObj = globalMesh.triangleObjIds.size() * sizeof(int32_t);
    const size_t bytesObjMat = objectMaterials.size() * sizeof(Material);

    CHECK_CUDA((cudaMalloc(&d_positions, bytesPos)), true);
    CHECK_CUDA((cudaMalloc(&d_indices, bytesIdx)), true);
    CHECK_CUDA((cudaMalloc(&d_triangle_obj_ids, bytesTriObj)), true);
    CHECK_CUDA((cudaMalloc(&d_object_materials, bytesObjMat)), true);
    if (!globalMesh.normals.empty()) {
        CHECK_CUDA((cudaMalloc(&d_normals, bytesNrm)), true);
    }

    CHECK_CUDA((cudaMemcpy(d_positions, globalMesh.positions.data(), bytesPos, cudaMemcpyHostToDevice)), true);
    CHECK_CUDA((cudaMemcpy(d_indices, globalMesh.indices.data(), bytesIdx, cudaMemcpyHostToDevice)), true);
    CHECK_CUDA((cudaMemcpy(d_triangle_obj_ids, globalMesh.triangleObjIds.data(), bytesTriObj, cudaMemcpyHostToDevice)), true);
    CHECK_CUDA((cudaMemcpy(d_object_materials, objectMaterials.data(), bytesObjMat, cudaMemcpyHostToDevice)), true);
    if (!globalMesh.normals.empty()) {
        CHECK_CUDA((cudaMemcpy(d_normals, globalMesh.normals.data(), bytesNrm, cudaMemcpyHostToDevice)), true);
    }

    MeshView d_mesh{};
    d_mesh.positions = d_positions;
    d_mesh.normals = d_normals;
    d_mesh.uvs = nullptr;
    d_mesh.indices = d_indices;
    d_mesh.triangleObjIds = d_triangle_obj_ids;
    d_mesh.numVertices = globalMesh.positions.size();
    d_mesh.numIndices = globalMesh.indices.size();
    d_mesh.numTriangles = P;

    CHECK_CUDA(bvh.calculateAABBs(d_mesh, bvhState.AABBs), true);
#else
    MeshView h_mesh = globalMesh.getView();
    bvh.calculateAABBs(h_mesh, bvhState.AABBs);
#endif

    AABB SceneBoundingBox;
    #ifdef __CUDACC__
        AABB default_aabb;
        // Reduce directly on device memory
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

        // --- GPU Warmup (Lightweight) ---
        // This initializes Thrust's internal allocators and kernels without touching real data.
        warmupGPU();

        auto start_gpu = std::chrono::high_resolution_clock::now();

        bvh.buildBVH(
            bvhState.Nodes,
            bvhState.AABBs,
            SceneBoundingBox,
            &TriangleIndices,
            static_cast<int>(P)
        );
        cudaDeviceSynchronize();
        auto end_gpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_gpu = end_gpu - start_gpu;
        printf("GPU LBVH Build Time: %.3f ms\n", ms_gpu.count());

    #else
        SceneBoundingBox = std::accumulate(
            bvhState.AABBs + (P - 1),
            bvhState.AABBs + (2 * P - 1),
            AABB(),
            [](const AABB& lhs, const AABB& rhs) {
                return AABB::merge(lhs, rhs);
            });

        std::vector<unsigned int> TriangleIndices(P);
        std::iota(TriangleIndices.begin(), TriangleIndices.end(), 0);
        auto start_cpu =  std::chrono::high_resolution_clock::now();
        bvh.buildBVH(
            bvhState.Nodes,
            bvhState.AABBs,
            SceneBoundingBox,
            TriangleIndices,
            static_cast<int>(P)
        );

        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_cpu = end_cpu - start_cpu;
        printf("CPU LBVH Build Time: %.3f ms\n", ms_cpu.count());

    #endif

    // --- Camera and Ray Generation ---
    int max_depth = has_scene ? scene.settings.max_depth : 1;
    int spp = has_scene ? scene.settings.spp : 1;
    if (cli_spp > 0) spp = cli_spp;
    bool diffuse_bounce = has_scene ? scene.settings.diffuse_bounce : true;

    Vec3 miss_color = has_scene ? scene.miss_color : make_vec3(0.0f, 0.0f, 0.0f);
    Environment env = has_scene ? scene.environment : Environment();
    if (cli_has_env_preset) {
        ApplyEnvironmentPreset(env, cli_env_preset);
        env.enabled = true;
    }
    if (cli_has_env_enabled) {
        env.enabled = cli_env_enabled;
    }
    if (cli_has_sky_intensity) env.sky_intensity = cli_sky_intensity;
    if (cli_has_sun_intensity) env.sun_intensity = cli_sun_intensity;
    if (cli_has_turbidity) env.turbidity = cli_turbidity;
    if (cli_has_moon_intensity) {
        env.moon_intensity = cli_moon_intensity;
        env.moon_enabled = (cli_moon_intensity > 0.0f);
    }
    if (cli_has_star_intensity) env.star_intensity = cli_star_intensity;
    if (cli_has_star_density) env.star_density = cli_star_density;
    if (cli_has_exposure) env.exposure = cli_exposure;
    if (cli_has_sun_elev || cli_has_sun_azim) {
        float elev = 0.0f;
        float azim = 0.0f;
        ElevationAzimuthFromDirectionDeg(env.sun_dir, elev, azim);
        if (cli_has_sun_elev) elev = cli_sun_elev_deg;
        if (cli_has_sun_azim) azim = cli_sun_azim_deg;
        env.sun_dir = DirectionFromElevationAzimuthDeg(elev, azim);
        env.preset = EnvironmentPreset::Custom;
    }
    Camera cam = has_scene ? scene.camera : Camera();
    std::vector<Light> render_lights = scene.lights;
    if (render_lights.empty()) {
        if (!env.enabled) {
            Light fallback;
            fallback.position = make_vec3(-3.0f, 0.0f, 1.0f);
            fallback.color = make_vec3(1.0f, 1.0f, 1.0f);
            fallback.intensity = 1;
            render_lights.push_back(fallback);
            printf("No lights in scene, using fallback light.\n");
        } else {
            printf("Environment enabled with zero point lights; fallback light disabled.\n");
        }
    }
    const int num_lights = static_cast<int>(render_lights.size());
    const int num_object_materials = static_cast<int>(objectMaterials.size());

    const int img_w = cam.pixel_width;
    const int img_h = cam.pixel_height;
    const int num_pixels = img_w * img_h;
    // const int num_rays = num_pixels * spp;
    std::vector<Vec3> image(num_pixels, make_vec3(0.0f, 0.0f, 0.0f));
    const std::string scene_name = has_scene ? basenameNoExt(scene_path) : std::string("mesh_input");
    std::string timing_preset = env.enabled
        ? std::string(EnvironmentPresetToString(env.preset))
        : (scene_name.find("pointlight") != std::string::npos ? std::string("pointlight") : std::string("env_disabled"));
    const char* preset_name = timing_preset.c_str();
    double render_seconds = 0.0;

#ifdef __CUDACC__
    Triangle* d_tris = nullptr;
    Vec3* d_image = nullptr;
    Light* d_lights = nullptr;

    CHECK_CUDA((cudaMalloc(&d_tris, sizeof(Triangle) * P)), true);
    CHECK_CUDA((cudaMalloc(&d_image, sizeof(Vec3) * img_w * img_h)), true);
    if (num_lights > 0) {
        CHECK_CUDA((cudaMalloc(&d_lights, sizeof(Light) * num_lights)), true);
        CHECK_CUDA((cudaMemcpy(d_lights, render_lights.data(), sizeof(Light) * num_lights, cudaMemcpyHostToDevice)), true);
    }

    const int threads = 256;
    const int tri_blocks = (static_cast<int>(P) + threads - 1) / threads;
    buildTrianglesKernel<<<tri_blocks, threads>>>(d_mesh, d_tris, static_cast<int>(P));
    CHECK_CUDA((cudaDeviceSynchronize()), true);

    // Warm up with a tiny launch to pay first-launch/JIT cost without full-frame work.
    render(P, 1, 1, cam, miss_color, env, max_depth, 1, bvhState.Nodes, bvhState.AABBs, d_tris,
           d_triangle_obj_ids, d_object_materials, num_object_materials,
           d_lights, num_lights, diffuse_bounce, d_image);

    // Zero image buffer before the real render (warmup wrote into it)
    CHECK_CUDA((cudaMemset(d_image, 0, sizeof(Vec3) * img_w * img_h)), true);

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    CHECK_CUDA((cudaEventCreate(&ev_start)), true);
    CHECK_CUDA((cudaEventCreate(&ev_stop)), true);
    CHECK_CUDA((cudaEventRecord(ev_start)), true);

    render(P, img_w, img_h, cam, miss_color, env, max_depth, spp, bvhState.Nodes, bvhState.AABBs, d_tris,
           d_triangle_obj_ids, d_object_materials, num_object_materials,
           d_lights, num_lights, diffuse_bounce, d_image);

    CHECK_CUDA((cudaEventRecord(ev_stop)), true);
    CHECK_CUDA((cudaEventSynchronize(ev_stop)), true);
    float ms_render = 0.0f;
    CHECK_CUDA((cudaEventElapsedTime(&ms_render, ev_start, ev_stop)), true);
    render_seconds = static_cast<double>(ms_render) / 1000.0;

    CHECK_CUDA((cudaMemcpy(image.data(), d_image, sizeof(Vec3) * img_w * img_h, cudaMemcpyDeviceToHost)), true);

    printf("GPU Render Time: %.3f ms\n", ms_render);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_tris);
    cudaFree(d_image);
    if (d_lights) cudaFree(d_lights);
    cudaFree(d_positions);
    cudaFree(d_normals);
    cudaFree(d_indices);
    cudaFree(d_triangle_obj_ids);
    cudaFree(d_object_materials);
#else
    std::vector<Triangle> h_tris(P);
    for (size_t i = 0; i < P; ++i) {
        const uint32_t i0 = globalMesh.indices[i * 3 + 0];
        const uint32_t i1 = globalMesh.indices[i * 3 + 1];
        const uint32_t i2 = globalMesh.indices[i * 3 + 2];

        Vec3 n0 = make_vec3(0.0f, 0.0f, 0.0f);
        Vec3 n1 = make_vec3(0.0f, 0.0f, 0.0f);
        Vec3 n2 = make_vec3(0.0f, 0.0f, 0.0f);
        if (!globalMesh.normals.empty()) {
            n0 = globalMesh.normals[i0];
            n1 = globalMesh.normals[i1];
            n2 = globalMesh.normals[i2];
        }

        h_tris[i] = Triangle(globalMesh.positions[i0], globalMesh.positions[i1], globalMesh.positions[i2], n0, n1, n2);
    }
    auto start_render = std::chrono::high_resolution_clock::now();
    render(P, img_w, img_h, cam, miss_color, env, max_depth, spp, bvhState.Nodes, bvhState.AABBs, h_tris.data(),
           globalMesh.triangleObjIds.data(), objectMaterials.data(), num_object_materials,
           render_lights.data(), num_lights, diffuse_bounce, image.data());
    auto end_render = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_render = end_render - start_render;
    render_seconds = ms_render.count() / 1000.0;
    printf("CPU Render Time: %.3f ms\n", ms_render.count());
#endif


//     std::cout << "Exporting all AABBs to file...\n";
//     size_t totalNodes = 2 * P - 1;
//     std::vector<AABB> allAABBs(totalNodes);
// #ifdef __CUDACC__
//     cudaMemcpy(allAABBs.data(), bvhState.AABBs, sizeof(AABB) * totalNodes, cudaMemcpyDeviceToHost);
// #else
//     for(size_t i=0; i<totalNodes; ++i) allAABBs[i] = bvhState.AABBs[i];
// #endif
//     ExportAABBsToOBJ("bvh_.obj", allAABBs.data(), totalNodes);

    // write image to disk
    std::vector<unsigned char> img_data(num_pixels * 3);
    for (size_t i = 0; i < num_pixels; ++i)    {
        img_data[i * 3 + 0] = static_cast<unsigned char>(255.0f * std::min(image[i].x, 1.0f));
        img_data[i * 3 + 1] = static_cast<unsigned char>(255.0f * std::min(image[i].y, 1.0f));
        img_data[i * 3 + 2] = static_cast<unsigned char>(255.0f * std::min(image[i].z, 1.0f));
    }
    stbi_write_png(output_path.c_str(), img_w, img_h, 3, img_data.data(), img_w * 3);
    printf("Image saved to %s\n", output_path.c_str());

#ifdef __CUDACC__
    const char* backend_name = "gpu";
#else
    const char* backend_name = "cpu";
#endif

    printf("TIMING backend=%s scene=%s preset=%s spp=%d width=%d height=%d bounces=%d seconds=%.6f\n",
           backend_name, scene_name.c_str(), preset_name, spp, img_w, img_h, max_depth, render_seconds);
    appendTimingCsv(
        timing_csv_path,
        backend_name,
        scene_name.c_str(),
        preset_name,
        spp,
        img_w,
        img_h,
        max_depth,
        render_seconds);

    return 0;
}
