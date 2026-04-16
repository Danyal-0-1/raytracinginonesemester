// include/query.h
#pragma once

#include "camera.h"
#include "ray.h"
#include "MeshOBJ.h"
#include "brdf.h"
#include "shader.h"
#include "bvh.h"
#include "antialias.h"
#include "medium.h"
#include "texture.h"

struct Light;

HYBRID_FUNC inline float power_heuristic(float pdf_a, float pdf_b) {
    const float a2 = pdf_a * pdf_a;
    const float b2 = pdf_b * pdf_b;
    return (a2 + b2 > 0.0f) ? (a2 / (a2 + b2)) : 0.0f;
}

HYBRID_FUNC inline bool update_slab_interval(
    float origin,
    float direction,
    float slab_min,
    float slab_max,
    float& t_enter,
    float& t_exit)
{
    if (fabsf(direction) < 1e-8f) {
        return origin >= slab_min && origin <= slab_max;
    }

    const float inv_dir = 1.0f / direction;
    float t0 = (slab_min - origin) * inv_dir;
    float t1 = (slab_max - origin) * inv_dir;
    if (t0 > t1) {
        const float tmp = t0;
        t0 = t1;
        t1 = tmp;
    }

    t_enter = fmaxf(t_enter, t0);
    t_exit  = fminf(t_exit,  t1);
    return t_enter <= t_exit;
}

// GPU-friendly volume region struct (mirrors VolumeRegion from scene.h)
struct VolumeRegionGPU {
    Vec3 min_bounds;
    Vec3 max_bounds;
    HomogeneousMedium medium;
    const float* density_data = nullptr;
    int density_nx = 0;
    int density_ny = 0;
    int density_nz = 0;
    float density_scale = 1.0f;
    float density_majorant = -1.0f;

    const float* temperature_data = nullptr;
    int temperature_nx = 0;
    int temperature_ny = 0;
    int temperature_nz = 0;
    float temperature_scale = 1.0f;

    const float* flame_data = nullptr;
    int flame_nx = 0;
    int flame_ny = 0;
    int flame_nz = 0;
    float flame_scale = 1.0f;

    // Emission (blackbody derived from temperature/flame channels)
    float emission_scale = 0.0f;
    float emission_temp_min = 500.0f;
    float emission_temp_max = 2500.0f;

    // Blackbody RGB approximation (Tanner Helland / CIE fit)
    HYBRID_FUNC inline static Vec3 blackbody_rgb(float temp_K) {
        const float t = temp_K / 100.0f;
        float r, g, b;
        if (t <= 66.0f) {
            r = 1.0f;
            g = (t > 1.0f) ? (0.390081579f * logf(t) - 0.631841443f) : 0.0f;
            b = (t > 19.0f) ? (0.543206789f * logf(t - 10.0f) - 1.196254089f) : 0.0f;
        } else {
            r = 1.292936186f * powf(t - 60.0f, -0.1332047592f);
            g = 1.129890861f * powf(t - 60.0f, -0.0755148492f);
            b = 1.0f;
        }
        return make_vec3(
            fminf(fmaxf(r, 0.0f), 1.0f),
            fminf(fmaxf(g, 0.0f), 1.0f),
            fminf(fmaxf(b, 0.0f), 1.0f));
    }

    HYBRID_FUNC inline static float saturate(float x) {
        return fminf(fmaxf(x, 0.0f), 1.0f);
    }

    HYBRID_FUNC inline bool has_temperature_grid() const {
        return temperature_data != nullptr &&
               temperature_nx > 0 && temperature_ny > 0 && temperature_nz > 0;
    }

    HYBRID_FUNC inline bool has_flame_grid() const {
        return flame_data != nullptr && flame_nx > 0 && flame_ny > 0 && flame_nz > 0;
    }

    HYBRID_FUNC inline float normalized_density(float scaled_density) const {
        if (density_scale > 1e-8f) {
            return saturate(scaled_density / density_scale);
        }
        return saturate(scaled_density);
    }

    HYBRID_FUNC inline float sample_scalar_grid(
        const Vec3& p,
        const float* grid_data,
        int nx,
        int ny,
        int nz,
        float grid_scale,
        float fallback) const
    {
        if (grid_data == nullptr || nx <= 0 || ny <= 0 || nz <= 0) return fallback;

        const float extent_x = max_bounds.x - min_bounds.x;
        const float extent_y = max_bounds.y - min_bounds.y;
        const float extent_z = max_bounds.z - min_bounds.z;
        if (extent_x <= 1e-8f || extent_y <= 1e-8f || extent_z <= 1e-8f) return 0.0f;

        const float u = fminf(fmaxf((p.x - min_bounds.x) / extent_x, 0.0f), 1.0f);
        const float v = fminf(fmaxf((p.y - min_bounds.y) / extent_y, 0.0f), 1.0f);
        const float w = fminf(fmaxf((p.z - min_bounds.z) / extent_z, 0.0f), 1.0f);

        const float gx = u * float(nx - 1);
        const float gy = v * float(ny - 1);
        const float gz = w * float(nz - 1);

        const int x0 = int(floorf(gx));
        const int y0 = int(floorf(gy));
        const int z0 = int(floorf(gz));
        const int x1 = (x0 + 1 < nx) ? x0 + 1 : x0;
        const int y1 = (y0 + 1 < ny) ? y0 + 1 : y0;
        const int z1 = (z0 + 1 < nz) ? z0 + 1 : z0;

        const float tx = gx - float(x0);
        const float ty = gy - float(y0);
        const float tz = gz - float(z0);

        auto voxel = [&](int x, int y, int z) -> float {
            const int idx = (z * ny + y) * nx + x;
            return grid_data[idx];
        };

        const float c000 = voxel(x0, y0, z0);
        const float c100 = voxel(x1, y0, z0);
        const float c010 = voxel(x0, y1, z0);
        const float c110 = voxel(x1, y1, z0);
        const float c001 = voxel(x0, y0, z1);
        const float c101 = voxel(x1, y0, z1);
        const float c011 = voxel(x0, y1, z1);
        const float c111 = voxel(x1, y1, z1);

        const float c00 = c000 * (1.0f - tx) + c100 * tx;
        const float c10 = c010 * (1.0f - tx) + c110 * tx;
        const float c01 = c001 * (1.0f - tx) + c101 * tx;
        const float c11 = c011 * (1.0f - tx) + c111 * tx;
        const float c0 = c00 * (1.0f - ty) + c10 * ty;
        const float c1 = c01 * (1.0f - ty) + c11 * ty;

        return grid_scale * (c0 * (1.0f - tz) + c1 * tz);
    }

    // Sample emission radiance from temperature/flame channels, with a density fallback.
    HYBRID_FUNC inline Vec3 sample_emission(
        float scaled_density,
        float scaled_temperature,
        float scaled_flame) const
    {
        if (emission_scale <= 0.0f || scaled_density <= 0.0f) return make_vec3(0.0f, 0.0f, 0.0f);

        const float temp_norm = saturate(scaled_temperature);
        const float flame_norm = saturate(scaled_flame);
        if (flame_norm <= 0.0f) return make_vec3(0.0f, 0.0f, 0.0f);

        const float temp = emission_temp_min + temp_norm * (emission_temp_max - emission_temp_min);
        const Vec3 color = blackbody_rgb(temp);
        const float rel = temp / fmaxf(emission_temp_max, 1.0f);
        const float intensity = emission_scale * flame_norm * rel * rel * rel * rel;
        return color * intensity;
    }

    HYBRID_FUNC inline bool contains(const Vec3& p) const {
        return p.x >= min_bounds.x && p.x <= max_bounds.x &&
               p.y >= min_bounds.y && p.y <= max_bounds.y &&
               p.z >= min_bounds.z && p.z <= max_bounds.z;
    }

    HYBRID_FUNC inline bool has_density_grid() const {
        return density_data != nullptr && density_nx > 0 && density_ny > 0 && density_nz > 0;
    }

    HYBRID_FUNC inline bool ray_interval(const Ray& ray, float& t_enter, float& t_exit) const {
        t_enter = 0.0f;
        t_exit = 1e30f;
        return update_slab_interval(ray.orig.x, ray.dir.x, min_bounds.x, max_bounds.x, t_enter, t_exit) &&
               update_slab_interval(ray.orig.y, ray.dir.y, min_bounds.y, max_bounds.y, t_enter, t_exit) &&
               update_slab_interval(ray.orig.z, ray.dir.z, min_bounds.z, max_bounds.z, t_enter, t_exit) &&
               t_exit >= 0.0f;
    }

    HYBRID_FUNC inline float segment_length(const Ray& ray, float max_distance) const {
        float t_enter, t_exit;
        if (!ray_interval(ray, t_enter, t_exit)) return 0.0f;
        const float seg_start = fmaxf(0.0f, t_enter);
        const float seg_end = fminf(max_distance, t_exit);
        return (seg_end > seg_start) ? (seg_end - seg_start) : 0.0f;
    }

    HYBRID_FUNC inline float sample_density(const Vec3& p) const {
        return sample_scalar_grid(
            p, density_data, density_nx, density_ny, density_nz, density_scale, 1.0f);
    }

    HYBRID_FUNC inline float sample_temperature(const Vec3& p, float scaled_density) const {
        const float fallback = normalized_density(scaled_density);
        if (!has_temperature_grid() && has_flame_grid()) {
            return sample_scalar_grid(
                p, flame_data, flame_nx, flame_ny, flame_nz, flame_scale, fallback);
        }
        return sample_scalar_grid(
            p,
            temperature_data,
            temperature_nx,
            temperature_ny,
            temperature_nz,
            temperature_scale,
            fallback);
    }

    HYBRID_FUNC inline float sample_flame(const Vec3& p, float scaled_density) const {
        const float fallback = normalized_density(scaled_density);
        if (!has_flame_grid() && has_temperature_grid()) {
            return sample_scalar_grid(
                p,
                temperature_data,
                temperature_nx,
                temperature_ny,
                temperature_nz,
                temperature_scale,
                fallback);
        }
        return sample_scalar_grid(
            p,
            flame_data,
            flame_nx,
            flame_ny,
            flame_nz,
            flame_scale,
            fallback);
    }
};

// ============================================================
// sampleHDRI — equirectangular sample for a Z-up world direction.
//   U = azimuth  (atan2(y, x) mapped to [0,1])
//   V = elevation (asin(z)    mapped to [0,1], 0=nadir, 1=zenith)
// ============================================================
HYBRID_FUNC inline Vec3 sampleHDRI(const HDRTextureData& hdri, const Vec3& dir) {
    constexpr float INV_2PI = 0.15915494309f;
    constexpr float INV_PI  = 0.31830988618f;
    float dz = fminf(fmaxf(dir.z, -1.0f), 1.0f);
    float u  = 0.5f + atan2f(dir.y, dir.x) * INV_2PI;
    float v  = 0.5f + asinf(dz) * INV_PI;
    return hdri.sample(u, v);
}

void render(
    const size_t numTriangles,
    int W, int H,
    const Camera cam,
    const Vec3 missColor,
    const int max_depth,
    const int spp,
    const BVHNode* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    const Triangle* __restrict__ triangles,
    const int32_t* __restrict__ triObjectIds,
    const Material* __restrict__ objectMaterials,
    const int numObjectMaterials,
    const Light* __restrict__ lights,
    const int numLights,
    const bool diffuse_bounce,
    const EmissiveTriInfo* __restrict__ emissiveTris,
    const float* __restrict__ emissiveCDF,
    const int numEmissiveTris,
    const float totalEmissiveArea,
    Vec3* __restrict__ output,
    Vec3* __restrict__ albedo_aov = nullptr,
    Vec3* __restrict__ normal_aov = nullptr,
    int nee_mode = 2,
    const HomogeneousMedium* __restrict__ objectMedia = nullptr,
    int numObjectMedia = 0,
    const TextureData* __restrict__ textures = nullptr,
    int numTextures = 0,
    const VolumeRegionGPU* __restrict__ volumeRegions = nullptr,
    int numVolumeRegions = 0,
    const HDRTextureData* __restrict__ hdri = nullptr);


HYBRID_FUNC inline float rng_next(unsigned int& state) {
    state = state * 1664525u + 1013904223u;
    unsigned int h = state;
    h = (h ^ 61u) ^ (h >> 16u);
    h *= 9u;
    h ^= h >> 4u;
    h *= 0x27d4eb2du;
    h ^= h >> 15u;
    return float(h) / float(0xFFFFFFFFu);
}

HYBRID_FUNC inline unsigned int make_rng_seed(int x, int y, int sample) {
    return (unsigned int)x * 73856093u
         ^ (unsigned int)y * 19349663u
         ^ (unsigned int)sample * 83492791u;
}

HYBRID_FUNC inline Vec3 random_unit_vector(unsigned int& state) {
    for (;;) {
        float x = 2.0f * rng_next(state) - 1.0f;
        float y = 2.0f * rng_next(state) - 1.0f;
        float z = 2.0f * rng_next(state) - 1.0f;
        float lensq = x*x + y*y + z*z;
        if (lensq > 1e-10f && lensq <= 1.0f) {
            float inv = 1.0f / sqrtf(lensq);
            return make_vec3(x * inv, y * inv, z * inv);
        }
    }
}

HYBRID_FUNC inline Vec3 random_on_hemisphere(const Vec3& normal, unsigned int& state) {
    Vec3 on_unit_sphere = random_unit_vector(state);
    if (dot(on_unit_sphere, normal) > 0.0f) return on_unit_sphere;
    return make_vec3(-on_unit_sphere.x, -on_unit_sphere.y, -on_unit_sphere.z);
}

HYBRID_FUNC inline void build_onb(const Vec3& N, Vec3& tangent, Vec3& bitangent) {
    Vec3 up = (fabsf(N.z) < 0.9f) ? make_vec3(0.0f, 0.0f, 1.0f)
                                   : make_vec3(1.0f, 0.0f, 0.0f);
    Vec3 cr  = cross(up, N);
    float len = sqrtf(dot(cr, cr));
    tangent   = (len > 1e-10f) ? cr * (1.0f / len) : make_vec3(1.0f, 0.0f, 0.0f);
    bitangent = cross(N, tangent);
}

HYBRID_FUNC inline Vec3 cosine_hemisphere_sample(const Vec3& N, unsigned int& rng_state) {
    const float u1  = rng_next(rng_state);
    const float u2  = rng_next(rng_state);
    const float r   = sqrtf(u1);
    const float phi = 6.28318530717958647692f * u2;
    const float x   = r * cosf(phi);
    const float y   = r * sinf(phi);
    const float z   = sqrtf(fmaxf(0.0f, 1.0f - u1));
    Vec3 tangent, bitangent;
    build_onb(N, tangent, bitangent);
    Vec3 wi = tangent * x + bitangent * y + N * z;
    float wilen = sqrtf(dot(wi, wi));
    return (wilen > 1e-10f) ? wi * (1.0f / wilen) : N;
}

HYBRID_FUNC inline float BRDFSamplingPdf(const HitRecord& rec,
                                          const Vec3& wo,
                                          const Vec3& wi,
                                          bool allow_diffuse = true)
{
    if (!allow_diffuse && rec.mat.ks <= 0.0f) return 0.0f;
    return allow_diffuse
         ? BRDFpdf(rec, wo, wi)
         : BlinnPhongSpecularPDF(normalize(rec.normal), wo, wi, rec.mat.shininess);
}

HYBRID_FUNC inline Vec3 sample_blinn_phong(const Vec3& N, const Vec3& V,
                                             float shininess,
                                             unsigned int& rng_state, float& pdf)
{
    const float u1    = rng_next(rng_state);
    const float u2    = rng_next(rng_state);
    const float twoPi = 6.28318530717958647692f;

    const float cos_theta_H = powf(fmaxf(u1, 1e-10f), 1.0f / (shininess + 2.0f));
    const float sin_theta_H = sqrtf(fmaxf(0.0f, 1.0f - cos_theta_H * cos_theta_H));
    const float phi         = twoPi * u2;

    Vec3 tangent, bitangent;
    build_onb(N, tangent, bitangent);

    Vec3 H_raw = tangent   * (sin_theta_H * cosf(phi))
               + bitangent * (sin_theta_H * sinf(phi))
               + N         * cos_theta_H;
    float hlen = sqrtf(dot(H_raw, H_raw));
    Vec3  H    = (hlen > 1e-10f) ? H_raw * (1.0f / hlen) : N;

    const float VdotH = fmaxf(dot(V, H), 0.0f);
    Vec3 wi_raw = H * (2.0f * VdotH) - V;
    float wilen = sqrtf(dot(wi_raw, wi_raw));
    Vec3  wi    = (wilen > 1e-10f) ? wi_raw * (1.0f / wilen) : N;

    pdf = BlinnPhongSpecularPDF(N, V, wi, shininess);
    return wi;
}

HYBRID_FUNC inline Vec3 SampleBRDF(const HitRecord& rec, const Vec3& Vo,
                                     unsigned int& rng_state, float& out_pdf,
                                     bool allow_diffuse = true)
{
    const Vec3  N      = normalize(rec.normal);
    const float kd     = allow_diffuse ? rec.mat.kd : 0.0f;
    const float ks     = rec.mat.ks;
    const float total  = kd + ks;
    if (total <= 0.0f) { out_pdf = 0.0f; return make_vec3(0,0,0); }

    Vec3 wi;
    if (kd > 0.0f && rng_next(rng_state) < kd / total) {
        wi = cosine_hemisphere_sample(N, rng_state);
    } else {
        float dummy;
        wi = sample_blinn_phong(N, Vo, rec.mat.shininess, rng_state, dummy);
    }
    out_pdf = BRDFSamplingPdf(rec, Vo, wi, allow_diffuse);
    return wi;
}

HYBRID_FUNC inline HitRecord intersectTriangle(const Ray& r,
                                                const Triangle& tri,
                                                float tmin, float tmax)
{
    HitRecord rec{};
    rec.triangleIdx = -1;

    const Vec3 e1 = tri.v1 - tri.v0;
    const Vec3 e2 = tri.v2 - tri.v0;
    const Vec3 pvec = cross(r.direction(), e2);
    const float det = dot(e1, pvec);
    if (fabsf(det) < 1e-8f) { rec.hit = false; return rec; }
    const float invDet = 1.0f / det;

    const Vec3 tvec = r.origin() - tri.v0;
    const float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) { rec.hit = false; return rec; }

    const Vec3 qvec = cross(tvec, e1);
    const float v = dot(r.direction(), qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) { rec.hit = false; return rec; }

    const float t = dot(e2, qvec) * invDet;
    if (t < tmin || t > tmax) { rec.hit = false; return rec; }

    rec.hit  = true;
    rec.t    = t;
    rec.p    = r.origin() + r.direction() * t;

    Vec3 geomN = normalize(cross(e1, e2));
    rec.front_face = dot(r.direction(), geomN) < 0.0f;
    if (!rec.front_face) geomN = -geomN;

    Vec3 shadingN = (1.0f - u - v) * tri.n0 + u * tri.n1 + v * tri.n2;
    if (length_squared(shadingN) < 1e-12f) {
        shadingN = geomN;
    } else {
        shadingN = normalize(shadingN);
        if (dot(shadingN, geomN) < 0.0f) shadingN = -shadingN;
    }
    rec.normal = shadingN;
    rec.mat    = Material();

    float w0 = 1.0f - u - v;
    rec.uv.x = w0 * tri.uv0.x + u * tri.uv1.x + v * tri.uv2.x;
    rec.uv.y = w0 * tri.uv0.y + u * tri.uv1.y + v * tri.uv2.y;

    return rec;
}

HYBRID_FUNC inline void assignMaterialToHit(
    HitRecord& hitRecord,
    const int numTriangles,
    const int32_t* __restrict__ triObjectIds,
    const Material* __restrict__ objectMaterials,
    const int numObjectMaterials,
    const TextureData* __restrict__ textures = nullptr,
    int numTextures = 0)
{
    if (!hitRecord.hit || triObjectIds == nullptr || objectMaterials == nullptr ||
        hitRecord.triangleIdx < 0 || hitRecord.triangleIdx >= numTriangles)
        return;

    const int objId = triObjectIds[hitRecord.triangleIdx];
    if (objId >= 0 && objId < numObjectMaterials)
        hitRecord.mat = objectMaterials[objId];

    ApplyTextures(hitRecord, textures, numTextures);
}

HYBRID_FUNC inline int binary_search_cdf(const float* cdf, int n, float u) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] < u) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// -----------------------------------------------------------------------
// findVolumeAtPoint
// Inline helper: returns the first VolumeRegionGPU that contains p, or a
// disabled medium if no region matches.  Written as a plain inline
// function rather than a lambda so it compiles cleanly under CUDA.
// -----------------------------------------------------------------------
HYBRID_FUNC inline HomogeneousMedium findVolumeAtPoint(
    const Vec3& p,
    const VolumeRegionGPU* __restrict__ volumeRegions,
    int numVolumeRegions)
{
    if (volumeRegions != nullptr) {
        for (int i = 0; i < numVolumeRegions; ++i) {
            if (volumeRegions[i].contains(p))
                return volumeRegions[i].medium;
        }
    }
    HomogeneousMedium none;
    none.enabled = false;
    return none;
}

HYBRID_FUNC inline int findVolumeIndexAtPoint(
    const Vec3& p,
    const VolumeRegionGPU* __restrict__ volumeRegions,
    int numVolumeRegions)
{
    if (volumeRegions != nullptr) {
        for (int i = 0; i < numVolumeRegions; ++i) {
            if (volumeRegions[i].contains(p))
                return i;
        }
    }
    return -1;
}

HYBRID_FUNC inline int findNextVolumeAlongRay(
    const Ray& ray,
    const VolumeRegionGPU* __restrict__ volumeRegions,
    int numVolumeRegions,
    float& out_t_enter,
    float& out_t_exit)
{
    int best_idx = -1;
    float best_enter = 1e30f;
    float best_exit = 1e30f;

    if (volumeRegions != nullptr) {
        for (int i = 0; i < numVolumeRegions; ++i) {
            float t_enter = 0.0f;
            float t_exit = 0.0f;
            if (!volumeRegions[i].ray_interval(ray, t_enter, t_exit)) continue;
            if (t_exit <= RT_EPS) continue;

            const float enter = fmaxf(t_enter, 0.0f);
            if (enter < best_enter) {
                best_enter = enter;
                best_exit = t_exit;
                best_idx = i;
            }
        }
    }

    out_t_enter = best_enter;
    out_t_exit = best_exit;
    return best_idx;
}

HYBRID_FUNC inline Vec3 estimateTransmittance(
    const Ray& ray,
    float tmax,
    const VolumeRegionGPU* volumeRegion,
    const HomogeneousMedium& medium,
    unsigned int& rng_state)
{
    if (tmax <= 0.0f || !medium.has_extinction()) {
        return make_vec3(1.0f, 1.0f, 1.0f);
    }
    if (volumeRegion == nullptr || !volumeRegion->has_density_grid()) {
        return medium.transmittance(tmax);
    }

    const float sigma_maj = volumeRegion->density_majorant;
    if (sigma_maj <= 1e-8f) {
        return make_vec3(1.0f, 1.0f, 1.0f);
    }

    Vec3 Tr = make_vec3(1.0f, 1.0f, 1.0f);
    float t = 0.0f;
    while (true) {
        const float xi = rng_next(rng_state);
        t += -logf(fmaxf(1.0f - xi, 1e-8f)) / sigma_maj;
        if (t >= tmax) break;

        const float density = volumeRegion->sample_density(ray.at(t));
        const Vec3 sigma_t_local = medium.sigma_t * density;
        Tr = Tr * make_vec3(
            fmaxf(0.0f, 1.0f - sigma_t_local.x / sigma_maj),
            fmaxf(0.0f, 1.0f - sigma_t_local.y / sigma_maj),
            fmaxf(0.0f, 1.0f - sigma_t_local.z / sigma_maj));
    }
    return Tr;
}

HYBRID_FUNC inline bool sampleHeterogeneousScatter(
    const Ray& ray,
    float tmax,
    const VolumeRegionGPU& volumeRegion,
    const HomogeneousMedium& medium,
    int channel,
    unsigned int& rng_state,
    float& t_scatter,
    Vec3* out_emission = nullptr)
{
    const float sigma_maj = volumeRegion.density_majorant;
    if (tmax <= 0.0f || sigma_maj <= 1e-8f) return false;

    const float inv_sigma_maj = 1.0f / sigma_maj;
    const bool accumulate_Le = (out_emission != nullptr && volumeRegion.emission_scale > 0.0f);
    Vec3 Tr_run = make_vec3(1.0f, 1.0f, 1.0f);
    Vec3 Le_acc = make_vec3(0.0f, 0.0f, 0.0f);

    float t = 0.0f;
    while (true) {
        const float xi = rng_next(rng_state);
        t += -logf(fmaxf(1.0f - xi, 1e-8f)) * inv_sigma_maj;
        if (t >= tmax) {
            if (out_emission) *out_emission = Le_acc;
            return false;
        }

        const Vec3 pos = ray.at(t);
        const float density = volumeRegion.sample_density(pos);
        const Vec3 sigma_t_local = medium.sigma_t * density;

        // Accumulate volumetric emission using the tracked null-collision points.
        if (accumulate_Le && density > 0.0f) {
            const float temperature = volumeRegion.sample_temperature(pos, density);
            const float flame = volumeRegion.sample_flame(pos, density);
            const Vec3 Le = volumeRegion.sample_emission(density, temperature, flame);
            Le_acc = Le_acc + Tr_run * (Le * inv_sigma_maj);
        }

        // Update running transmittance via ratio tracking
        Tr_run = Tr_run * make_vec3(
            fmaxf(0.0f, 1.0f - sigma_t_local.x * inv_sigma_maj),
            fmaxf(0.0f, 1.0f - sigma_t_local.y * inv_sigma_maj),
            fmaxf(0.0f, 1.0f - sigma_t_local.z * inv_sigma_maj));

        const float accept = fminf(1.0f, fmaxf(0.0f, spectrum_channel(sigma_t_local, channel) * inv_sigma_maj));
        if (rng_next(rng_state) < accept) {
            t_scatter = t;
            if (out_emission) *out_emission = Le_acc;
            return true;
        }
    }
}

// -----------------------------------------------------------------------
// TraceRayIterative
//
// Volume path tracing with per-channel homogeneous media:
//
//   Per bounce:
//     1. Check if current ray origin is inside a volume region.
//     2. If inside:
//          sample a free path in one RGB extinction channel
//          if t < t_surface: VOLUME SCATTER
//            throughput *= Tr * sigma_s / pdf_t
//            sample new dir from HG phase function
//            continue to next depth
//          else if exiting the AABB volume first:
//            apply Tr / pdf_t up to the boundary and keep marching
//          else: SURFACE HIT
//            apply Tr / pdf_t up to the surface and proceed with shading
//     3. If not inside a volume: normal surface shading only.
// -----------------------------------------------------------------------
HYBRID_FUNC inline Vec3 TraceRayIterative(
    const Ray& primaryRay,
    const int maxDepth,
    const Vec3 missColor,
    const int numTriangles,
    const BVHNode* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    const Triangle* __restrict__ triangles,
    const int32_t* __restrict__ triObjectIds,
    const Material* __restrict__ objectMaterials,
    const int numObjectMaterials,
    const Light* __restrict__ lights,
    const int numLights,
    const EmissiveTriInfo* __restrict__ emissiveTris,
    const float* __restrict__ emissiveCDF,
    const int numEmissiveTris,
    const float totalEmissiveArea,
    unsigned int rng_state = 42u,
    bool diffuse_bounce = true,
    int nee_mode = 2,
    const HomogeneousMedium* __restrict__ objectMedia = nullptr,
    int numObjectMedia = 0,
    const TextureData* __restrict__ textures = nullptr,
    int numTextures = 0,
    const VolumeRegionGPU* __restrict__ volumeRegions = nullptr,
    int numVolumeRegions = 0,
    const HDRTextureData* __restrict__ hdri = nullptr)
{
    if (maxDepth <= 0) return make_vec3(0.0f, 0.0f, 0.0f);

    Ray  ray        = primaryRay;
    Vec3 radiance   = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3 throughput = make_vec3(1.0f, 1.0f, 1.0f);
    float prev_pdf   = 0.0f;
    bool  prev_delta = false;
    Vec3  prev_N     = make_vec3(0.0f, 0.0f, 0.0f);
    bool  prev_medium_event = false;
    Vec3  prev_scatter_pos  = make_vec3(0.0f, 0.0f, 0.0f);

    for (int depth = 0; depth < maxDepth; ++depth) {

        // ----------------------------------------------------------------
        // 1. Determine if the current ray origin is inside a volume region.
        //    We re-check every bounce because the ray may have scattered out.
        // ----------------------------------------------------------------
        const int activeVolumeIdx =
            findVolumeIndexAtPoint(ray.origin(), volumeRegions, numVolumeRegions);
        HomogeneousMedium activeMedium =
            findVolumeAtPoint(ray.origin(), volumeRegions, numVolumeRegions);
        const VolumeRegionGPU* activeVolume =
            (activeVolumeIdx >= 0 && volumeRegions != nullptr) ? &volumeRegions[activeVolumeIdx] : nullptr;
        float t_volume_exit = 1e30f;
        if (activeVolumeIdx >= 0) {
            float t_enter = 0.0f;
            if (!activeVolume->ray_interval(ray, t_enter, t_volume_exit)) {
                activeMedium.enabled = false;
                t_volume_exit = 1e30f;
            }
        } else {
            float t_volume_enter = 1e30f;
            float t_next_volume_exit = 1e30f;
            const int nextVolumeIdx = findNextVolumeAlongRay(
                ray, volumeRegions, numVolumeRegions, t_volume_enter, t_next_volume_exit);
            if (nextVolumeIdx >= 0) {
                HitRecord preHit;
                SearchBVH(numTriangles, ray, nodes, aabbs, triangles, preHit);
                const float t_surf_pre = preHit.hit ? preHit.t : 1e30f;
                if (t_volume_enter + RT_EPS < t_surf_pre) {
                    ray = Ray(ray.at(t_volume_enter + RT_EPS), ray.direction());
                    --depth;
                    continue;
                }
            }
        }

        // ----------------------------------------------------------------
        // 2. BVH traversal to find the nearest surface hit
        // ----------------------------------------------------------------
        HitRecord hitRecord;
        SearchBVH(numTriangles, ray, nodes, aabbs, triangles, hitRecord);

        const float t_surf = hitRecord.hit ? hitRecord.t : 1e30f;

        // ----------------------------------------------------------------
        // 3. Volume scatter decision  [Lecture 14, slides 54-59]
        // ----------------------------------------------------------------
        if (activeMedium.has_extinction()) {
            const float t_medium_limit = fminf(t_surf, t_volume_exit);

            if (t_medium_limit > RT_EPS) {
                const int channel = (int)fminf(2.0f, floorf(rng_next(rng_state) * 3.0f));
                float t_vol = 0.0f;
                bool sampled_scatter = false;

                Vec3 vol_emission = make_vec3(0.0f, 0.0f, 0.0f);
                if (activeVolume != nullptr && activeVolume->has_density_grid()) {
                    sampled_scatter = sampleHeterogeneousScatter(
                        ray, t_medium_limit, *activeVolume, activeMedium, channel, rng_state, t_vol, &vol_emission);
                } else {
                    const float xi_t  = rng_next(rng_state);
                    t_vol = activeMedium.sampleFreePath(xi_t, channel);
                    sampled_scatter = (t_vol < t_medium_limit);
                }

                // Add volume emission accumulated during delta tracking
                if (vol_emission.x > 0.0f || vol_emission.y > 0.0f || vol_emission.z > 0.0f) {
                    radiance = radiance + throughput * vol_emission;
                }

                if (sampled_scatter) {
                    const Vec3 scatter_pos = ray.origin() + ray.direction() * t_vol;
                    Vec3 Tr = make_vec3(1.0f, 1.0f, 1.0f);
                    Vec3 sigma_s_event = activeMedium.sigma_s;
                    Vec3 sigma_t_event = activeMedium.sigma_t;

                    if (activeVolume != nullptr && activeVolume->has_density_grid()) {
                        Tr = estimateTransmittance(ray, t_vol, activeVolume, activeMedium, rng_state);
                        const float density = activeVolume->sample_density(scatter_pos);
                        sigma_s_event = activeMedium.sigma_s * density;
                        sigma_t_event = activeMedium.sigma_t * density;
                    } else {
                        Tr = activeMedium.transmittance(t_vol);
                    }

                    const float Tr_avg = fmaxf(spectrum_average(Tr), 1e-8f);
                    const float sigma_t_avg = fmaxf(spectrum_average(sigma_t_event), 1e-8f);
                    throughput = throughput * (Tr * (1.0f / Tr_avg)) *
                                 (sigma_s_event * (1.0f / sigma_t_avg));

                    // ---- VOLUME NEE: direct illumination at scatter point ----
                    if (numEmissiveTris > 0 && nee_mode != 1) {
                        const float u_sel = rng_next(rng_state);
                        const int eidx = binary_search_cdf(emissiveCDF, numEmissiveTris, u_sel);
                        const EmissiveTriInfo& emi = emissiveTris[eidx];
                        const Triangle& eTri = triangles[emi.triangleIdx];

                        float u1 = rng_next(rng_state), u2 = rng_next(rng_state);
                        if (u1 + u2 > 1.0f) { u1 = 1.0f - u1; u2 = 1.0f - u2; }
                        const Vec3 lightPoint = eTri.v0*(1.0f-u1-u2) + eTri.v1*u1 + eTri.v2*u2;

                        Vec3 toLight = lightPoint - scatter_pos;
                        const float r2 = dot(toLight, toLight);
                        if (r2 > 1e-8f) {
                            const float r_dist  = sqrtf(r2);
                            const Vec3 wi_light = toLight * (1.0f / r_dist);
                            const float cosLight = fabsf(dot(emi.normal, -wi_light));

                            if (cosLight > 1e-6f) {
                                Ray shadowRay(scatter_pos + wi_light * RT_EPS, wi_light);
                                HitRecord shadowHit{}; shadowHit.hit = false;
                                SearchBVH(numTriangles, shadowRay, nodes, aabbs, triangles, shadowHit);

                                if (!shadowHit.hit || shadowHit.t >= r_dist - RT_EPS) {
                                    const Vec3 wo_vol = make_vec3(-ray.direction().x,
                                                                   -ray.direction().y,
                                                                   -ray.direction().z);
                                    const float cos_theta_l = dot(normalize(wo_vol), wi_light);
                                    const float phase_l = activeMedium.phaseHG(cos_theta_l);

                                    const float G = cosLight / r2;
                                    const float pdf_area_sa = (G > 1e-10f)
                                        ? (1.0f / totalEmissiveArea) / G : 0.0f;
                                    const float light_medium_dist = (activeVolume != nullptr)
                                        ? activeVolume->segment_length(shadowRay, r_dist)
                                        : 0.0f;
                                    const Vec3 Tr_light = (activeVolume != nullptr && activeVolume->has_density_grid())
                                        ? estimateTransmittance(shadowRay, light_medium_dist, activeVolume, activeMedium, rng_state)
                                        : activeMedium.transmittance(light_medium_dist);
                                    const float w_light =
                                        (nee_mode == 0) ? 1.0f : power_heuristic(pdf_area_sa, phase_l);

                                    if (pdf_area_sa > 1e-10f) {
                                        radiance = radiance + throughput *
                                                   (emi.emission * phase_l * Tr_light * (w_light / pdf_area_sa));
                                    }
                                }
                            }
                        }
                    }
                    // ---- END VOLUME NEE ----

                    const float xi1 = rng_next(rng_state);
                    const float xi2 = rng_next(rng_state);
                    const Vec3 wi_in = make_vec3(-ray.direction().x,
                                                 -ray.direction().y,
                                                 -ray.direction().z);
                    const Vec3 new_dir = activeMedium.samplePhaseHG(wi_in, xi1, xi2);

                    ray = Ray(scatter_pos, new_dir);
                    prev_pdf   = activeMedium.phaseHG(dot(normalize(wi_in), normalize(new_dir)));
                    prev_delta = false;
                    prev_medium_event = true;
                    prev_scatter_pos = scatter_pos;

                    const float p_survive = fminf(luminance(throughput), 0.95f);
                    if (p_survive < 1e-4f || rng_next(rng_state) > p_survive) break;
                    throughput = throughput * (1.0f / p_survive);

                    continue;
                }

                Vec3 Tr = make_vec3(1.0f, 1.0f, 1.0f);
                if (activeVolume != nullptr && activeVolume->has_density_grid()) {
                    Tr = estimateTransmittance(ray, t_medium_limit, activeVolume, activeMedium, rng_state);
                    const float pdf_t = fmaxf(spectrum_average(Tr), 1e-8f);
                    if (pdf_t <= 1e-10f) break;
                    throughput = throughput * (Tr * (1.0f / pdf_t));
                } else {
                    Tr = activeMedium.transmittance(t_medium_limit);
                    const float pdf_t = spectrum_average(Tr);
                    if (pdf_t <= 1e-10f) break;
                    throughput = throughput * (Tr * (1.0f / pdf_t));
                }
            }

            if (t_volume_exit + RT_EPS < t_surf) {
                const float advance = fmaxf(t_volume_exit, 0.0f) + RT_EPS;
                ray = Ray(ray.at(advance), ray.direction());
                --depth;
                continue;
            }
        }

        // ----------------------------------------------------------------
        // 4. No hit → sky / miss color
        // ----------------------------------------------------------------
        if (!hitRecord.hit) {
            Vec3 sky = (hdri && hdri->width > 0)
                       ? sampleHDRI(*hdri, ray.direction())
                       : missColor;
            radiance = radiance + throughput * sky;
            break;
        }

        assignMaterialToHit(hitRecord, numTriangles, triObjectIds,
                            objectMaterials, numObjectMaterials,
                            textures, numTextures);

        // ----------------------------------------------------------------
        // 4b. Alpha cutout: transparent texel → skip this hit and continue
        // ----------------------------------------------------------------
        if (hitRecord.alpha_masked) {
            ray = Ray(hitRecord.p + ray.direction() * RT_EPS, ray.direction());
            --depth;  // don't count as a bounce
            continue;
        }

        // ----------------------------------------------------------------
        // 5. Emissive surface
        // ----------------------------------------------------------------
        const Vec3 Le = hitRecord.mat.emission;
        if (Le.x > 0.0f || Le.y > 0.0f || Le.z > 0.0f) {
            if (depth == 0 || prev_delta) {
                radiance = radiance + throughput * Le;
            } else if (prev_medium_event) {
                if (nee_mode == 1) {
                    radiance = radiance + throughput * Le;
                } else if (nee_mode == 2 && numEmissiveTris > 0 && prev_pdf > 1e-6f) {
                    Vec3 hitGeomN = normalize(cross(
                        triangles[hitRecord.triangleIdx].v1 - triangles[hitRecord.triangleIdx].v0,
                        triangles[hitRecord.triangleIdx].v2 - triangles[hitRecord.triangleIdx].v0));
                    const Vec3 to_light = hitRecord.p - prev_scatter_pos;
                    const float dist2 = dot(to_light, to_light);
                    if (dist2 > 1e-10f) {
                        const Vec3 wi_prev = to_light * (1.0f / sqrtf(dist2));
                        const float cosLight = fabsf(dot(hitGeomN, -wi_prev));
                        const float G = cosLight / dist2;
                        const float pdf_area_sa =
                            (G > 1e-10f) ? (1.0f / totalEmissiveArea) / G : 0.0f;
                        const float w_phase = power_heuristic(prev_pdf, pdf_area_sa);
                        radiance = radiance + throughput * Le * w_phase;
                    }
                }
            } else if (nee_mode == 1) {
                radiance = radiance + throughput * Le;
            } else if (nee_mode == 2 && numEmissiveTris > 0 && prev_pdf > 1e-6f) {
                Vec3 hitGeomN = normalize(cross(
                    triangles[hitRecord.triangleIdx].v1 - triangles[hitRecord.triangleIdx].v0,
                    triangles[hitRecord.triangleIdx].v2 - triangles[hitRecord.triangleIdx].v0));
                float cosLight = fabsf(dot(hitGeomN, -unit_vector(ray.direction())));
                float dist2    = hitRecord.t * hitRecord.t;
                float G        = (dist2 > 1e-10f) ? cosLight / dist2 : 0.0f;
                float pdf_area_sa = (G > 1e-10f) ? (1.0f / totalEmissiveArea) / G : 0.0f;
                float NdotWi_prev = fmaxf(dot(prev_N, unit_vector(ray.direction())), 0.0f);
                float pdf_cos     = NdotWi_prev * 0.31830988618f;
                float pdf_nee     = (pdf_area_sa + pdf_cos + prev_pdf) / 3.0f;
                float p2_brdf     = prev_pdf * prev_pdf;
                float p2_nee      = pdf_nee  * pdf_nee;
                float w_brdf = (p2_brdf + p2_nee > 0.0f) ? p2_brdf / (p2_brdf + p2_nee) : 0.0f;
                radiance = radiance + throughput * Le * w_brdf;
            }
        }

        // ----------------------------------------------------------------
        // 6. Direct lighting (point lights)
        // ----------------------------------------------------------------
        Vec3 direct = ShadeDirect(ray, hitRecord, lights, numLights,
                                  numTriangles, nodes, aabbs, triangles,
                                  objectMedia, numObjectMedia, triObjectIds);
        radiance = radiance + throughput * direct;

        const Vec3  N  = normalize(hitRecord.normal);
        const Vec3  Vo = unit_vector(-ray.direction());
        prev_N = N;

        // ----------------------------------------------------------------
        // 7. NEE (next event estimation for emissive triangles)
        // ----------------------------------------------------------------
        if (numEmissiveTris > 0 && nee_mode != 1) {
            int strategy;
            if (nee_mode == 0) {
                strategy = 0;
            } else {
                const float strategy_u = rng_next(rng_state);
                strategy = (strategy_u < 0.333333f) ? 0
                         : (strategy_u < 0.666667f) ? 1 : 2;
            }

            Vec3  wi_nee       = make_vec3(0,0,0);
            Vec3  Le_nee       = make_vec3(0,0,0);
            float dist_nee     = 0.0f;
            float cosLight_nee = 0.0f;
            float NdotL_nee    = 0.0f;
            bool  nee_valid    = false;

            if (strategy == 0) {
                const float u_sel = rng_next(rng_state);
                const int eidx = binary_search_cdf(emissiveCDF, numEmissiveTris, u_sel);
                const EmissiveTriInfo& emi = emissiveTris[eidx];
                const Triangle& eTri = triangles[emi.triangleIdx];
                float u1 = rng_next(rng_state), u2 = rng_next(rng_state);
                if (u1 + u2 > 1.0f) { u1 = 1.0f - u1; u2 = 1.0f - u2; }
                const Vec3 lightPoint = eTri.v0 * (1.0f - u1 - u2) + eTri.v1 * u1 + eTri.v2 * u2;
                Vec3 toLight = lightPoint - hitRecord.p;
                const float r2 = dot(toLight, toLight);
                if (r2 > 1e-10f) {
                    const float r_dist = sqrtf(r2);
                    wi_nee       = toLight * (1.0f / r_dist);
                    NdotL_nee    = fmaxf(dot(N, wi_nee), 0.0f);
                    cosLight_nee = fabsf(dot(emi.normal, -wi_nee));
                    if (NdotL_nee > 0.0f && cosLight_nee > 0.0f) {
                        Ray shadowRay(hitRecord.p + N * RT_EPS, wi_nee);
                        HitRecord shadowHit{}; shadowHit.hit = false;
                        SearchBVH(numTriangles, shadowRay, nodes, aabbs, triangles, shadowHit);
                        if (!shadowHit.hit || shadowHit.t >= r_dist - RT_EPS) {
                            Le_nee    = emi.emission;
                            dist_nee  = r_dist;
                            nee_valid = true;
                        }
                    }
                }
            } else {
                if (strategy == 1)
                    wi_nee = cosine_hemisphere_sample(N, rng_state);
                else {
                    float dummy_pdf;
                    wi_nee = SampleBRDF(hitRecord, Vo, rng_state, dummy_pdf, diffuse_bounce);
                }
                NdotL_nee = fmaxf(dot(N, wi_nee), 0.0f);
                if (NdotL_nee > 0.0f) {
                    Ray neeRay(hitRecord.p + N * RT_EPS, wi_nee);
                    HitRecord neeHit; neeHit.hit = false;
                    SearchBVH(numTriangles, neeRay, nodes, aabbs, triangles, neeHit);
                    if (neeHit.hit) {
                        assignMaterialToHit(neeHit, numTriangles, triObjectIds,
                                            objectMaterials, numObjectMaterials,
                                            textures, numTextures);
                        const Vec3 Le_hit = neeHit.mat.emission;
                        if (Le_hit.x > 0.0f || Le_hit.y > 0.0f || Le_hit.z > 0.0f) {
                            Vec3 geomN = normalize(cross(
                                triangles[neeHit.triangleIdx].v1 - triangles[neeHit.triangleIdx].v0,
                                triangles[neeHit.triangleIdx].v2 - triangles[neeHit.triangleIdx].v0));
                            cosLight_nee = fabsf(dot(geomN, -wi_nee));
                            dist_nee     = neeHit.t;
                            Le_nee       = Le_hit;
                            nee_valid    = true;
                        }
                    }
                }
            }

            if (nee_valid && dist_nee > 1e-6f && cosLight_nee > 1e-6f) {
                const float r2          = dist_nee * dist_nee;
                const float G           = cosLight_nee / r2;
                const float pdf_area_sa = (1.0f / totalEmissiveArea) / G;
                const Vec3  f_nee       = EvaluateBRDF(hitRecord, Vo, wi_nee);

                // Apply medium transmittance along NEE shadow ray if in volume
                Vec3 Tr_nee = make_vec3(1.0f, 1.0f, 1.0f);
                if (activeMedium.has_extinction() && activeVolume != nullptr) {
                    Ray mediumShadowRay(hitRecord.p + N * RT_EPS, wi_nee);
                    const float medium_dist = activeVolume->segment_length(mediumShadowRay, dist_nee);
                    Tr_nee = activeVolume->has_density_grid()
                        ? estimateTransmittance(mediumShadowRay, medium_dist, activeVolume, activeMedium, rng_state)
                        : activeMedium.transmittance(medium_dist);
                }

                if (nee_mode == 0) {
                    if (pdf_area_sa > 1e-10f) {
                        radiance = radiance + throughput *
                                   (Le_nee * f_nee * (NdotL_nee / pdf_area_sa) * Tr_nee);
                    }
                } else {
                    const float pdf_cos      = NdotL_nee * 0.31830988618f;
                    const float pdf_brdf     = BRDFSamplingPdf(hitRecord, Vo, wi_nee, diffuse_bounce);
                    const float pdf_combined = (pdf_area_sa + pdf_cos + pdf_brdf) / 3.0f;
                    if (pdf_combined > 1e-10f) {
                        const float p2_nee  = pdf_combined * pdf_combined;
                        const float p2_brdf = pdf_brdf     * pdf_brdf;
                        const float w_nee   = (p2_nee + p2_brdf > 0.0f)
                                            ? p2_nee / (p2_nee + p2_brdf) : 0.0f;
                        radiance = radiance + throughput *
                                   (Le_nee * f_nee * (NdotL_nee / pdf_combined) * Tr_nee * w_nee);
                    }
                }
            }
        }

        // ----------------------------------------------------------------
        // 8. Sample next bounce direction (BRDF, mirror, or glass)
        // ----------------------------------------------------------------
        const float kd  = hitRecord.mat.kd;
        const float ks  = hitRecord.mat.ks;
        const float kr  = hitRecord.mat.kr;
        const float ior = hitRecord.mat.ior;

        if (kd > 1e-6f || ks > 1e-6f) {
            // ---- Diffuse / glossy BRDF ----
            float pdf;
            Vec3 wi = SampleBRDF(hitRecord, Vo, rng_state, pdf, diffuse_bounce);
            if (pdf < 1e-6f || dot(wi, N) < 0.0f) break;
            ray = Ray(hitRecord.p + N * RT_EPS, wi);
            const float NdotWi = fmaxf(dot(N, wi), 0.0f);
            const Vec3  f      = EvaluateBRDF(hitRecord, Vo, wi);
            throughput = throughput * (f * (NdotWi / pdf));
            prev_pdf   = pdf;
            prev_delta = false;
            prev_medium_event = false;
        } else if (ior > 1.0f) {
            // ---- Dielectric glass (Snell's law + Schlick Fresnel) ----
            //
            // hitRecord.normal is always oriented toward the incident side (i.e. toward
            // the ray origin) for both entry and exit — this matches the convention that
            // refract_dir and the Schlick formula expect for N.
            const bool  entering  = hitRecord.front_face;
            const float eta       = entering ? (1.0f / ior) : ior;  // n_incident / n_transmitted
            const Vec3  rayDir    = unit_vector(ray.direction());

            // cos(θᵢ): always positive since N points toward incident side
            const float cos_i     = fminf(-dot(rayDir, N), 1.0f);

            // Schlick Fresnel (r0 is symmetric in n1<->n2, so ior suffices)
            const float F         = schlick(cos_i, ior);

            const Vec3  refractDir = refract_dir(rayDir, N, eta);
            const bool  tir        = (refractDir.x == 0.0f && refractDir.y == 0.0f
                                      && refractDir.z == 0.0f);

            if (!tir && rng_next(rng_state) > F) {
                // Transmit: offset past the surface into the transmitted side (opposite N)
                ray = Ray(hitRecord.p - N * RT_EPS, refractDir);
            } else {
                // Fresnel reflection or total internal reflection: stay on incident side
                const Vec3 reflDir = reflect_dir(rayDir, N);
                ray = Ray(hitRecord.p + N * RT_EPS, reflDir);
            }
            throughput = throughput * hitRecord.mat.specularColor;
            prev_pdf   = 0.0f;
            prev_delta = true;
            prev_medium_event = false;
        } else {
            // ---- Perfect mirror ----
            const Vec3 reflDir = reflect_dir(unit_vector(ray.direction()), N);
            ray = Ray(hitRecord.p + N * RT_EPS, reflDir);
            throughput = throughput * (hitRecord.mat.specularColor * kr);
            prev_pdf   = 0.0f;
            prev_delta = true;
            prev_medium_event = false;
        }

        // Russian roulette
        const float p_survive = fminf(luminance(throughput), 0.95f);
        if (p_survive < 1e-4f || rng_next(rng_state) > p_survive) break;
        throughput = throughput * (1.0f / p_survive);
    }

    return radiance;
}


HYBRID_FUNC inline void SearchBVH(
    const int numTriangles,
    const Ray& ray,
    const BVHNode* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    const Triangle* __restrict__ triangles,
    HitRecord& hitRecord)
{
    constexpr float kRayTMin = 1e-4f;
    const float tmin = kRayTMin;
    float bestT = FLT_MAX;
    HitRecord bestHit;
    bestHit.triangleIdx = -1;
    bestHit.hit        = false;
    bestHit.t          = -1.0;
    bestHit.p          = make_vec3(0,0,0);
    bestHit.normal     = make_vec3(0,0,0);
    bestHit.front_face = false;
    bestHit.mat        = Material();
    bestHit.uv         = make_vec2(0.0f, 0.0f);

    constexpr int STACK_CAPACITY = 512;
    std::uint32_t stack[STACK_CAPACITY];
    std::uint32_t* stack_ptr = stack;
    bool stackOverflow = false;
    *stack_ptr++ = 0;

    while (stack_ptr > stack) {
        const std::uint32_t nodeIdx = *--stack_ptr;
        if (!intersectAABB(ray, aabbs[nodeIdx], tmin, bestT)) continue;

        const BVHNode       node    = nodes[nodeIdx];
        const std::uint32_t obj_idx = node.object_idx;

        if (obj_idx != 0xFFFFFFFF) {
            if (obj_idx < static_cast<std::uint32_t>(numTriangles)) {
                HitRecord rec = intersectTriangle(ray, triangles[obj_idx], tmin, bestT);
                if (rec.hit) {
                    rec.triangleIdx = static_cast<int>(obj_idx);
                    bestT   = rec.t;
                    bestHit = rec;
                }
            }
            continue;
        }

        const std::uint32_t left_idx  = node.left_idx;
        const std::uint32_t right_idx = node.right_idx;

        if (left_idx != 0xFFFFFFFF) {
            if (intersectAABB(ray, aabbs[left_idx], tmin, bestT)) {
                if (stack_ptr - stack < STACK_CAPACITY) *stack_ptr++ = left_idx;
                else stackOverflow = true;
            }
        }
        if (right_idx != 0xFFFFFFFF) {
            if (intersectAABB(ray, aabbs[right_idx], tmin, bestT)) {
                if (stack_ptr - stack < STACK_CAPACITY) *stack_ptr++ = right_idx;
                else stackOverflow = true;
            }
        }
    }

    if (stackOverflow) {
        for (int i = 0; i < numTriangles; ++i) {
            HitRecord rec = intersectTriangle(ray, triangles[i], tmin, bestT);
            if (rec.hit) {
                rec.triangleIdx = i;
                bestT   = rec.t;
                bestHit = rec;
            }
        }
    }

    hitRecord = bestHit;
}
