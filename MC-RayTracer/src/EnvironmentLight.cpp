#include "EnvironmentLight.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;
constexpr float kInvPi = 0.31830988618379067154f;
constexpr float kInvTwoPi = 0.15915494309189533577f;

inline float clamp01(float x) {
    return env_clampf(x, 0.0f, 1.0f);
}

inline float luminance709(const Vec3f& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

inline std::array<float, 9> mat_mul(const std::array<float, 9>& a, const std::array<float, 9>& b) {
    std::array<float, 9> r{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r[i * 3 + j] =
                a[i * 3 + 0] * b[0 * 3 + j] +
                a[i * 3 + 1] * b[1 * 3 + j] +
                a[i * 3 + 2] * b[2 * 3 + j];
        }
    }
    return r;
}

inline std::array<float, 9> mat_transpose(const std::array<float, 9>& m) {
    return {m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]};
}

inline Vec3f mat_mul_vec(const std::array<float, 9>& m, const Vec3f& v) {
    return make_vec3(
        m[0] * v.x + m[1] * v.y + m[2] * v.z,
        m[3] * v.x + m[4] * v.y + m[5] * v.z,
        m[6] * v.x + m[7] * v.y + m[8] * v.z);
}

inline Vec3f dir_from_uv(float u, float v) {
    const float phi = (u - 0.5f) * kTwoPi;
    const float theta = clamp01(v) * kPi;
    const float sin_theta = sinf(theta);
    return env_safe_normalize(
        make_vec3(sin_theta * sinf(phi), sin_theta * cosf(phi), cosf(theta)),
        make_vec3(0.0f, 1.0f, 0.0f));
}

inline void uv_from_dir(const Vec3f& dir, float& u, float& v) {
    const Vec3f d = env_safe_normalize(dir, make_vec3(0.0f, 1.0f, 0.0f));
    u = env_fract(atan2f(d.x, d.y) * kInvTwoPi + 0.5f);
    v = acosf(env_clampf(d.z, -1.0f, 1.0f)) * kInvPi;
}

inline int sample_cdf(const float* cdf, int n, float xi, float& remapped_u) {
    const float u = env_clampf(xi, 0.0f, 0.99999994f);
    const float* it = std::upper_bound(cdf + 1, cdf + n + 1, u);
    int idx = static_cast<int>(it - cdf) - 1;
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;

    const float c0 = cdf[idx];
    const float c1 = cdf[idx + 1];
    remapped_u = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.5f;
    return idx;
}
} // namespace

EnvironmentLight::EnvironmentLight(const EnvironmentLightParams& params)
    : params_(params) {
    env_.enabled = params_.enabled;
    env_.map_intensity = params_.intensity_scale;
    env_.tint = params_.tint;
    env_.map_rotation_deg = 0.0f; // handled by full Euler rotation below.
    BuildRotationMatrices();
    if (!params_.hdri_path.empty()) {
        LoadLatLongMap(params_.hdri_path);
    }
    BuildImportanceDistribution();
}

bool EnvironmentLight::HasMap() const {
    return map_view_.data != nullptr && map_view_.width > 0 && map_view_.height > 0;
}

bool EnvironmentLight::LoadLatLongMap(const std::string& path) {
    int w = 0, h = 0, c = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!pixels || w <= 0 || h <= 0) {
        std::fprintf(stderr, "[EnvironmentLight] Failed to load hdri_path='%s'\n", path.c_str());
        return false;
    }

    const size_t count = static_cast<size_t>(w) * static_cast<size_t>(h) * 3u;
    map_pixels_.assign(pixels, pixels + count);
    stbi_image_free(pixels);

    map_view_.width = w;
    map_view_.height = h;
    map_view_.channels = 3;
    map_view_.data = map_pixels_.data();
    env_.latlong_map = &map_view_;
    return true;
}

void EnvironmentLight::BuildRotationMatrices() {
    constexpr float kDegToRad = 0.01745329251994329577f;
    const float rx = params_.rotation_euler_deg.x * kDegToRad;
    const float ry = params_.rotation_euler_deg.y * kDegToRad;
    const float rz = params_.rotation_euler_deg.z * kDegToRad;

    const float cx = cosf(rx), sx = sinf(rx);
    const float cy = cosf(ry), sy = sinf(ry);
    const float cz = cosf(rz), sz = sinf(rz);

    const std::array<float, 9> Rx{1,0,0, 0,cx,-sx, 0,sx,cx};
    const std::array<float, 9> Ry{cy,0,sy, 0,1,0, -sy,0,cy};
    const std::array<float, 9> Rz{cz,-sz,0, sz,cz,0, 0,0,1};

    env_to_world_ = mat_mul(Rz, mat_mul(Ry, Rx));
    world_to_env_ = mat_transpose(env_to_world_);
}

Vec3f EnvironmentLight::EnvToWorld(const Vec3f& d) const {
    return env_safe_normalize(mat_mul_vec(env_to_world_, d), make_vec3(0.0f, 1.0f, 0.0f));
}

Vec3f EnvironmentLight::WorldToEnv(const Vec3f& d) const {
    return env_safe_normalize(mat_mul_vec(world_to_env_, d), make_vec3(0.0f, 1.0f, 0.0f));
}

Vec3f EnvironmentLight::EvalLatLong(const Vec3f& wi_env) const {
    if (!HasMap() || !env_.enabled) return make_vec3(0.0f, 0.0f, 0.0f);
    return EvaluateEnvironment(wi_env, env_);
}

void EnvironmentLight::BuildImportanceDistribution() {
    dist_ = ImportanceDistribution{};
    if (!HasMap() || !env_.enabled) return;

    dist_.width = map_view_.width;
    dist_.height = map_view_.height;
    const int w = dist_.width;
    const int h = dist_.height;

    dist_.pmf.assign(static_cast<size_t>(w) * static_cast<size_t>(h), 0.0f);
    dist_.row_cdf.assign(static_cast<size_t>(h) + 1u, 0.0f);
    dist_.col_cdf.assign(static_cast<size_t>(h) * static_cast<size_t>(w + 1), 0.0f);

    for (int y = 0; y < h; ++y) {
        const float theta = (static_cast<float>(y) + 0.5f) * (kPi / static_cast<float>(h));
        const float sin_theta = fmaxf(sinf(theta), 1e-7f);

        const int row_off = y * (w + 1);
        dist_.col_cdf[static_cast<size_t>(row_off)] = 0.0f;
        float row_sum = 0.0f;

        for (int x = 0; x < w; ++x) {
            Vec3f c = env_sample_latlong_texel(&map_view_, x, y);
            c = c * params_.tint;
            const float weight = fmaxf(luminance709(c), 0.0f) * sin_theta;
            dist_.pmf[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = weight;
            row_sum += weight;
            dist_.col_cdf[static_cast<size_t>(row_off + x + 1)] = row_sum;
        }

        if (row_sum > 0.0f) {
            const float inv = 1.0f / row_sum;
            for (int i = 1; i <= w; ++i) {
                dist_.col_cdf[static_cast<size_t>(row_off + i)] *= inv;
            }
        } else {
            for (int i = 0; i <= w; ++i) {
                dist_.col_cdf[static_cast<size_t>(row_off + i)] = static_cast<float>(i) / static_cast<float>(w);
            }
        }

        dist_.row_cdf[static_cast<size_t>(y + 1)] = dist_.row_cdf[static_cast<size_t>(y)] + row_sum;
    }

    const float total = dist_.row_cdf[static_cast<size_t>(h)];
    if (total <= 0.0f) {
        const float uni = 1.0f / static_cast<float>(w * h);
        std::fill(dist_.pmf.begin(), dist_.pmf.end(), uni);
        for (int y = 0; y <= h; ++y) {
            dist_.row_cdf[static_cast<size_t>(y)] = static_cast<float>(y) / static_cast<float>(h);
        }
        dist_.valid = true;
        return;
    }

    const float inv_total = 1.0f / total;
    for (float& p : dist_.pmf) p *= inv_total;
    for (int y = 1; y <= h; ++y) dist_.row_cdf[static_cast<size_t>(y)] *= inv_total;
    dist_.row_cdf[0] = 0.0f;
    dist_.row_cdf[static_cast<size_t>(h)] = 1.0f;
    dist_.valid = true;
}

Vec3f EnvironmentLight::Sample(const Vec3f& normal, const Vec2f& u, float& pdf_out) const {
    (void)normal;
    pdf_out = 0.0f;

    if (!env_.enabled || !HasMap()) {
        return make_vec3(0.0f, 1.0f, 0.0f);
    }

    if (!dist_.valid) {
        // Uniform-sphere fallback.
        const float z = 1.0f - 2.0f * clamp01(u.y);
        const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
        const float phi = kTwoPi * clamp01(u.x);
        pdf_out = 1.0f / (4.0f * kPi);
        return make_vec3(r * cosf(phi), r * sinf(phi), z);
    }

    float uy = 0.0f;
    const int y = sample_cdf(dist_.row_cdf.data(), dist_.height, u.y, uy);
    float ux = 0.0f;
    const int row_off = y * (dist_.width + 1);
    const int x = sample_cdf(&dist_.col_cdf[static_cast<size_t>(row_off)], dist_.width, u.x, ux);

    const float u_map = (static_cast<float>(x) + ux) / static_cast<float>(dist_.width);
    const float v_map = (static_cast<float>(y) + uy) / static_cast<float>(dist_.height);
    const Vec3f wi_env = dir_from_uv(u_map, v_map);
    const Vec3f wi_world = EnvToWorld(wi_env);

    pdf_out = Pdf(wi_world);
    return wi_world;
}

float EnvironmentLight::Pdf(const Vec3f& wi) const {
    if (!env_.enabled || !HasMap()) return 0.0f;
    if (!dist_.valid) return 1.0f / (4.0f * kPi);

    const Vec3f wi_env = WorldToEnv(wi);
    float u = 0.0f;
    float v = 0.0f;
    uv_from_dir(wi_env, u, v);

    const int x = std::max(0, std::min(dist_.width - 1, static_cast<int>(u * dist_.width)));
    const int y = std::max(0, std::min(dist_.height - 1, static_cast<int>(v * dist_.height)));
    const float pmf = dist_.pmf[static_cast<size_t>(y) * static_cast<size_t>(dist_.width) + static_cast<size_t>(x)];

    const float theta = (static_cast<float>(y) + 0.5f) * (kPi / static_cast<float>(dist_.height));
    const float sin_theta = fmaxf(sinf(theta), 1e-7f);
    const float d_omega = (kTwoPi * kPi * sin_theta) /
                          static_cast<float>(dist_.width * dist_.height);
    return (d_omega > 0.0f) ? (pmf / d_omega) : 0.0f;
}

Vec3f EnvironmentLight::Eval(const Vec3f& wi) const {
    if (!env_.enabled || !HasMap()) return make_vec3(0.0f, 0.0f, 0.0f);
    return EvalLatLong(WorldToEnv(wi));
}

bool EnvironmentLight::IsDelta() const {
    return false;
}

