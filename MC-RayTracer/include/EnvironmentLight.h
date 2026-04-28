#pragma once

#include <array>
#include <string>
#include <vector>

#include "ILight.h"
#include "environment.h"

struct EnvironmentLightParams {
    bool enabled = true;
    std::string hdri_path;
    float intensity_scale = 1.0f;
    Vec3f tint = make_vec3(1.0f, 1.0f, 1.0f);
    std::array<int, 2> importance_sampling_resolution{1024, 512};
    Vec3f rotation_euler_deg = make_vec3(0.0f, 0.0f, 0.0f);
};

class EnvironmentLight final : public ILight {
public:
    explicit EnvironmentLight(const EnvironmentLightParams& params);

    Vec3f Sample(const Vec3f& normal, const Vec2f& u, float& pdf_out) const override;
    float Pdf(const Vec3f& wi) const override;
    Vec3f Eval(const Vec3f& wi) const override;
    bool IsDelta() const override;

private:
    struct ImportanceDistribution {
        int width = 0;
        int height = 0;
        bool valid = false;
        std::vector<float> pmf;         // width * height
        std::vector<float> row_cdf;     // height + 1
        std::vector<float> col_cdf;     // height * (width + 1)
    };

    bool HasMap() const;
    bool LoadLatLongMap(const std::string& path);
    void BuildImportanceDistribution();
    void BuildRotationMatrices();

    Vec3f EnvToWorld(const Vec3f& d) const;
    Vec3f WorldToEnv(const Vec3f& d) const;
    Vec3f EvalLatLong(const Vec3f& wi_env) const;

private:
    EnvironmentLightParams params_{};
    Environment env_{};

    std::vector<unsigned char> map_pixels_;
    TextureData map_view_{};

    ImportanceDistribution dist_{};

    // Row-major 3x3
    std::array<float, 9> env_to_world_{{1,0,0, 0,1,0, 0,0,1}};
    std::array<float, 9> world_to_env_{{1,0,0, 0,1,0, 0,0,1}};
};

