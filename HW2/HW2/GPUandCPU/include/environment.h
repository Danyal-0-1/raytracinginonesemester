#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <cmath>

#include "vec3.h"

enum class EnvironmentPreset : int {
    Custom = 0,
    Sunrise = 1,
    Midday = 2,
    Sunset = 3,
    NightMoon = 4,
    NightNoMoon = 5
};

struct Environment {
    bool enabled = false;
    EnvironmentPreset preset = EnvironmentPreset::Custom;

    Vec3 sun_dir = make_vec3(0.0f, 0.5f, 0.8660254f);
    float sun_intensity = 8.0f;
    Vec3 sun_tint = make_vec3(1.0f, 0.98f, 0.94f);

    float sky_intensity = 1.0f;
    float turbidity = 2.5f;
    Vec3 horizon_tint = make_vec3(1.0f, 0.58f, 0.28f);
    float warm_scatter = 1.0f;
    Vec3 tint = make_vec3(1.0f, 1.0f, 1.0f);

    Vec3 ground_color = make_vec3(0.56f, 0.56f, 0.58f);
    float ground_intensity = 0.30f;

    bool moon_enabled = true;
    Vec3 moon_dir = make_vec3(-0.55f, -0.65f, 0.52f);
    float moon_intensity = 0.15f;

    float star_intensity = 0.35f;
    float star_density = 1.0f;
    float star_threshold = 0.9975f;

    // Optional exposure style tone mapping for environment contribution only.
    float exposure = 1.0f;
};

HYBRID_FUNC inline float env_clampf(float x, float lo = 0.0f, float hi = 1.0f) {
    return fmaxf(lo, fminf(x, hi));
}

HYBRID_FUNC inline float env_lerp_f(float a, float b, float t) {
    return a + (b - a) * t;
}

HYBRID_FUNC inline Vec3 env_lerp(const Vec3& a, const Vec3& b, float t) {
    return make_vec3(env_lerp_f(a.x, b.x, t), env_lerp_f(a.y, b.y, t), env_lerp_f(a.z, b.z, t));
}

HYBRID_FUNC inline float env_smoothstep(float edge0, float edge1, float x) {
    const float t = env_clampf((x - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

HYBRID_FUNC inline float env_fract(float x) {
    return x - floorf(x);
}

HYBRID_FUNC inline float env_hash3(float x, float y, float z) {
    const float h = sinf(x * 127.1f + y * 311.7f + z * 74.7f) * 43758.5453123f;
    return env_fract(h);
}

HYBRID_FUNC inline Vec3 env_safe_normalize(const Vec3& v, const Vec3& fallback) {
    const float len2 = length_squared(v);
    if (len2 <= 1e-12f) return fallback;
    const float inv_len = 1.0f / sqrtf(len2);
    return v * inv_len;
}

HYBRID_FUNC inline Vec3 env_exp_tonemap(const Vec3& c, float exposure) {
    const float exp_scale = fmaxf(exposure, 1e-4f);
    return make_vec3(
        1.0f - expf(-c.x * exp_scale),
        1.0f - expf(-c.y * exp_scale),
        1.0f - expf(-c.z * exp_scale));
}

// Coordinate convention used everywhere in this renderer:
// - +Z is up.
// - elevation_deg: 0 at horizon, +90 at zenith.
// - azimuth_deg: 0 points +Y, 90 points +X.
HYBRID_FUNC inline Vec3 DirectionFromElevationAzimuthDeg(float elevation_deg, float azimuth_deg) {
    const float deg_to_rad = 0.01745329251994329577f;
    const float elev = elevation_deg * deg_to_rad;
    const float azim = azimuth_deg * deg_to_rad;
    const float ce = cosf(elev);
    const Vec3 d = make_vec3(ce * sinf(azim), ce * cosf(azim), sinf(elev));
    return env_safe_normalize(d, make_vec3(0.0f, -1.0f, 0.0f));
}

HYBRID_FUNC inline void ElevationAzimuthFromDirectionDeg(const Vec3& dir_in, float& elevation_deg, float& azimuth_deg) {
    const float rad_to_deg = 57.295779513082320876f;
    const Vec3 d = env_safe_normalize(dir_in, make_vec3(0.0f, 1.0f, 0.0f));
    elevation_deg = asinf(env_clampf(d.z, -1.0f, 1.0f)) * rad_to_deg;
    float az = atan2f(d.x, d.y) * rad_to_deg;
    if (az < 0.0f) az += 360.0f;
    azimuth_deg = az;
}

HYBRID_FUNC inline void ApplyEnvironmentPreset(Environment& env, EnvironmentPreset preset) {
    env.preset = preset;
    env.ground_color = make_vec3(0.56f, 0.56f, 0.58f);
    env.ground_intensity = 0.30f;
    env.moon_dir = DirectionFromElevationAzimuthDeg(25.0f, 220.0f);
    env.moon_enabled = true;
    env.moon_intensity = 0.0f;
    env.star_intensity = 0.0f;
    env.star_density = 1.0f;
    env.star_threshold = 0.9975f;
    env.tint = make_vec3(1.0f, 1.0f, 1.0f);
    env.exposure = 1.0f;

    switch (preset) {
        case EnvironmentPreset::Sunrise:
            env.sun_dir = DirectionFromElevationAzimuthDeg(15.0f, 70.0f);
            env.sun_intensity = 10.5f;
            env.sun_tint = make_vec3(1.0f, 0.73f, 0.82f);     // pinker/cooler than sunset
            env.sky_intensity = 1.20f;
            env.turbidity = 4.4f;
            env.horizon_tint = make_vec3(1.0f, 0.62f, 0.55f);
            env.warm_scatter = 1.05f;
            env.ground_intensity = 0.34f;
            env.exposure = 1.08f;
            break;
        case EnvironmentPreset::Midday:
            // High sun and camera-facing azimuth for a visible noon halo.
            env.sun_dir = DirectionFromElevationAzimuthDeg(62.0f, 25.0f);
            env.sun_intensity = 18.0f;
            env.sun_tint = make_vec3(1.0f, 0.99f, 0.95f);
            env.sky_intensity = 2.85f;                         // explicit midday energy boost
            env.turbidity = 1.9f;
            env.horizon_tint = make_vec3(1.0f, 0.90f, 0.72f);
            env.warm_scatter = 0.20f;
            env.ground_intensity = 0.42f;
            env.exposure = 1.20f;
            break;
        case EnvironmentPreset::Sunset:
            env.sun_dir = DirectionFromElevationAzimuthDeg(9.0f, 300.0f);
            env.sun_intensity = 9.5f;
            env.sun_tint = make_vec3(1.0f, 0.52f, 0.30f);     // deeper orange/red than sunrise
            env.sky_intensity = 1.10f;
            env.turbidity = 5.8f;
            env.horizon_tint = make_vec3(1.0f, 0.35f, 0.14f);
            env.warm_scatter = 1.50f;
            env.ground_intensity = 0.30f;
            env.exposure = 1.04f;
            break;
        case EnvironmentPreset::NightMoon:
            env.sun_dir = DirectionFromElevationAzimuthDeg(-18.0f, 330.0f);
            env.sun_intensity = 0.0f;
            env.sun_tint = make_vec3(1.0f, 1.0f, 1.0f);
            env.sky_intensity = 0.18f;
            env.turbidity = 2.0f;
            env.horizon_tint = make_vec3(0.2f, 0.25f, 0.35f);
            env.warm_scatter = 0.0f;
            env.ground_intensity = 0.085f;
            env.moon_enabled = true;
            env.moon_intensity = 0.38f;
            env.star_intensity = 0.42f;
            env.star_density = 1.10f;
            env.star_threshold = 0.9975f;
            env.exposure = 1.30f;
            break;
        case EnvironmentPreset::NightNoMoon:
            env.sun_dir = DirectionFromElevationAzimuthDeg(-18.0f, 330.0f);
            env.sun_intensity = 0.0f;
            env.sun_tint = make_vec3(1.0f, 1.0f, 1.0f);
            env.sky_intensity = 0.07f;
            env.turbidity = 1.9f;
            env.horizon_tint = make_vec3(0.1f, 0.12f, 0.16f);
            env.warm_scatter = 0.0f;
            env.ground_intensity = 0.040f;
            env.moon_enabled = false;
            env.moon_intensity = 0.0f;
            env.star_intensity = 1.25f;
            env.star_density = 1.55f;
            env.star_threshold = 0.9965f;
            env.exposure = 1.35f;
            break;
        case EnvironmentPreset::Custom:
        default:
            break;
    }
}

HYBRID_FUNC inline const char* EnvironmentPresetToString(EnvironmentPreset preset) {
    switch (preset) {
        case EnvironmentPreset::Sunrise: return "sunrise";
        case EnvironmentPreset::Midday: return "midday";
        case EnvironmentPreset::Sunset: return "sunset";
        case EnvironmentPreset::NightMoon: return "night_moon";
        case EnvironmentPreset::NightNoMoon: return "night_nomoon";
        case EnvironmentPreset::Custom:
        default: return "custom";
    }
}

HYBRID_FUNC inline Vec3 EvaluateSunSky(const Vec3& dir_in, const Environment& env) {
    if (!env.enabled) return make_vec3(0.0f, 0.0f, 0.0f);

    const Vec3 view_dir = env_safe_normalize(dir_in, make_vec3(0.0f, 1.0f, 0.0f));
    const Vec3 sun_dir = env_safe_normalize(env.sun_dir, make_vec3(0.0f, 1.0f, 0.0f));
    const Vec3 moon_dir = env_safe_normalize(env.moon_dir, make_vec3(-0.55f, -0.65f, 0.52f));

    const float up = view_dir.z;
    const float sky_up = env_clampf(up);
    const float sun_elev = sun_dir.z;
    const float day_factor = env_smoothstep(-0.18f, 0.03f, sun_elev);
    const float night_factor = 1.0f - day_factor;
    const float high_sun = env_smoothstep(0.707f, 0.98f, sun_elev);  // >45 deg elevation
    const float low_sun = 1.0f - env_smoothstep(0.20f, 0.75f, sun_elev);

    const float horizon = expf(-4.6f * sky_up);
    float haze = env_clampf((env.turbidity - 1.0f) / 10.0f);
    haze *= (1.0f - 0.35f * high_sun); // reduce haze when sun is high

    // Blue daylight baseline. These constants intentionally avoid a green cast.
    const Vec3 zenith_blue = make_vec3(0.20f, 0.45f, 1.00f) * (1.0f + 1.4f * high_sun);
    const Vec3 horizon_blue = make_vec3(0.90f, 0.95f, 1.00f) * (1.0f + 0.25f * high_sun + 0.45f * haze);
    Vec3 day_sky = env_lerp(zenith_blue, horizon_blue, horizon);

    // Warm horizon band and forward scattering around the sun.
    const float sun_cos = env_clampf(dot(view_dir, sun_dir), 0.0f, 1.0f);
    const float forward_scatter = expf((sun_cos - 1.0f) / (0.055f + 0.07f * haze));
    const float warm_horizon = expf(-18.0f * fabsf(up));
    const float warm_strength = low_sun * env.warm_scatter;
    day_sky = day_sky + env.horizon_tint * (warm_horizon * warm_strength * (0.55f + 0.45f * haze));
    day_sky = day_sky + env.sun_tint * (forward_scatter * low_sun * 0.45f);

    const float deg_to_rad = 0.01745329251994329577f;
    const float sun_radius = 0.53f * deg_to_rad;
    const float sun_disk = (sun_cos >= cosf(sun_radius)) ? 1.0f : 0.0f;
    const float sun_glow = expf((sun_cos - 1.0f) / (0.010f + 0.020f * haze));
    const float sun_halo = expf((sun_cos - 1.0f) / (0.090f + 0.050f * haze)); // broad halo
    const float sun_visible = env_smoothstep(-0.07f, 0.02f, sun_elev);
    const float sun_energy_boost = 1.0f + 3.5f * high_sun;            // midday should be bright
    const Vec3 sun_term = env.sun_tint * env.sun_intensity * sun_visible *
                          (sun_disk * 12.0f + sun_glow * 2.0f + sun_halo * 0.75f) *
                          sun_energy_boost;

    const float sky_energy_boost = 1.05f + 2.6f * high_sun;           // midday sky lift
    Vec3 sky_day_total = day_sky * (env.sky_intensity * sky_energy_boost) + sun_term;

    // Night sky base.
    const Vec3 night_zenith = make_vec3(0.0018f, 0.0032f, 0.0100f);
    const Vec3 night_horizon = make_vec3(0.0080f, 0.0110f, 0.0170f);
    Vec3 sky_night_total = env_lerp(night_zenith, night_horizon, horizon) * env.sky_intensity;

    // Moon disk + glow + broad sky lift (optional).
    if (env.moon_enabled && env.moon_intensity > 0.0f) {
        const float moon_cos = env_clampf(dot(view_dir, moon_dir));
        const float moon_radius = 0.55f * deg_to_rad;
        const float moon_disk = (moon_cos >= cosf(moon_radius)) ? 1.0f : 0.0f;
        const float moon_glow = expf((moon_cos - 1.0f) / 0.020f);
        const float moon_halo = expf((moon_cos - 1.0f) / 0.110f);
        const Vec3 moon_color = make_vec3(0.72f, 0.80f, 1.00f);
        const Vec3 moon_term = moon_color * env.moon_intensity * (moon_disk * 4.5f + moon_glow * 1.2f + moon_halo * 0.35f);
        const Vec3 moon_lift = moon_color * (0.08f * env.moon_intensity) * (0.3f + 0.7f * horizon);
        sky_night_total = sky_night_total + moon_term + moon_lift;
    }

    // Deterministic star field above horizon.
    Vec3 stars = make_vec3(0.0f, 0.0f, 0.0f);
    if (up > 0.0f && night_factor > 0.0f) {
        const float density = fmaxf(env.star_density, 0.1f);
        const float grid = 280.0f + 520.0f * density;
        const float gx = floorf((view_dir.x * 0.5f + 0.5f) * grid);
        const float gy = floorf((view_dir.y * 0.5f + 0.5f) * grid);
        const float gz = floorf(env_clampf(view_dir.z) * grid);
        const float h0 = env_hash3(gx + 13.0f * gz, gy - 7.0f * gz, gz);
        const float threshold = env_clampf(env.star_threshold - 0.0009f * (density - 1.0f), 0.94f, 0.9999f);
        if (h0 > threshold) {
            const float local = (h0 - threshold) / fmaxf(1e-5f, (1.0f - threshold));
            const float spark = powf(local, 6.0f);
            const float twinkle = env_hash3(gx + 17.0f, gy + 11.0f, gz + 5.0f);
            const float temp = env_hash3(gx + 2.0f, gy + 31.0f, gz + 43.0f);
            const Vec3 star_color = env_lerp(make_vec3(0.80f, 0.87f, 1.00f), make_vec3(1.0f, 0.92f, 0.82f), temp);
            float star_vis = env.star_intensity * spark * (0.35f + 0.65f * twinkle);
            if (env.moon_enabled && env.moon_intensity > 0.0f) {
                star_vis *= env_clampf(1.0f - 0.65f * env.moon_intensity);
            }
            star_vis *= powf(env_clampf(up), 0.20f);
            star_vis *= night_factor;
            stars = star_color * star_vis;
        }
    }

    Vec3 sky_total = sky_day_total * day_factor + (sky_night_total + stars) * night_factor;

    float ground_boost = 0.025f * day_factor * env.sun_intensity;
    if (env.moon_enabled && env.moon_intensity > 0.0f) {
        ground_boost += 0.06f * night_factor * env.moon_intensity;
    }
    Vec3 ground_total = env.ground_color * (env.ground_intensity + ground_boost);
    ground_total = ground_total + day_sky * (0.10f * day_factor) + sky_night_total * (0.08f * night_factor);

    sky_total = sky_total * env.tint;
    ground_total = ground_total * env.tint;

    // Smooth blend removes visible horizon seam from sky/ground branch transitions.
    const float horizon_eps = 0.035f;
    const float sky_mix = env_smoothstep(-horizon_eps, horizon_eps, up);
    const Vec3 radiance = env_lerp(ground_total, sky_total, sky_mix);

    return env_exp_tonemap(radiance, env.exposure);
}

#endif
