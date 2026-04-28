#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

// =============================================================================
// Environment lighting module (physically-based, linear HDR output)
// =============================================================================
// This version explicitly separates:
//   1) LINEAR HDR radiance evaluation (EvaluateEnvironment)
//      - Used by the path tracer for direct lighting, indirect bounces, MIS.
//      - Returns unbounded physical radiance. NEVER applies tone mapping.
//   2) DISPLAY FINALIZATION (FinalizeImagePixel)
//      - Used once per final pixel by the image pipeline.
//      - Applies exposure and ACES tone mapping.
//
// WHY THIS MATTERS:
//   A path tracer evaluates the environment many times per pixel for light
//   sampling, MIS, and indirect illumination. If the environment is already
//   tone-mapped, bright features (sun disk, horizon glow) are clipped and the
//   integrator loses energy forever. BSDF-side and light-side estimators must
//   both see linear radiance for MIS weights to be valid. Tone mapping is a
//   display operation and must happen AFTER all sampling is complete.
//
// Physics content:
//   - Rayleigh scattering (molecules -> blue sky, red sunsets)
//   - Mie scattering (aerosols/dust -> sun halo, haze, orange horizon)
//   - Single-scattering Nishita-style integration on a spherical shell
//   - Sun disk via solid-angle-derived radiance
//
// Kept features: lat-long HDRI map, moon, stars, ground hemisphere,
//                legacy artistic sky as opt-in fallback.
// =============================================================================

#include <cmath>
#include <cstring>
#include <cstdio>
#include "vec3.h"
#include "texture.h"

#ifndef HYBRID_FUNC
#  ifdef __CUDACC__
#    define HYBRID_FUNC __host__ __device__
#  else
#    define HYBRID_FUNC
#  endif
#endif

#ifndef ENV_PI
#  define ENV_PI 3.14159265358979323846f
#endif


// =============================================================================
// Enumerations
// =============================================================================

enum class EnvironmentPreset : int {
    Custom      = 0,
    Sunrise     = 1,
    Midday      = 2,
    Sunset      = 3,
    NightMoon   = 4,
    NightNoMoon = 5
};

enum class EnvironmentDebugMode : int {
    None              = 0,
    RayleighOnly      = 1,   // return only Rayleigh in-scatter
    MieOnly           = 2,   // return only Mie in-scatter
    SunDiskOnly       = 3,   // return only the direct solar disc
    TransmittanceOnly = 4    // return view-ray transmittance T(origin -> atmo exit)
};


// =============================================================================
// Environment struct
// =============================================================================
// tone_mapping_enabled and exposure are kept for JSON back-compat BUT they
// are NO LONGER used by EvaluateEnvironment. They are consumed by
// FinalizeImagePixel at the image pipeline stage.
// =============================================================================

struct Environment {
    bool              enabled = false;
    EnvironmentPreset preset  = EnvironmentPreset::Custom;

    // -------- Lat-long HDRI --------
    TextureData* latlong_map     = nullptr;
    float        map_intensity   = 1.0f;
    float        map_rotation_deg = 0.0f;

    // -------- Sun --------
    Vec3  sun_dir                 = make_vec3(0.0f, 0.5f, 0.8660254f);
    float sun_intensity           = 20.0f;
    Vec3  sun_tint                = make_vec3(1.0f, 1.0f, 1.0f);
    float sun_angular_radius_deg  = 0.265f;

    // -------- Physical atmosphere --------
    bool  use_physical_sky         = true;
    float rayleigh_strength        = 1.0f;
    float mie_strength             = 1.0f;
    float mie_g                    = 0.76f;
    float atmosphere_height_km     = 60.0f;
    float rayleigh_scale_height_km = 8.0f;
    float mie_scale_height_km      = 1.2f;
    int   view_samples             = 8;
    int   light_samples            = 4;
    float camera_altitude_m        = 1.0f;

    // -------- Artistic tuning (NOW connected to physical path) --------
    // sky_intensity: linear multiplier on scattered sky radiance (not sun disk).
    // turbidity:     physically-inspired Mie multiplier, 1 = pure molecular.
    // horizon_tint/warm_scatter: optional additive artistic overlay.
    float sky_intensity   = 1.0f;
    float turbidity       = 2.5f;
    Vec3  horizon_tint    = make_vec3(1.0f, 0.58f, 0.28f);
    float warm_scatter    = 0.0f;
    // -------- Horizon band controls --------
    // horizon_falloff: controls the vertical spread of the warm horizon glow.
    //   Lower value = wider band. Higher value = narrower band.
    //   30.0f should reproduce the current default behavior.
    float horizon_falloff = 30.0f;

    // horizon_strength: controls the multiplier applied to the warm horizon overlay
    // in the physical-sky path.
    //   0.15f should reproduce the current default behavior.
    float horizon_strength = 0.15f;
    Vec3  tint            = make_vec3(1.0f, 1.0f, 1.0f);

    // -------- Ground hemisphere --------
    Vec3  ground_color            = make_vec3(0.44f, 0.44f, 0.46f);
    float ground_intensity        = 0.12f;
    // When false (default): no artificial ground hemisphere; sky continues cleanly
    // below the horizon and the real mesh ground handles visible ground.
    // When true: restores the legacy ground/sky blend for backward compatibility.
    bool  use_ground_hemisphere   = false;

    // -------- Moon & stars --------
    bool  moon_enabled   = true;
    Vec3  moon_dir       = make_vec3(-0.55f, -0.65f, 0.52f);
    float moon_intensity = 0.15f;
    float star_intensity = 0.35f;
    float star_density   = 1.0f;
    float star_threshold = 0.9975f;

    // -------- Image pipeline (NOT used inside EvaluateEnvironment) --------
    bool  tone_mapping_enabled = true;
    float exposure             = 1.0f;

    // -------- Debug --------
    EnvironmentDebugMode debug_mode = EnvironmentDebugMode::None;
};


// =============================================================================
// Small math helpers
// =============================================================================

HYBRID_FUNC inline float env_clampf(float x, float lo = 0.0f, float hi = 1.0f) {
    return fmaxf(lo, fminf(x, hi));
}
HYBRID_FUNC inline float env_lerp_f(float a, float b, float t) {
    return a + (b - a) * t;
}
HYBRID_FUNC inline Vec3 env_lerp(const Vec3& a, const Vec3& b, float t) {
    return make_vec3(env_lerp_f(a.x, b.x, t),
                     env_lerp_f(a.y, b.y, t),
                     env_lerp_f(a.z, b.z, t));
}
HYBRID_FUNC inline float env_smoothstep(float e0, float e1, float x) {
    const float t = env_clampf((x - e0) / (e1 - e0));
    return t * t * (3.0f - 2.0f * t);
}
HYBRID_FUNC inline float env_fract(float x) { return x - floorf(x); }

HYBRID_FUNC inline Vec3 env_safe_normalize(const Vec3& v, const Vec3& fallback) {
    const float len2 = length_squared(v);
    if (len2 <= 1e-12f) return fallback;
    return v * (1.0f / sqrtf(len2));
}

HYBRID_FUNC inline float env_hash3(float x, float y, float z) {
    const unsigned int ix = static_cast<unsigned int>(static_cast<int>(x));
    const unsigned int iy = static_cast<unsigned int>(static_cast<int>(y));
    const unsigned int iz = static_cast<unsigned int>(static_cast<int>(z));
    unsigned int vx = ix * 1664525u + 1013904223u;
    unsigned int vy = iy * 1664525u + 1013904223u;
    unsigned int vz = iz * 1664525u + 1013904223u;
    vx += vy * vz;  vy += vz * vx;  vz += vx * vy;
    vx ^= vx >> 16u; vy ^= vy >> 16u; vz ^= vz >> 16u;
    vx += vy * vz;
    return static_cast<float>(vx >> 8u) * (1.0f / 16777216.0f);
}


// =============================================================================
// Angle <-> direction conversions
// =============================================================================

HYBRID_FUNC inline Vec3 DirectionFromElevationAzimuthDeg(float elevation_deg,
                                                        float azimuth_deg) {
    constexpr float kDegToRad = 0.01745329251994329577f;
    const float elev = elevation_deg * kDegToRad;
    const float azim = azimuth_deg   * kDegToRad;
    const float ce = cosf(elev);
    return env_safe_normalize(
        make_vec3(ce * sinf(azim), ce * cosf(azim), sinf(elev)),
        make_vec3(0.0f, -1.0f, 0.0f));
}

HYBRID_FUNC inline void ElevationAzimuthFromDirectionDeg(
    const Vec3& dir_in, float& elevation_deg, float& azimuth_deg)
{
    constexpr float kRadToDeg = 57.295779513082320876f;
    const Vec3 d = env_safe_normalize(dir_in, make_vec3(0.0f, 1.0f, 0.0f));
    elevation_deg = asinf(env_clampf(d.z, -1.0f, 1.0f)) * kRadToDeg;
    float az = atan2f(d.x, d.y) * kRadToDeg;
    if (az < 0.0f) az += 360.0f;
    azimuth_deg = az;
}


// =============================================================================
// Preset application
// =============================================================================

HYBRID_FUNC inline void ApplyEnvironmentPreset(Environment& env,
                                               EnvironmentPreset preset) {
    env.preset = preset;

    // --- FIX: Exit immediately if custom, preserving JSON values ---
    if (preset == EnvironmentPreset::Custom) {
        return; 
    }
    // ---------------------------------------------------------------

    env.sun_dir                  = DirectionFromElevationAzimuthDeg(30.0f, 180.0f);
    env.sun_intensity            = 20.0f;
    env.sun_tint                 = make_vec3(1.0f, 1.0f, 1.0f);
    env.sun_angular_radius_deg   = 0.265f;
    env.use_physical_sky         = true;
    env.rayleigh_strength        = 1.0f;
    env.mie_strength             = 1.0f;
    env.mie_g                    = 0.76f;
    env.atmosphere_height_km     = 60.0f;
    env.rayleigh_scale_height_km = 8.0f;
    env.mie_scale_height_km      = 1.2f;
    env.view_samples             = 8;
    env.light_samples            = 4;
    env.camera_altitude_m        = 1.0f;

    env.sky_intensity    = 1.0f;
    env.turbidity        = 2.5f;
    env.horizon_tint     = make_vec3(1.0f, 0.58f, 0.28f);
    env.warm_scatter     = 0.0f;
    env.horizon_falloff  = 30.0f;
    env.horizon_strength = 0.15f;
    env.tint             = make_vec3(1.0f, 1.0f, 1.0f);
    env.ground_color     = make_vec3(0.44f, 0.44f, 0.46f);
    env.ground_intensity = 0.12f;

    env.moon_enabled     = true;
    env.moon_dir         = DirectionFromElevationAzimuthDeg(25.0f, 220.0f);
    env.moon_intensity   = 0.0f;
    env.star_intensity   = 0.0f;
    env.star_density     = 1.0f;
    env.star_threshold   = 0.9975f;

    env.tone_mapping_enabled = true;
    env.exposure             = 1.0f;
    env.debug_mode           = EnvironmentDebugMode::None;

    switch (preset) {
        case EnvironmentPreset::Sunrise:
            env.sun_dir       = DirectionFromElevationAzimuthDeg(6.0f, 278.0f);
            env.sun_intensity = 22.0f;
            env.mie_strength  = 2.2f;
            env.mie_g         = 0.80f;
            env.turbidity     = 3.5f;
            break;
        case EnvironmentPreset::Midday:
            env.sun_dir       = DirectionFromElevationAzimuthDeg(68.0f, 25.0f);
            env.sun_intensity = 22.0f;
            env.mie_strength  = 0.6f;
            env.mie_g         = 0.76f;
            env.turbidity     = 2.0f;
            break;
        case EnvironmentPreset::Sunset:
            env.sun_dir       = DirectionFromElevationAzimuthDeg(5.0f, 80.0f);
            env.sun_intensity = 22.0f;
            env.mie_strength  = 2.8f;
            env.mie_g         = 0.82f;
            env.turbidity     = 4.5f;
            break;
        case EnvironmentPreset::NightMoon:
            env.sun_dir        = DirectionFromElevationAzimuthDeg(-18.0f, 330.0f);
            env.sun_intensity  = 0.0f;
            env.moon_enabled   = true;
            env.moon_intensity = 0.38f;
            env.star_intensity = 0.42f;
            env.star_density   = 1.10f;
            env.star_threshold = 0.9975f;
            env.exposure       = 1.3f;
            break;
        case EnvironmentPreset::NightNoMoon:
            env.sun_dir        = DirectionFromElevationAzimuthDeg(-18.0f, 330.0f);
            env.sun_intensity  = 0.0f;
            env.moon_enabled   = false;
            env.moon_intensity = 0.0f;
            env.star_intensity = 1.25f;
            env.star_density   = 1.55f;
            env.star_threshold = 0.9965f;
            env.exposure       = 1.35f;
            break;
        case EnvironmentPreset::Custom:
        default: break;
    }
}

inline const char* EnvironmentPresetToString(EnvironmentPreset p) {
    switch (p) {
        case EnvironmentPreset::Sunrise:     return "sunrise";
        case EnvironmentPreset::Midday:      return "midday";
        case EnvironmentPreset::Sunset:      return "sunset";
        case EnvironmentPreset::NightMoon:   return "night_moon";
        case EnvironmentPreset::NightNoMoon: return "night_nomoon";
        default:                             return "custom";
    }
}
inline EnvironmentPreset EnvironmentPresetFromString(const char* s) {
    if (s == nullptr) return EnvironmentPreset::Custom;
    if (std::strcmp(s, "sunrise") == 0)      return EnvironmentPreset::Sunrise;
    if (std::strcmp(s, "midday") == 0)       return EnvironmentPreset::Midday;
    if (std::strcmp(s, "sunset") == 0)       return EnvironmentPreset::Sunset;
    if (std::strcmp(s, "night_moon") == 0)   return EnvironmentPreset::NightMoon;
    if (std::strcmp(s, "night_nomoon") == 0) return EnvironmentPreset::NightNoMoon;
    return EnvironmentPreset::Custom;
}


// =============================================================================
// Lat-long map sampling (returns linear radiance; NO tone mapping here)
// =============================================================================

HYBRID_FUNC inline Vec3 env_sample_latlong_texel(const TextureData* tex, int x, int y) {
    if (tex == nullptr || tex->data == nullptr || tex->width <= 0 ||
        tex->height <= 0 || tex->channels <= 0)
        return make_vec3(0.0f, 0.0f, 0.0f);
    const int w = tex->width;
    const int h = tex->height;
    x = (x % w + w) % w;
    y = y < 0 ? 0 : (y >= h ? h - 1 : y);
    const int idx = (y * w + x) * tex->channels;
    const unsigned char* p = tex->data + idx;
    if (tex->channels >= 3) {
        return make_vec3(p[0] * (1.0f / 255.0f),
                         p[1] * (1.0f / 255.0f),
                         p[2] * (1.0f / 255.0f));
    }
    const float c = p[0] * (1.0f / 255.0f);
    return make_vec3(c, c, c);
}

HYBRID_FUNC inline Vec3 EvaluateLatLongEnvironment(const Vec3& dir_in,
                                                   const Environment& env) {
    if (env.latlong_map == nullptr || env.latlong_map->data == nullptr)
        return make_vec3(0.0f, 0.0f, 0.0f);

    constexpr float kInv2Pi = 0.15915494309189533577f;
    constexpr float kInvPi  = 0.31830988618379067154f;
    const Vec3 d = env_safe_normalize(dir_in, make_vec3(0.0f, 1.0f, 0.0f));
    const float rot = env.map_rotation_deg * (1.0f / 360.0f);

    float u = atan2f(d.x, d.y) * kInv2Pi + 0.5f + rot;
    u = env_fract(u);
    const float z = env_clampf(d.z, -1.0f, 1.0f);
    const float v = acosf(z) * kInvPi;

    const float x = u * (env.latlong_map->width  - 1);
    const float y = v * (env.latlong_map->height - 1);
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const Vec3 c00 = env_sample_latlong_texel(env.latlong_map, x0, y0);
    const Vec3 c10 = env_sample_latlong_texel(env.latlong_map, x1, y0);
    const Vec3 c01 = env_sample_latlong_texel(env.latlong_map, x0, y1);
    const Vec3 c11 = env_sample_latlong_texel(env.latlong_map, x1, y1);
    const Vec3 c0  = env_lerp(c00, c10, tx);
    const Vec3 c1  = env_lerp(c01, c11, tx);
    return env_lerp(c0, c1, ty) * env.map_intensity;
}


// =============================================================================
// Physical constants & phase functions
// =============================================================================

HYBRID_FUNC inline float env_earth_radius_m() { return 6360000.0f; }

// Rayleigh per-channel coefficient at sea level (1/m), for RGB primaries.
HYBRID_FUNC inline Vec3 env_beta_R_sea_level() {
    return make_vec3(5.8e-6f, 13.5e-6f, 33.1e-6f);
}
HYBRID_FUNC inline float env_beta_M_sea_level() { return 21.0e-6f; }

// Map artistic turbidity (Preetham-inspired) to an additional Mie multiplier.
//   turbidity = 1   -> factor 1.0  (pure molecular atmosphere)
//   turbidity = 2.5 -> factor 1.45 (slightly hazy)
//   turbidity = 10  -> factor 3.7  (desert dust storm)
// Physically inspired but simplified: gives users a single predictable haze
// knob that composes with mie_strength.
HYBRID_FUNC inline float env_turbidity_to_mie_factor(float turbidity) {
    return 1.0f + fmaxf(0.0f, turbidity - 1.0f) * 0.3f;
}

HYBRID_FUNC inline bool env_ray_sphere(const Vec3& o, const Vec3& d,
                                        const Vec3& c, float R,
                                        float& t0, float& t1) {
    const Vec3  L    = o - c;
    const float b    = dot(L, d);
    const float k    = dot(L, L) - R * R;
    const float disc = b * b - k;
    if (disc < 0.0f) return false;
    const float s = sqrtf(disc);
    t0 = -b - s;
    t1 = -b + s;
    return true;
}

// Rayleigh phase: P_R(mu) = 3/(16 pi) (1 + mu^2)
HYBRID_FUNC inline float env_phase_rayleigh(float mu) {
    return (3.0f / (16.0f * ENV_PI)) * (1.0f + mu * mu);
}
// Henyey-Greenstein phase for Mie.
HYBRID_FUNC inline float env_phase_mie_hg(float mu, float g) {
    const float g2    = g * g;
    const float denom = powf(fmaxf(1.0f + g2 - 2.0f * g * mu, 1e-6f), 1.5f);
    return (1.0f / (4.0f * ENV_PI)) * ((1.0f - g2) / denom);
}


// =============================================================================
// View-ray transmittance through the atmosphere (TransmittanceOnly debug)
// =============================================================================
// Integrates optical depth from observer to atmosphere exit along view_dir
// and returns per-channel Beer-Lambert transmittance T = exp(-tau).
//
// Visual meaning:
//   - Near zenith: short atmosphere path, T ~ (1,1,1), image near-white.
//   - Near horizon: long path, blue scatters out most, T is warm/red.
//   - Ray that clips earth: T = 0 (black).
// This is the factor by which light coming from beyond the atmosphere in
// view_dir (e.g., the sun disk) would be attenuated before reaching the eye.
// =============================================================================

HYBRID_FUNC inline Vec3 env_view_transmittance(const Vec3& view_dir,
                                                const Environment& env) {
    const float R_e = env_earth_radius_m();
    const float R_a = R_e + env.atmosphere_height_km * 1000.0f;
    const float H_r = env.rayleigh_scale_height_km  * 1000.0f;
    const float H_m = env.mie_scale_height_km       * 1000.0f;

    const float mie_factor = env_turbidity_to_mie_factor(env.turbidity);
    const Vec3  beta_R = env_beta_R_sea_level() * env.rayleigh_strength;
    const float beta_M = env_beta_M_sea_level() * env.mie_strength * mie_factor;

    const Vec3 earth_c = make_vec3(0.0f, 0.0f, -R_e);
    const Vec3 origin  = make_vec3(0.0f, 0.0f, env.camera_altitude_m);

    float t0, t1;
    if (!env_ray_sphere(origin, view_dir, earth_c, R_a, t0, t1))
        return make_vec3(0.0f, 0.0f, 0.0f);
        
    const float t_start = fmaxf(0.0f, t0);
    float t_end         = t1; // Made this mutable so we can cap it

    // --- Capping ray at the Earth's surface ---
    float e0, e1;
    if (env_ray_sphere(origin, view_dir, earth_c, R_e, e0, e1) && e0 > 0.0f) {
        t_end = fminf(t_end, e0);
    }

    if (t_end <= t_start) return make_vec3(0.0f, 0.0f, 0.0f);

    const int   N  = env.view_samples > 0 ? env.view_samples : 8;
    const float ds = (t_end - t_start) / float(N);

    float od_R = 0.0f;
    float od_M = 0.0f;
    float t = t_start + ds * 0.5f;
    for (int i = 0; i < N; ++i) {
        const Vec3  P   = origin + view_dir * t;
        const Vec3  de  = P - earth_c;
        
        // Clamp to 0.0f to avoid negative altitude from float precision errors
        const float h   = fmaxf(0.0f, sqrtf(dot(de, de)) - R_e); 
        
        od_R += expf(-h / H_r) * ds;
        od_M += expf(-h / H_m) * ds;
        t += ds;
    }

    const Vec3 tau = beta_R * od_R
                   + make_vec3(beta_M, beta_M, beta_M) * (od_M * 1.1f);
    return make_vec3(expf(-tau.x), expf(-tau.y), expf(-tau.z));
}

// =============================================================================
// Single-scattering atmosphere evaluation (linear HDR radiance)
// =============================================================================
// sky_intensity acts as a linear multiplier on the SCATTERED sky (not sun
// disk). turbidity contributes an additional physically-inspired Mie factor.
// Both are applied here so users tuning JSON get a predictable effect.
// =============================================================================

HYBRID_FUNC inline Vec3 env_single_scattering(const Vec3& view_dir,
                                              const Vec3& sun_dir,
                                              const Environment& env) {
    const float R_e = env_earth_radius_m();
    const float R_a = R_e + env.atmosphere_height_km * 1000.0f;
    const float H_r = env.rayleigh_scale_height_km  * 1000.0f;
    const float H_m = env.mie_scale_height_km       * 1000.0f;

    const float mie_factor = env_turbidity_to_mie_factor(env.turbidity);
    const Vec3  beta_R = env_beta_R_sea_level() * env.rayleigh_strength;
    const float beta_M = env_beta_M_sea_level() * env.mie_strength * mie_factor;

    const Vec3 earth_c = make_vec3(0.0f, 0.0f, -R_e);
    const Vec3 origin  = make_vec3(0.0f, 0.0f, env.camera_altitude_m);

    float t0, t1;
    if (!env_ray_sphere(origin, view_dir, earth_c, R_a, t0, t1))
        return make_vec3(0.0f, 0.0f, 0.0f);
    const float t_start = fmaxf(0.0f, t0);
    const float t_end   = t1;
    if (t_end <= t_start) return make_vec3(0.0f, 0.0f, 0.0f);

    const int   N  = env.view_samples > 0 ? env.view_samples : 8;
    const float ds = (t_end - t_start) / float(N);

    Vec3 sum_R = make_vec3(0.0f, 0.0f, 0.0f);
    Vec3 sum_M = make_vec3(0.0f, 0.0f, 0.0f);

    float od_R_view = 0.0f;
    float od_M_view = 0.0f;

    float t = t_start + ds * 0.5f;
    for (int i = 0; i < N; ++i) {
        const Vec3  P        = origin + view_dir * t;
        const Vec3  d_e      = P - earth_c;
        const float altitude = sqrtf(dot(d_e, d_e)) - R_e;

        const float dens_R = expf(-altitude / H_r);
        const float dens_M = expf(-altitude / H_m);

        od_R_view += dens_R * ds;
        od_M_view += dens_M * ds;

        // Optical depth from P to atmosphere exit along sun_dir.
        float s0, s1;
        bool  earth_shadow = false;
        float od_R_light   = 0.0f;
        float od_M_light   = 0.0f;
        if (env_ray_sphere(P, sun_dir, earth_c, R_a, s0, s1) && s1 > 0.0f) {
            const int   M  = env.light_samples > 0 ? env.light_samples : 4;
            const float dl = s1 / float(M);
            float sl = dl * 0.5f;
            for (int j = 0; j < M; ++j) {
                const Vec3  Pl    = P + sun_dir * sl;
                const Vec3  dl_e  = Pl - earth_c;
                const float h_l   = sqrtf(dot(dl_e, dl_e)) - R_e;
                if (h_l < 0.0f) { earth_shadow = true; break; }
                od_R_light += expf(-h_l / H_r) * dl;
                od_M_light += expf(-h_l / H_m) * dl;
                sl += dl;
            }
        } else {
            earth_shadow = true;
        }

        if (!earth_shadow) {
            const float od_M_total = (od_M_view + od_M_light) * 1.1f;
            const float od_R_total = (od_R_view + od_R_light);
            const Vec3 tau = beta_R * od_R_total
                           + make_vec3(beta_M, beta_M, beta_M) * od_M_total;
            const Vec3 T = make_vec3(expf(-tau.x), expf(-tau.y), expf(-tau.z));
            sum_R = sum_R + T * (dens_R * ds);
            sum_M = sum_M + T * (dens_M * ds);
        }

        t += ds;
    }

    const float mu = env_clampf(dot(view_dir, sun_dir), -1.0f, 1.0f);
    const float pR = env_phase_rayleigh(mu);
    const float pM = env_phase_mie_hg(mu, env.mie_g);

    const Vec3 in_R = (sum_R * beta_R) * pR;
    const Vec3 in_M = (sum_M * make_vec3(beta_M, beta_M, beta_M)) * pM;

    // sun_intensity scales incoming solar energy; sky_intensity is a user
    // multiplier to brighten/darken scattered sky without touching sun disk.
    const float scale = env.sun_intensity * env.sky_intensity;

    if (env.debug_mode == EnvironmentDebugMode::RayleighOnly)
        return in_R * scale;
    if (env.debug_mode == EnvironmentDebugMode::MieOnly)
        return in_M * scale;
    return (in_R + in_M) * scale;
}


// =============================================================================
// Sun disk
// =============================================================================
// Radiance derived from solid angle: L = I_sun / Omega_sun, where
// Omega_sun = 2 pi (1 - cos(angular_radius)). This makes the disk's integrated
// energy proportional to sun_intensity (physically consistent) rather than
// relying on a magic constant like the old 120.0. kDiskFudge prevents the
// fully-physical peak radiance (~15000x) from causing fireflies in naive
// path tracing while still reading as "very bright sun".
// =============================================================================

HYBRID_FUNC inline Vec3 env_sun_disk(const Vec3& view_dir, const Vec3& sun_dir,
                                     const Environment& env) {
    const float radius_rad = env.sun_angular_radius_deg * 0.01745329251994329577f;
    const float cos_outer  = cosf(radius_rad);
    const float cos_inner  = cosf(radius_rad * 0.75f);
    const float mu         = dot(view_dir, sun_dir);

    const float disk = env_smoothstep(cos_outer, cos_inner, mu);
    if (disk <= 0.0f) return make_vec3(0.0f, 0.0f, 0.0f);

    const float sun_vis = env_smoothstep(-0.07f, 0.02f, sun_dir.z);
    if (sun_vis <= 0.0f) return make_vec3(0.0f, 0.0f, 0.0f);

    const float omega_sun  = 2.0f * ENV_PI * (1.0f - cos_outer);
    const float kDiskFudge = 0.02f;
    const float radiance   = (env.sun_intensity / fmaxf(omega_sun, 1e-8f)) * kDiskFudge;

    // Physically motivated attenuation through the atmosphere along the
    // viewing direction toward the sun. This makes the sun dim and redden
    // naturally near the horizon (blue scatters out on long horizon paths).
    const Vec3 T_sun = env_view_transmittance(sun_dir, env);
    return env.sun_tint * T_sun * (radiance * disk * sun_vis);
}


// =============================================================================
// Legacy artistic sky (opt-in fallback, use_physical_sky == false)
// =============================================================================

HYBRID_FUNC inline Vec3 env_legacy_sky(const Vec3& view_dir,
                                       const Vec3& sun_dir,
                                       const Environment& env) {
    const float up     = view_dir.z;
    const float sky_up = env_clampf(up);
    const float high_sun = env_smoothstep(0.707f, 0.98f, sun_dir.z);
    const float low_sun  = 1.0f - env_smoothstep(0.20f, 0.75f, sun_dir.z);
    const float horizon  = expf(-8.6f * sky_up);
    float haze = env_clampf((env.turbidity - 1.0f) / 10.0f);
    haze *= (1.0f - 0.35f * high_sun);
    const Vec3 zen_blue = make_vec3(0.12f, 0.38f, 0.90f) * (1.0f + 0.7f * high_sun);
    const Vec3 hor_blue = make_vec3(0.52f, 0.72f, 1.00f)
                        * (1.0f + 0.12f * high_sun + 0.28f * haze);
    Vec3 sky = env_lerp(zen_blue, hor_blue, horizon);
    const float mu      = env_clampf(dot(view_dir, sun_dir), 0.0f, 1.0f);
    const float forward = expf((mu - 1.0f) / (0.055f + 0.07f * haze));
    const float warm_h  = expf(-6.0f * fabsf(up));
    const float warm_s  = low_sun * env.warm_scatter;
    sky = sky + env.horizon_tint * (warm_h * warm_s * (0.28f + 0.22f * haze));
    sky = sky + env.sun_tint * (forward * low_sun * 0.45f);
    return sky * env.sky_intensity;
}


// =============================================================================
// Night sky (moon + stars)
// =============================================================================

HYBRID_FUNC inline Vec3 env_night_sky(const Vec3& view_dir,
                                      const Environment& env,
                                      float night_factor,
                                      float horizon_weight) {
    constexpr float kDegToRad = 0.01745329251994329577f;
    const Vec3 moon_dir = env_safe_normalize(env.moon_dir,
                                             make_vec3(-0.5513f, -0.6515f, 0.5212f));
    const Vec3 z_night = make_vec3(0.0018f, 0.0032f, 0.0100f);
    const Vec3 h_night = make_vec3(0.0080f, 0.0110f, 0.0170f);
    Vec3 sky = env_lerp(z_night, h_night, horizon_weight) * env.sky_intensity;

    if (env.moon_enabled && env.moon_intensity > 0.0f) {
        const float m_cos = env_clampf(dot(view_dir, moon_dir));
        const float m_r   = 0.55f * kDegToRad;
        const float disk  = env_smoothstep(cosf(m_r * 1.4f), cosf(m_r * 0.6f), m_cos);
        const float glow  = expf((m_cos - 1.0f) / 0.020f);
        const float halo  = expf((m_cos - 1.0f) / 0.110f);
        const Vec3  mc    = make_vec3(0.72f, 0.80f, 1.00f);
        sky = sky + mc * env.moon_intensity * (disk * 4.5f + glow * 1.2f + halo * 0.35f);
        sky = sky + mc * (0.08f * env.moon_intensity) * (0.3f + 0.7f * horizon_weight);
    }

    Vec3 stars = make_vec3(0.0f, 0.0f, 0.0f);
    if (view_dir.z > 0.0f && night_factor > 0.0f) {
        constexpr float kPi    = 3.14159265358979f;
        constexpr float kTwoPi = 6.28318530717959f;
        const float density = fmaxf(env.star_density, 0.1f);
        const float grid_az = floorf(512.0f * density);
        const float grid_el = floorf(256.0f * density);
        const float phi     = atan2f(view_dir.x, view_dir.y);
        const float theta   = acosf(env_clampf(view_dir.z, -1.0f, 1.0f));
        const float gx = floorf((phi / kTwoPi + 0.5f) * grid_az);
        const float gy = floorf((theta / kPi) * grid_el);
        const float gz = floorf(density * 17.0f);
        const float thr = env_clampf(env.star_threshold - 0.0009f * (density - 1.0f),
                                      0.94f, 0.9999f);
        const float h0 = env_hash3(gx + 13.0f * gz, gy - 7.0f * gz, gz);
        if (h0 > thr) {
            const float local = (h0 - thr) / fmaxf(1e-5f, 1.0f - thr);
            const float spark = powf(local, 6.0f);
            const float twk   = env_hash3(gx + 17.0f, gy + 11.0f, gz + 5.0f);
            const float tmp   = env_hash3(gx + 2.0f,  gy + 31.0f, gz + 43.0f);
            const Vec3 cC = make_vec3(0.80f, 0.87f, 1.00f);
            const Vec3 cW = make_vec3(1.00f, 0.92f, 0.82f);
            const Vec3 col = env_lerp(cC, cW, tmp);
            float vis = env.star_intensity * spark * (0.35f + 0.65f * twk);
            if (env.moon_enabled && env.moon_intensity > 0.0f)
                vis *= env_clampf(1.0f - 0.65f * env.moon_intensity);
            vis *= powf(env_clampf(view_dir.z), 0.20f);
            vis *= night_factor;
            stars = col * vis;
        }
    }
    return sky + stars;
}


// =============================================================================
// ACES tone map (kept as helper for the IMAGE PIPELINE only)
// =============================================================================

HYBRID_FUNC inline float env_aces_channel(float v) {
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return env_clampf((v * (a * v + b)) / (v * (c * v + d) + e), 0.0f, 1.0f);
}
HYBRID_FUNC inline Vec3 env_tonemap_aces(const Vec3& x) {
    return make_vec3(env_aces_channel(x.x),
                     env_aces_channel(x.y),
                     env_aces_channel(x.z));
}


// =============================================================================
// Main entry point: LINEAR HDR RADIANCE
// =============================================================================
// Contract:
//   - Input: world-space view direction (normalized internally).
//   - Output: linear HDR radiance. No clamping, no tone mapping, no exposure.
//   - Safe to use inside MIS, direct lighting, indirect bounces.
//   - Debug modes bypass normal blending to isolate specific terms.
// =============================================================================

HYBRID_FUNC inline Vec3 EvaluateEnvironment(const Vec3& dir_in, const Environment& env) {
    if (!env.enabled) return make_vec3(0.0f, 0.0f, 0.0f);

    const Vec3 view_dir = env_safe_normalize(dir_in,      make_vec3(0.0f, 1.0f, 0.0f));
    const Vec3 sun_dir  = env_safe_normalize(env.sun_dir, make_vec3(0.0f, 1.0f, 0.0f));

    // Debug: transmittance visualization (works regardless of path).
    if (env.debug_mode == EnvironmentDebugMode::TransmittanceOnly) {
        return env_view_transmittance(view_dir, env);
    }

    // Debug: sun disk only.
    if (env.debug_mode == EnvironmentDebugMode::SunDiskOnly) {
        return env_sun_disk(view_dir, sun_dir, env);
    }

    // Lat-long HDRI takes priority when bound.
    if (env.latlong_map != nullptr) {
        return EvaluateLatLongEnvironment(dir_in, env) * env.tint;
    }

    // Procedural sky path.
    const float up        = view_dir.z;
    const float sun_elev  = sun_dir.z;
    const float day       = env_smoothstep(-0.18f, 0.03f, sun_elev);
    const float night     = 1.0f - day;
    const float horizon_w = expf(-8.6f * env_clampf(up));

    Vec3 sky_day;
    if (env.use_physical_sky) {
        sky_day = env_single_scattering(view_dir, sun_dir, env);
    } else {
        sky_day = env_legacy_sky(view_dir, sun_dir, env);
    }

    // RayleighOnly and MieOnly return the isolated in-scatter contribution
    // from env_single_scattering() with NO further processing: no sun disk,
    // no warm overlay, no night sky, no ground hemisphere, no horizon blend.
    // Any of those additions would pollute the isolation and hide the term
    // the user is trying to inspect.
    if (env.debug_mode == EnvironmentDebugMode::RayleighOnly ||
        env.debug_mode == EnvironmentDebugMode::MieOnly) {
        return sky_day;
    }

    sky_day = sky_day + env_sun_disk(view_dir, sun_dir, env);

    if (env.warm_scatter > 0.0f && env.use_physical_sky) {
        const float low_sun = 1.0f - env_smoothstep(0.20f, 0.75f, sun_elev);
        const float warm_h  = expf(-env.horizon_falloff * fabsf(up));
        sky_day = sky_day + env.horizon_tint
                * (warm_h * low_sun * env.warm_scatter * env.horizon_strength);
    }

    const Vec3 sky_night = env_night_sky(view_dir, env, night, horizon_w);

    Vec3 sky_total = sky_day * day + sky_night * night;

    sky_total = sky_total * env.tint;

    if (!env.use_ground_hemisphere) {
        // No artificial ground hemisphere: sky continues smoothly below horizon.
        // The real mesh ground plane is responsible for visible ground appearance.
        if (up < 0.0f) {
            const float lower_blend = env_smoothstep(-0.35f, 0.0f, up);
            const Vec3 lower_sky = sky_total * 0.65f;
            return env_lerp(lower_sky, sky_total, lower_blend);
        }
        return sky_total;
    }

    // Ground hemisphere contribution (legacy). sun_intensity is clamped here so
    // raising it for sharper direct light doesn't blow out the ground indirectly.
    const float sun_for_ground = fminf(env.sun_intensity, 30.0f);
    float ground_boost = 0.0075f * day * sun_for_ground;
    if (env.moon_enabled && env.moon_intensity > 0.0f)
        ground_boost += 0.06f * night * env.moon_intensity;

    Vec3 ground_total = env.ground_color * (env.ground_intensity + ground_boost);
    ground_total = ground_total + sky_day   * (0.05f * day)
                                + sky_night * (0.06f * night);
    ground_total = ground_total * env.tint;

    const float sky_mix = env_smoothstep(-0.12f, 0.005f, up);
    return env_lerp(ground_total, sky_total, sky_mix);
}


// =============================================================================
// Image pipeline finalization
// =============================================================================
// Call ONCE per final pixel, after all rays have been integrEvaluateEnvironmentated. Applies
// exposure as a linear multiplier and optional ACES tone mapping to compress
// the HDR signal into displayable [0,1] LDR. Never call this inside the path
// tracer or inside any BSDF/light evaluation.
// =============================================================================

HYBRID_FUNC inline Vec3 FinalizeImagePixel(const Vec3& linear_hdr,
                                           const Environment& env) {
    const Vec3 exposed = linear_hdr * env.exposure;
    if (env.tone_mapping_enabled) return env_tonemap_aces(exposed);
    return exposed;
}


// =============================================================================
// Debug helper: print resolved parameters
// =============================================================================

inline void EnvironmentDebugPrint(const Environment& env) {
    std::printf("[Environment] preset=%s enabled=%d use_physical=%d\n",
                EnvironmentPresetToString(env.preset),
                (int)env.enabled, (int)env.use_physical_sky);
    std::printf("  sun_dir=(%.3f,%.3f,%.3f) sun_intensity=%.2f\n",
                env.sun_dir.x, env.sun_dir.y, env.sun_dir.z, env.sun_intensity);
    std::printf("  rayleigh=%.2f mie=%.2f mie_g=%.2f turbidity=%.2f (mie_factor=%.3f)\n",
                env.rayleigh_strength, env.mie_strength, env.mie_g, env.turbidity,
                env_turbidity_to_mie_factor(env.turbidity));
    std::printf("  sky_intensity=%.2f atmo_h=%.1fkm Hr=%.1fkm Hm=%.1fkm samples=%d/%d\n",
                env.sky_intensity,
                env.atmosphere_height_km,
                env.rayleigh_scale_height_km, env.mie_scale_height_km,
                env.view_samples, env.light_samples);
    std::printf("  horizon_falloff=%.2f horizon_strength=%.3f\n",
                env.horizon_falloff, env.horizon_strength);
    std::printf("  use_ground_hemisphere=%d\n", (int)env.use_ground_hemisphere);
    std::printf("  [image] tone_map=%d exposure=%.2f  [debug] mode=%d\n",
                (int)env.tone_mapping_enabled, env.exposure, (int)env.debug_mode);
}

#endif // ENVIRONMENT_H
