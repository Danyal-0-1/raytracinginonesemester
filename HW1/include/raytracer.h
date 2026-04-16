#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <cmath>
#include <limits>
#include <vector>
#include "ray.h"
#include "brdf.h"

struct Light {
    Vec3 position;
    Vec3 color;
};

// Small offset to avoid self-intersection / "shadow acne"
static constexpr float RT_EPS = 1e-4f;

HYBRID_FUNC inline Vec3 clamp(Vec3 color) {
    if (color.x > 1.0f) color.x = 1.0f;
    if (color.y > 1.0f) color.y = 1.0f;
    if (color.z > 1.0f) color.z = 1.0f;
    if (color.x < 0.0f) color.x = 0.0f;
    if (color.y < 0.0f) color.y = 0.0f;
    if (color.z < 0.0f) color.z = 0.0f;
    return color;
}

HYBRID_FUNC inline float length3(const Vec3& v) {
    return sqrtf(dot(v, v));
}

HYBRID_FUNC inline Vec3 reflect_dir(const Vec3& I, const Vec3& N) {
    // I points *in the ray direction* (from origin toward scene)
    // Reflection: R = I - 2*(I·N)*N
    return I - (2.0f * dot(I, N)) * N;
}

// Scene intersection: find closest hit among triangles
inline bool IntersectScene(const Ray& r,
                           const std::vector<Triangle>& tris,
                           double t_min,
                           double t_max,
                           HitRecord& outRec)
{
    bool hit_anything = false;
    double closest = t_max;

    HitRecord temp{};
    for (const auto& tri : tris) {
        temp = ray_intersection(r, tri);
        if (!temp.hit) continue;

        if (temp.t >= t_min && temp.t < closest) {
            hit_anything = true;
            closest = temp.t;
            outRec = temp;
        }
    }

    return hit_anything;
}

// Check if there is any occluder between P and the light
inline bool IsInShadow(const Vec3& P,
                       const Vec3& N,
                       const Light& light,
                       const std::vector<Triangle>& tris)
{
    Vec3 toL = light.position - P;
    float distToL = length3(toL);
    if (distToL <= 0.0f) return false;

    Vec3 Ldir = toL / distToL;

    // offset along normal to avoid self-hit
    Ray shadowRay(P + N * RT_EPS, Ldir);

    HitRecord shadowHit{};
    // Only count occluders strictly before the light !!!
    return IntersectScene(shadowRay, tris, RT_EPS, double(distToL) - RT_EPS, shadowHit);
}

// Direct lighting with hard shadows (should work with multiple lights)
inline Vec3 ShadeDirect(const Ray& r,
                        const HitRecord& rec,
                        const std::vector<Light>& lights,
                        const std::vector<Triangle>& tris)
{
    // Assuming rec.hit == true already
    Vec3 N = unit_vector(rec.normal);
    Vec3 V = unit_vector(r.origin() - rec.p);

    Vec3 Lo = make_vec3(0,0,0);

    // small ambient (looks nicer)
    Vec3 ambient = rec.mat.albedo * 0.05f;
    Lo = Lo + ambient;

    // add emission (placeholder math for now)
    Lo = Lo + rec.mat.emission;

    for (const auto& light : lights) {
        Vec3 L = unit_vector(light.position - rec.p);
        float NdotL = fmaxf(dot(N, L), 0.0f);
        if (NdotL <= 0.0f) continue;

        // Hard shadows
        if (IsInShadow(rec.p, N, light, tris)) {
            continue;
        }

        // BRDF value
        Vec3 f = EvaluateBRDF(rec.mat, N, V, L);

        // Light contribution
        Vec3 radiance = light.color;
        Vec3 direct = (radiance * f) * NdotL;

        Lo = Lo + direct;
    }

    return Lo;
}

// Recursive tracer: direct + perfect mirror where possible
inline Vec3 TraceRay(const Ray& r,
                     const std::vector<Triangle>& tris,
                     const std::vector<Light>& lights,
                     int depth)
{
    // If out of bounces
    if (depth <= 0) return make_vec3(0,0,0);

    HitRecord rec{};
    if (!IntersectScene(r, tris, RT_EPS, std::numeric_limits<double>::infinity(), rec)) {
        // same !rec.hit logic as before (sky gradient)
        Vec3 unit_dir = unit_vector(r.direction());
        float t = 0.5f * (unit_dir.z + 1.0f);
        return make_vec3(1.0f, 1.0f, 1.0f)*(1.0f-t) + make_vec3(0.5f, 0.7f, 1.0f)*t;
    }

    // Geometry terms
    Vec3 N = unit_vector(rec.normal);

    // 1) direct lighting (with shadows)
    Vec3 Lo = ShadeDirect(r, rec, lights, tris);

    // 2) perfect mirror reflection (simple recursion, we can expand this)
    // Use mat.kr as the weight (0..1). For metals you can set kr=1 and kd=0 etc.
    if (rec.mat.kr > 0.0f) {
        Vec3 refl = reflect_dir(unit_vector(r.direction()), N);
        Ray rr(rec.p + N * RT_EPS, refl);
        Vec3 bounced = TraceRay(rr, tris, lights, depth - 1);

        // Tint reflection by specularColor for colored metals (more of a beauty thing)
        Vec3 tint = rec.mat.specularColor;
        Lo = Lo + (rec.mat.kr * (tint * bounced));
    }

    return clamp(Lo);
}

#endif