#ifndef SHADER_H
#define SHADER_H

#include <cmath>
#include <limits>

#include "ray.h"
#include "brdf.h"
#include "MeshOBJ.h"
#include "scene.h"
#include "bvh.h"
#include "medium.h"
#include "query.h"

HYBRID_FUNC inline void SearchBVH(
    const int numTriangles,
    const Ray& ray,
    const BVHNode* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    const Triangle* __restrict__ triangles,
    HitRecord& hitRecord);

static constexpr float RT_EPS = 1e-3f;

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
    return I - (2.0f * dot(I, N)) * N;
}

// Snell's law refraction.  eta = n_incident / n_transmitted.
// N must point from the surface into the incident medium.
// Returns refracted direction (unit length), or zero-vector if total internal reflection.
HYBRID_FUNC inline Vec3 refract_dir(const Vec3& I, const Vec3& N, float eta) {
    const float cos_i   = fminf(-dot(I, N), 1.0f);
    const float sin2_t  = eta * eta * fmaxf(0.0f, 1.0f - cos_i * cos_i);
    if (sin2_t >= 1.0f) return make_vec3(0.0f, 0.0f, 0.0f);  // total internal reflection
    const float cos_t   = sqrtf(1.0f - sin2_t);
    return I * eta + N * (eta * cos_i - cos_t);
}

// Schlick approximation for Fresnel reflectance at a dielectric interface.
// cos_theta is the angle between the incoming ray and the surface normal (positive).
HYBRID_FUNC inline float schlick(float cos_theta, float ior) {
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cos_theta, 5.0f);
}

HYBRID_FUNC inline float GeometryTerm(const Vec3& hitPoint,
                                       const Vec3& lightPoint,
                                       const Vec3& lightNormal)
{
    Vec3  wi  = lightPoint - hitPoint;
    float r2  = dot(wi, wi);
    if (r2 < 1e-10f) return 0.0f;
    Vec3  wid = wi * (1.0f / sqrtf(r2));
    return fabsf(dot(lightNormal, -wid)) / r2;
}

HYBRID_FUNC inline bool IsInShadow(const Vec3& P,
                                    const Vec3& N,
                                    const Light& light,
                                    const Triangle* tris,
                                    int triCount,
                                    const BVHNode* nodes,
                                    const AABB* aabbs)
{
    if (light.type == 2) {
        // Directional (sun): shadow ray in the toward-light direction, no distance limit
        Ray shadowRay(P + N * RT_EPS, light.direction);
        HitRecord shadowHit{};
        SearchBVH(triCount, shadowRay, nodes, aabbs, tris, shadowHit);
        return shadowHit.hit;
    }
    Vec3 toL = light.position - P;
    float distToL = length3(toL);
    if (distToL <= 0.0f) return false;
    Vec3 Ldir = toL / distToL;
    Ray shadowRay(P + N * RT_EPS, Ldir);
    HitRecord shadowHit{};
    SearchBVH(triCount, shadowRay, nodes, aabbs, tris, shadowHit);
    return shadowHit.hit && shadowHit.t < distToL;
}

// ============================================================
// ShadeDirect — now accepts per-object medium for transmittance
// ============================================================
HYBRID_FUNC inline Vec3 ShadeDirect(const Ray& r,
                                     const HitRecord& rec,
                                     const Light* lights,
                                     const int numLights,
                                     const int numTriangles,
                                     const BVHNode* nodes,
                                     const AABB* aabbs,
                                     const Triangle* triangles,
                                     const HomogeneousMedium* objectMedia = nullptr,
                                     int numObjectMedia = 0,
                                     const int32_t* triObjectIds = nullptr)
{
    Vec3 N = unit_vector(rec.normal);
    Vec3 V = unit_vector(r.origin() - rec.p);
    Vec3 Lo = make_vec3(0,0,0);

    for (int i = 0; i < numLights; ++i) {
        const Light& light = lights[i];
        Vec3 L = (light.type == 2)
                     ? light.direction                        // directional: fixed sun dir
                     : unit_vector(light.position - rec.p);  // point/area: toward light
        float NdotL = fmaxf(dot(N, L), 0.0f);
        if (NdotL <= 0.0f) continue;

        if (IsInShadow(rec.p, N, light, triangles, numTriangles, nodes, aabbs))
            continue;

        Vec3 f = EvaluateBRDF(rec, V, L);
        // Directional lights have no distance falloff
        Vec3 radiance = light.color * light.intensity;

        // Apply medium transmittance along shadow ray if applicable
        Vec3 Tr = make_vec3(1.0f, 1.0f, 1.0f);
        if (objectMedia != nullptr && triObjectIds != nullptr &&
            rec.triangleIdx >= 0 && rec.triangleIdx < numTriangles) {
            int objId = triObjectIds[rec.triangleIdx];
            if (objId >= 0 && objId < numObjectMedia && objectMedia[objId].enabled) {
                float dist = length3(light.position - rec.p);
                Tr = objectMedia[objId].transmittance(dist);
            }
        }

        Vec3 direct = (radiance * f) * Tr * NdotL;
        Lo = Lo + direct;
    }
    return Lo;
}

#endif
