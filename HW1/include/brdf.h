// brdf.h
#ifndef BRDF_H
#define BRDF_H

#include <cmath>
#include "vec3.h"
#include "material.h"

HYBRID_FUNC inline float saturate(float x) { return (x < 0.f) ? 0.f : (x > 1.f ? 1.f : x); }

// Returns f(wo, wi) (does NOT include N·L)
HYBRID_FUNC inline Vec3 EvaluateBRDF(const Material& m,
                                    const Vec3& N,
                                    const Vec3& V,  // to viewer
                                    const Vec3& L)  // to light
{
    const float NdotL = fmaxf(dot(N, L), 0.0f);
    const float NdotV = fmaxf(dot(N, V), 0.0f);
    if (NdotL <= 0.f || NdotV <= 0.f) return make_vec3(0,0,0);

    // Lambertian diffuse: rho/pi
    const float invPi = 0.31830988618f;
    Vec3 fd = m.albedo * (m.kd * invPi);

    // Blinn-Phong specular lobe (simple for now, will update)
    Vec3 H = unit_vector(L + V);
    float NdotH = fmaxf(dot(N, H), 0.0f);

    // Normalized Blinn-Phong: (n+2)/(2π) * (N·H)^n
    const float inv2Pi = 0.15915494309f;
    float specNorm = (m.shininess + 2.0f) * inv2Pi;
    float specLobe = specNorm * powf(NdotH, m.shininess);

    Vec3 fs = m.specularColor * (m.ks * specLobe);

    return fd + fs;
}

#endif
