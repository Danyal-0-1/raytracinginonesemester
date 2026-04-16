#ifndef BRDF_H
#define BRDF_H

#include <cmath>
#include "vec3.h"
#include "material.h"
#include "texture.h"

HYBRID_FUNC inline float saturate_f(float x) {
    return (x < 0.f) ? 0.f : (x > 1.f ? 1.f : x);
}

HYBRID_FUNC inline float BlinnPhongSpecularPDF(const Vec3& N,
                                                const Vec3& wo,
                                                const Vec3& wi,
                                                float shininess)
{
    const Vec3 Hraw = wo + wi;
    const float hlen2 = dot(Hraw, Hraw);
    if (hlen2 <= 1e-12f) return 0.0f;
    const Vec3 H = Hraw * (1.0f / sqrtf(hlen2));

    const float NdotH = fmaxf(dot(normalize(N), H), 0.0f);
    const float VdotH = fmaxf(dot(wo, H), 0.0f);
    const float inv2Pi = 0.15915494309f;
    return (VdotH > 1e-6f)
        ? (shininess + 2.0f) * inv2Pi * powf(NdotH, shininess) * NdotH / (4.0f * VdotH)
        : 0.0f;
}

// ============================================================
// Apply texture lookups to a HitRecord's material.
// Call this after triangle intersection to replace flat albedo
// with texture-sampled values and optionally perturb the normal.
// ============================================================
HYBRID_FUNC inline void ApplyTextures(HitRecord& rec,
                                       const TextureData* textures,
                                       int numTextures)
{
    if (textures == nullptr || numTextures <= 0) return;

    // Apply UV tiling scale before all lookups
    const float su = rec.uv.x * rec.mat.uv_scale;
    const float sv = rec.uv.y * rec.mat.uv_scale;

    // Diffuse texture
    const int diffIdx = rec.mat.diffuseTexIdx;
    if (diffIdx >= 0 && diffIdx < numTextures && textures[diffIdx].data != nullptr) {
        rec.mat.albedo = textures[diffIdx].sample(su, sv);
    }

    // Alpha cutout: sample alpha mask and flag the hit for rejection if opaque < 50%
    const int alphaIdx = rec.mat.alphaTexIdx;
    if (alphaIdx >= 0 && alphaIdx < numTextures && textures[alphaIdx].data != nullptr) {
        const float alpha = textures[alphaIdx].sampleAlpha(su, sv);
        if (alpha < 0.5f) {
            rec.alpha_masked = true;
            return;  // skip normal map — hit will be discarded
        }
    }

    // Normal map
    const int normIdx = rec.mat.normalTexIdx;
    if (normIdx >= 0 && normIdx < numTextures && textures[normIdx].data != nullptr) {
        // Sample tangent-space normal
        Vec3 tsN = textures[normIdx].sampleNormal(su, sv);

        // Build TBN basis from surface normal and UV-derived tangent
        Vec3 N = normalize(rec.normal);
        Vec3 tangent, bitangent;

        // Use a stable tangent calculation
        Vec3 up = (fabsf(N.z) < 0.9f) ? make_vec3(0.0f, 0.0f, 1.0f)
                                       : make_vec3(1.0f, 0.0f, 0.0f);
        Vec3 cr = cross(up, N);
        float len = sqrtf(dot(cr, cr));
        tangent   = (len > 1e-10f) ? cr * (1.0f / len) : make_vec3(1.0f, 0.0f, 0.0f);
        bitangent = cross(N, tangent);

        // Transform tangent-space normal to world space
        Vec3 worldN = tangent * tsN.x + bitangent * tsN.y + N * tsN.z;
        float wlen = sqrtf(dot(worldN, worldN));
        if (wlen > 1e-10f) {
            rec.normal = worldN * (1.0f / wlen);
        }
    }
}


// Returns f(wo, wi) — does NOT include N·L
HYBRID_FUNC inline Vec3 EvaluateBRDF(const HitRecord& rec,
                                      const Vec3& V,   // to viewer
                                      const Vec3& L)    // to light
{
    const Material& m = rec.mat;
    const Vec3 N = rec.normal;
    const float NdotL = fmaxf(dot(N, L), 0.0f);
    const float NdotV = fmaxf(dot(N, V), 0.0f);
    if (NdotL <= 0.f || NdotV <= 0.f) return make_vec3(0,0,0);

    // Lambertian diffuse: rho/pi
    // NOTE: m.albedo may have been replaced by texture lookup via ApplyTextures()
    const float invPi = 0.31830988618f;
    Vec3 fd = m.albedo * (m.kd * invPi);

    // Blinn-Phong specular lobe
    Vec3 H = unit_vector(L + V);
    float NdotH = fmaxf(dot(N, H), 0.0f);
    const float inv2Pi = 0.15915494309f;
    float specNorm = (m.shininess + 2.0f) * inv2Pi;
    float specLobe = specNorm * powf(NdotH, m.shininess);
    Vec3 fs = m.specularColor * (m.ks) * specLobe;

    return fd + fs;
}

HYBRID_FUNC inline float BRDFpdf(const HitRecord& rec,
                                  const Vec3& wo,
                                  const Vec3& wi)
{
    const Material& m = rec.mat;
    const Vec3  N     = normalize(rec.normal);
    const float kd    = m.kd;
    const float ks    = m.ks;
    const float total = kd + ks;
    if (total <= 0.0f) return 0.0f;

    const float NdotWi  = fmaxf(dot(N, wi), 0.0f);
    const float invPi   = 0.31830988618f;
    const float pdf_diff = NdotWi * invPi;
    const float pdf_spec = BlinnPhongSpecularPDF(N, wo, wi, m.shininess);
    return (kd / total) * pdf_diff + (ks / total) * pdf_spec;
}

#endif