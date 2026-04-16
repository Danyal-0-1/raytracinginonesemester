#ifndef MATERIAL_H
#define MATERIAL_H

#include "vec3.h"

struct Material {
    // Diffuse (Lambert)
    Vec3  albedo        = make_vec3(0.8f, 0.8f, 0.8f);   // diffuse reflectance (rho)
    float kd            = 1.0f;                           // diffuse weight

    // Specular lobe (BRDF)
    Vec3  specularColor = make_vec3(0.04f, 0.04f, 0.04f); // specular tint
    float ks            = 0.0f;                            // specular weight
    float shininess     = 32.0f;                           // Blinn-Phong exponent

    // Reflectance / glass
    float kr            = 0.0f;
    float ior           = 1.0f;   // index of refraction (1.0 = opaque, 1.5 = glass)

    // Emission — nonzero makes this surface an area light (Mitsuba-style)
    Vec3  emission      = make_vec3(0.0f, 0.0f, 0.0f);

    // Texture mapping
    // Index into the global texture array. -1 means "no texture, use flat color".
    int   diffuseTexIdx = -1;   // albedo / diffuse map
    int   normalTexIdx  = -1;   // tangent-space normal map
    int   alphaTexIdx   = -1;   // alpha mask: texel R < 0.5 → cutout (skip hit)

    // UV tiling: UVs are multiplied by this before all texture lookups.
    // 1.0 = no tiling; 4.0 = texture repeats 4× in each axis.
    float uv_scale = 1.0f;
};

// Precomputed info for each emissive triangle, used for NEE sampling.
struct EmissiveTriInfo {
    int   triangleIdx;  // index into the global Triangle[] array
    Vec3  emission;     // Le copied from material
    float area;         // triangle surface area = 0.5 * |cross(e1, e2)|
    Vec3  normal;       // geometric face normal (outward)
};

#endif