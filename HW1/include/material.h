#ifndef MATERIAL_H
#define MATERIAL_H

#include "vec3.h"

enum MaterialType { MAT_LAMBERTIAN, MAT_METAL };

struct Material {
    MaterialType type = MAT_LAMBERTIAN;

    // Diffuse (Lambert)
    Vec3  albedo = make_vec3(0.8f, 0.8f, 0.8f);   // diffuse reflectance (rho)
    float kd     = 1.0f;                          // diffuse weight

    // Specular lobe (BRDF)
    Vec3  specularColor = make_vec3(0.04f, 0.04f, 0.04f);   // specular tint
    float ks            = 0.0f;                             // specular weight
    float shininess     = 32.0f;                            // Blinn-Phong exponent

    // Reflectance
    float kr            = 0.0f;

    // Emission (we will need to update this later, basic placeholder for now)
    Vec3  emission      = make_vec3(0.0f, 0.0f, 0.0f);
};

#endif
