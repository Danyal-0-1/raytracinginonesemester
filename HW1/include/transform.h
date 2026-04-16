#pragma once

#include "MeshOBJ.h"
#include "vec3.h"

#include <cmath>

// Local-to-world transform for each mesh (default unit: meters; rotation unit: degrees)
// scale -> rotate(X,Y,Z) -> translate
// i.e., p' = T + Rz(Ry(Rx(p * S)))
struct Transform {
    Vec3 position = make_vec3(0.0f, 0.0f, 0.0f);     // translate
    Vec3 rotation_deg = make_vec3(0.0f, 0.0f, 0.0f); // Euler angles in degrees: x,y,z
    Vec3 scale = make_vec3(1.0f, 1.0f, 1.0f);        // non-uniform scale
};

inline float deg_to_rad(float deg) {
    constexpr float kPi = 3.14159265358979323846f;
    return deg * (kPi / 180.0f);
}

inline Vec3 rotate_x_rad(const Vec3& v, float a) {
    const float c = std::cos(a);
    const float s = std::sin(a);
    return make_vec3(v.x, c * v.y - s * v.z, s * v.y + c * v.z);
}

inline Vec3 rotate_y_rad(const Vec3& v, float a) {
    const float c = std::cos(a);
    const float s = std::sin(a);
    return make_vec3(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

inline Vec3 rotate_z_rad(const Vec3& v, float a) {
    const float c = std::cos(a);
    const float s = std::sin(a);
    return make_vec3(c * v.x - s * v.y, s * v.x + c * v.y, v.z);
}

inline Vec3 rotate_xyz_deg(const Vec3& v, const Vec3& rotDeg) {
    Vec3 out = v;
    out = rotate_x_rad(out, deg_to_rad(rotDeg.x));
    out = rotate_y_rad(out, deg_to_rad(rotDeg.y));
    out = rotate_z_rad(out, deg_to_rad(rotDeg.z));
    return out;
}

inline float safe_inv(float x) {
    const float ax = std::fabs(x);
    if (ax < 1e-12f) return 0.0f;
    return 1.0f / x;
}

// Apply full transformation to a point (including translation)
inline Vec3 transform_point(const Transform& t, const Vec3& p) {
    Vec3 s = p * t.scale;
    Vec3 r = rotate_xyz_deg(s, t.rotation_deg);
    return r + t.position;
}

// Apply linear part to a vector (excluding translation)
inline Vec3 transform_vector(const Transform& t, const Vec3& v) {
    Vec3 s = v * t.scale;
    return rotate_xyz_deg(s, t.rotation_deg);
}

// Normal uses inverse-transpose(linear):
// linear = R * S  => normal' = normalize(R * (n / S))
inline Vec3 transform_normal(const Transform& t, const Vec3& n) {
    Vec3 invS = make_vec3(safe_inv(t.scale.x), safe_inv(t.scale.y), safe_inv(t.scale.z));
    Vec3 scaled = n * invS;
    Vec3 rotated = rotate_xyz_deg(scaled, t.rotation_deg);
    return unit_vector(rotated);
}

inline void ApplyTransformToMeshSOA(MeshSOA& mesh, const Transform& t) {
    for (auto& p : mesh.positions) {
        p = transform_point(t, p);
    }
    if (!mesh.normals.empty()) {
        for (auto& n : mesh.normals) {
            n = transform_normal(t, n);
        }
    }
}