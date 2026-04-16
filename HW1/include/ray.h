#ifndef RAY_H
#define RAY_H
#include "vec3.h"
#include "material.h"
#include <cmath>
#include <limits>

class Ray {
    using point3 = Vec3;
public:
    Ray() = default;

    Ray(const point3& origin, const Vec3& direction)
        : orig(origin), dir(unit_vector(direction)) {}

    const point3& origin() const { return orig; }
    const Vec3& direction() const { return dir; }

    point3 at(double t) const { return orig + float(t) * dir; }

private:
    point3 orig{};
    Vec3   dir{};
};

struct Triangle {
    Vec3 v0{}, v1{}, v2{};
    Vec3 n0{}, n1{}, n2{};

    // attach material to primitive so render.cpp can set it
    Material mat{};
};

struct HitRecord {
    bool hit = false;
    Vec3 p{};
    Vec3 normal{};
    double t = 0.0;
    bool front_face = true;
    Material mat{};

    void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

inline HitRecord ray_intersection(const Ray& r, const Triangle& tri) {
    HitRecord rec{};
    const float eps = std::numeric_limits<float>::epsilon();

    Vec3 edge_vec_1 = tri.v1 - tri.v0;
    Vec3 edge_vec_2 = tri.v2 - tri.v0;

    Vec3 pvec = cross(r.direction(), edge_vec_2);
    float det = dot(pvec, edge_vec_1);

    if (std::abs(det) < eps) return rec;

    float invDet = 1.0f / det;

    Vec3 tvec = r.origin() - tri.v0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return rec;

    Vec3 qvec = cross(tvec, edge_vec_1);
    float v = dot(r.direction(), qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) return rec;

    float t = dot(edge_vec_2, qvec) * invDet;
    if (t < 0.0f) return rec;

    rec.hit = true;
    rec.t   = t;
    rec.p   = r.at(t);

    // Interpolated normal (normalized!)
    rec.normal = unit_vector((1.0f - u - v) * tri.n0 + u * tri.n1 + v * tri.n2);

    // Pull material from triangle (temporary just for testing BRDF)
    rec.mat = tri.mat;

    return rec;
}

#endif
