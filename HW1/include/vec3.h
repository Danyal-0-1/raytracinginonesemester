#ifndef VEC3_H
#define VEC3_H

#include <iostream>
#include <cmath>

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define HYBRID_FUNC __host__ __device__
    using Vec3 = float3; // Use CUDA's float3
#else
    #define HYBRID_FUNC
    struct float3_cpu { float x, y, z; };
    using Vec3 = float3_cpu; // Use custom CPU vector
#endif


HYBRID_FUNC inline Vec3 make_vec3(float x, float y, float z) {
    Vec3 v; v.x = x; v.y = y; v.z = z; return v;
}
HYBRID_FUNC inline Vec3 vec3(float x, float y, float z) {
    Vec3 v = make_vec3(x, y, z);
    return v;
};
HYBRID_FUNC inline Vec3 vec3(int x, int y, int z) {
    Vec3 v = make_vec3(float(x), float(y), float(z));
    return v;
};
HYBRID_FUNC inline Vec3 point3(float x, float y, float z) {
    Vec3 v = make_vec3(x, y ,z);
    return v;
};
HYBRID_FUNC inline Vec3 point3(int x, int y, int z) {
    Vec3 v = make_vec3(float(x), float(y), float(z));
    return v;
};

HYBRID_FUNC inline Vec3 operator+(const Vec3& a, const Vec3& b) { return make_vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
HYBRID_FUNC inline Vec3 operator-(const Vec3& a, const Vec3& b) { return make_vec3(a.x-b.x, a.y-b.y, a.z-b.z); }
HYBRID_FUNC inline Vec3 operator-(const Vec3& a)                { return make_vec3(-a.x, -a.y, -a.z); }
HYBRID_FUNC inline Vec3 operator*(const Vec3& a, const Vec3& b) { return make_vec3(a.x*b.x, a.y*b.y, a.z*b.z); }
HYBRID_FUNC inline Vec3 operator*(const Vec3& v, float t)       { return make_vec3(v.x*t, v.y*t, v.z*t); }
HYBRID_FUNC inline Vec3 operator*(float t, const Vec3& v)       { return make_vec3(v.x*t, v.y*t, v.z*t); }
HYBRID_FUNC inline Vec3 operator/(const Vec3& a, const Vec3& b) { return make_vec3(a.x/b.x, a.y/b.y, a.z/b.z); }
HYBRID_FUNC inline Vec3 operator/(const Vec3& a, double t)       { return make_vec3(a.x/t, a.y/t, a.z/t); }
HYBRID_FUNC inline float dot(const Vec3& u, const Vec3& v)      { return u.x*v.x + u.y*v.y + u.z*v.z; }
HYBRID_FUNC inline Vec3 cross(const Vec3& u, const Vec3& v) {
    return make_vec3(u.y * v.z - u.z * v.y,
                     u.z * v.x - u.x * v.z,
                     u.x * v.y - u.y * v.x);
}

HYBRID_FUNC inline Vec3 unit_vector(Vec3 v) {
    float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if (len < 1e-12f) return make_vec3(0,0,0);
    return make_vec3(v.x/len, v.y/len, v.z/len);
}


inline void PrintVec3(const Vec3& v) { std::cout << "(" << v.x << ", " << v.y << ", " << v.z << ")"; }

#endif