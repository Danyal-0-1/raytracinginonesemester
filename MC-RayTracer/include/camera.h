#ifndef CAM_H
#define CAM_H

#include "vec3.h"
#include "ray.h"
#include <cmath>

class Camera {
  public:
    using vec3 = Vec3;
    using point3 = Vec3;

    HYBRID_FUNC Camera(point3 pos = make_vec3(0, 0, 0),
           point3 lookAt = make_vec3(0, 1, 0),
           vec3 up = make_vec3(0, 0, 1),
           double focal_length_mm = 50.0,
           double sensor_height_mm = 24.0,
           int width = 100,
           int height = 100)
        : center(pos),
          look_at(lookAt),
          up_vector(up),
          focal_length_mm(focal_length_mm),
          sensor_height_mm(sensor_height_mm),
          pixel_width(width),
          pixel_height(height) {
        initialize();
    }

    int pixel_width = 100;
    int pixel_height = 100;

    HYBRID_FUNC point3 get_center() const { return center; }
    HYBRID_FUNC point3 get_look_at() const { return look_at; }
    HYBRID_FUNC vec3 get_up_vector() const { return up_vector; }
    HYBRID_FUNC double get_focal_length_mm() const { return focal_length_mm; }
    HYBRID_FUNC double get_sensor_height_mm() const { return sensor_height_mm; }

    HYBRID_FUNC point3 get_pixel_position(int i, int j) const {
        return pixel00_loc + (double(i) * pixel_delta_u) + (double(j) * pixel_delta_v);
    }

    HYBRID_FUNC Ray get_ray(int i, int j) const {
        point3 pixel = get_pixel_position(i, j);
        vec3 dir = unit_vector(pixel - center);
        return Ray(center, dir);
    }

    HYBRID_FUNC Ray get_ray(float i, float j) const {
        point3 pixel = pixel00_loc + (double(i) * pixel_delta_u) + (double(j) * pixel_delta_v);
        vec3 dir = unit_vector(pixel - center);
        return Ray(center, dir);
    }

    // Forward direction (camera looks toward look_at).
    HYBRID_FUNC vec3 get_forward() const { return unit_vector(look_at - center); }

    // Project a world-space point onto the image plane (pinhole).
    // Returns true if the point is in front of the camera AND lands inside the
    // [0,W) x [0,H) image rectangle. Writes sub-pixel float coords (px, py) and
    // cos_theta = forward · unit(p - center). Used by BDPT t=1 light tracing splat.
    HYBRID_FUNC bool world_to_pixel(const point3& p, float& px, float& py, float& cos_theta) const {
        const vec3  d        = p - center;
        const vec3  forward  = get_forward();
        const float dot_fd   = float(dot(forward, d));
        const float dlen2    = float(dot(d, d));
        if (dot_fd <= 1e-6f || dlen2 < 1e-24f) {
            cos_theta = 0.0f;
            return false;
        }
        const float dlen = sqrtf(dlen2);
        cos_theta = dot_fd / dlen;

        const float fl_m = float(focal_length_mm / 1000.0);
        const vec3 hit_world = center + d * (fl_m / dot_fd);

        const vec3 corner = pixel00_loc - 0.5 * (pixel_delta_u + pixel_delta_v);
        const vec3 from_corner = hit_world - corner;

        const float du2 = float(dot(pixel_delta_u, pixel_delta_u));
        const float dv2 = float(dot(pixel_delta_v, pixel_delta_v));
        if (du2 <= 0.0f || dv2 <= 0.0f) return false;
        px = float(dot(from_corner, pixel_delta_u)) / du2;
        py = float(dot(from_corner, pixel_delta_v)) / dv2;

        return (px >= 0.0f && px < float(pixel_width)
             && py >= 0.0f && py < float(pixel_height));
    }

    // Image-plane area projected to z=1 (camera-coord depth 1). Used in the
    // pinhole We = 1 / (A_z1 * cos⁴θ) for BDPT t=1 splatting.
    HYBRID_FUNC float image_area_at_z1() const {
        const float du = sqrtf(float(dot(pixel_delta_u, pixel_delta_u)));
        const float dv = sqrtf(float(dot(pixel_delta_v, pixel_delta_v)));
        const float fl_m = float(focal_length_mm / 1000.0);
        if (fl_m <= 0.0f) return 0.0f;
        return du * dv * float(pixel_width) * float(pixel_height) / (fl_m * fl_m);
    }

  private:
    point3 center;
    point3 look_at;
    vec3 up_vector;
    double focal_length_mm;
    double sensor_height_mm;
    point3 pixel00_loc;
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;

    HYBRID_FUNC static vec3 unit_vector(const vec3& v, const vec3& fallback = make_vec3(0.0f, 0.0f, 1.0f)) {
        float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
        const double eps = 1e-12;
        if (len < eps) return fallback;
        return v / len;
    }

    HYBRID_FUNC void initialize() {
        if (pixel_width < 1) pixel_width = 1;
        if (pixel_height < 1) pixel_height = 1;

        vec3 forward = unit_vector(look_at - center);
        vec3 right = unit_vector(cross(forward, up_vector));
        vec3 up_corrected = cross(right, forward);

        double focal_length_m = focal_length_mm / 1000.0;
        double sensor_height_m = sensor_height_mm / 1000.0;

        double viewport_height = sensor_height_m;
        double viewport_width = viewport_height * (double(pixel_width) / double(pixel_height));

        vec3 viewport_u = viewport_width * right;
        vec3 viewport_v = -viewport_height * up_corrected;
        pixel_delta_u = viewport_u / double(pixel_width);
        pixel_delta_v = viewport_v / double(pixel_height);

        point3 viewport_center = center + focal_length_m * forward;
        point3 viewport_upper_left = viewport_center - (viewport_u * 0.5) - (viewport_v * 0.5);
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }
};

#endif
