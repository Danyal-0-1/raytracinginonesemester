#pragma once

#include "vec2.h"
#include "vec3.h"

using Vec2f = Vec2;
using Vec3f = Vec3;

class ILight {
public:
    virtual ~ILight() = default;

    virtual Vec3f Sample(const Vec3f& normal, const Vec2f& u, float& pdf_out) const = 0;
    virtual float Pdf(const Vec3f& wi) const = 0;
    virtual Vec3f Eval(const Vec3f& wi) const = 0;
    virtual bool IsDelta() const = 0;
};

