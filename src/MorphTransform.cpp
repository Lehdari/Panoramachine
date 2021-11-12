//
// Project: image_demorphing
// File: MorphTransform.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "MorphTransform.hpp"
#include <random>


#define RND ((rnd()%1000001) * 0.000001)


MorphTransform::MorphTransform(
    const Vec2f& position,
    float distance,
    float rotation,
    float scale,
    const Vec2f& translation
) :
    _p  (position),
    _d2 (distance*distance/9.0f)
{
    Mat3f   m1, m2, t, r, s;
    m1  <<  1.0f,   0.0f,   -position(0),
            0.0f,   1.0f,   -position(1),
            0.0f,   0.0f,   1.0f;
    m2  <<  1.0f,   0.0f,   position(0),
            0.0f,   1.0f,   position(1),
            0.0f,   0.0f,   1.0f;
    t   <<  1.0f,   0.0f,   translation(0),
            0.0f,   1.0f,   translation(1),
            0.0f,   0.0f,   1.0f;
    r   <<  std::cos(rotation), -std::sin(rotation),    0.0f,
            std::sin(rotation), std::cos(rotation),     0.0f,
            0.0f,               0.0f,                   1.0f;
    s   <<  scale,  0.0f,   0.0f,
            0.0f,   scale,  0.0f,
            0.0f,   0.0f,   1.0f;

    _mat = (m2 * t * r * s * m1).inverse();
}

Vec2f MorphTransform::operator*(const Vec2f& v)
{
    Vec3f vv;
    vv << v, 1.0f;
    Vec2f v2 = (_mat * vv).block<2,1>(0,0);

    // transformation filtering
    float f = std::exp( -0.5f * ((v-_p).squaredNorm() / _d2) );

    return v + f*(v2-v);
}

MorphTransform MorphTransform::randomTransform(
    const Vec2f& maxPosition,
    float minDistance, float maxDistance,
    float maxRotation,
    float minScale, float maxScale,
    const Vec2f& maxTranslation)
{
    static std::default_random_engine rnd(1507715517);
    return MorphTransform(Vec2f(RND*maxPosition(0), RND*maxPosition(1)),
        minDistance + RND*(maxDistance-minDistance),
        maxRotation - 2.0*RND*maxRotation,
        minScale + RND*(maxScale-minScale),
        maxTranslation - 2.0*RND*maxTranslation);
}
