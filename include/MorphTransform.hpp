//
// Project: image_demorphing
// File: MorphTransform.hpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#ifndef IMAGE_DEMORPHING_MORPHTRANSFORM_HPP
#define IMAGE_DEMORPHING_MORPHTRANSFORM_HPP


#include "MathTypes.hpp"
#include <random>


template <typename T_Scalar>
class MorphTransform {
public:
    using Vec2 = Eigen::Matrix<T_Scalar, 2, 1>;
    using Vec3 = Eigen::Matrix<T_Scalar, 3, 1>;
    using Mat3 = Eigen::Matrix<T_Scalar, 3, 3>;

    MorphTransform(
        const Vec2& position,
        T_Scalar distance = 0.0,
        T_Scalar rotation = 0.0,
        T_Scalar scale = 1.0,
        const Vec2& translation = Vec2(0.0f, 0.0f));

    Vec2 operator*(const Vec2& v);

    static MorphTransform randomTransform(
        const Vec2& maxPosition,
        T_Scalar minDistance, T_Scalar maxDistance,
        T_Scalar maxRotation,
        T_Scalar minScale, T_Scalar maxScale,
        const Vec2& maxTranslation);

private:
    Mat3    _mat;
    Vec2    _p;
    T_Scalar   _d2;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


#define RND ((rnd()%1000001) * 0.000001)


template <typename T_Scalar>
MorphTransform<T_Scalar>::MorphTransform(
    const Vec2& position,
    T_Scalar distance,
    T_Scalar rotation,
    T_Scalar scale,
    const Vec2& translation
) :
    _p  (position),
    _d2 (distance*distance/9.0f)
{
    Mat3 m1, m2, t, r, s;
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

template <typename T_Scalar>
typename MorphTransform<T_Scalar>::Vec2 MorphTransform<T_Scalar>::operator*(const Vec2& v)
{
    Vec3 vv;
    vv << v, 1.0f;
    Vec2 v2 = (_mat * vv).template block<2,1>(0,0);

    // transformation filtering
    T_Scalar f = std::exp( -0.5f * ((v-_p).squaredNorm() / _d2) );

    return v + f*(v2-v);
}

template <typename T_Scalar>
MorphTransform<T_Scalar> MorphTransform<T_Scalar>::randomTransform(
    const Vec2& maxPosition,
    T_Scalar minDistance, T_Scalar maxDistance,
    T_Scalar maxRotation,
    T_Scalar minScale, T_Scalar maxScale,
    const Vec2& maxTranslation)
{
    static std::default_random_engine rnd(1507715517);
    return MorphTransform(Vec2(RND*maxPosition(0), RND*maxPosition(1)),
        minDistance + RND*(maxDistance-minDistance),
        maxRotation - 2.0*RND*maxRotation,
        minScale + RND*(maxScale-minScale),
        maxTranslation - 2.0*RND*maxTranslation);
}


#undef RND


#endif //IMAGE_DEMORPHING_MORPHTRANSFORM_HPP
