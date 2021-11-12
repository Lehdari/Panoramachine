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


class MorphTransform {
public:
    MorphTransform(
        const Vec2f& position,
        float distance = 0.0f,
        float rotation = 0.0f,
        float scale = 1.0f,
        const Vec2f& translation = Vec2f(0.0f, 0.0f));

    Vec2f operator*(const Vec2f& v);

    static MorphTransform randomTransform(
        const Vec2f& maxPosition,
        float minDistance, float maxDistance,
        float maxRotation,
        float minScale, float maxScale,
        const Vec2f& maxTranslation);

private:
    Mat3f   _mat;
    Vec2f   _p;
    float   _d2;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


#endif //IMAGE_DEMORPHING_MORPHTRANSFORM_HPP
