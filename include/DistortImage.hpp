//
// Project: panoramachine
// File: DistortImage.hpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#ifndef PANORAMACHINE_DISTORTIMAGE_HPP
#define PANORAMACHINE_DISTORTIMAGE_HPP


#include <opencv2/core/mat.hpp>

#include "MathTypes.hpp"
#include "Image.hpp"


struct DistortSettings {
    // Transform settings
    int nMinTransforms;
    int nMaxTransforms;

    double  minDistance;
    double  maxDistance;
    double  maxRotation;
    double  minScale;
    double  maxScale;
    Vec2d   maxTranslation;
};

struct DistortedImage {
    Image<Vec3f>    distorted;
    cv::Mat         backwardMap;
    cv::Mat         forwardMap;
};


DistortedImage distortImage(const cv::Mat& image, const DistortSettings& settings);


#endif //PANORAMACHINE_DISTORTIMAGE_HPP
