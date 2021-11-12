//
// Project: image_demorphing
// File: DistortImage.hpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#ifndef IMAGE_DEMORPHING_DISTORTIMAGE_HPP
#define IMAGE_DEMORPHING_DISTORTIMAGE_HPP


#include <opencv2/core/mat.hpp>
#include "MathTypes.hpp"


struct DistortSettings {
    // Transform settings
    int nMinTransforms;
    int nMaxTransforms;

    Vec2d   maxPosition;
    double  minDistance;
    double  maxDistance;
    double  maxRotation;
    double  minScale;
    double  maxScale;
    Vec2d   maxTranslation;
};

struct DistortedImage {
    cv::Mat distorted;
    cv::Mat backwardMap;
    cv::Mat forwardMap;
};


DistortedImage distortImage(const cv::Mat& image, const DistortSettings& settings);


#endif //IMAGE_DEMORPHING_DISTORTIMAGE_HPP
