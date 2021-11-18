//
// Project: image_demorphing
// File: Feature.hpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#ifndef IMAGE_DEMORPHING_FEATURE_HPP
#define IMAGE_DEMORPHING_FEATURE_HPP


#include <opencv2/core/mat.hpp>

#include "MathTypes.hpp"


struct Feature {
    static constexpr int fsa = 32; // feature axial size
    static constexpr int fsr = 128; // feature radial size
    static constexpr float frm = 1.1387886347566f; // feature radius multiplier

    Eigen::Matrix<float, fsa*6, fsr>    polar;

    Feature();
    Feature(const cv::Mat& img, const Vec2f& p, float firstRadius);
};


void visualizeFeature(Feature& feature, const std::string& windowName);


#endif //IMAGE_DEMORPHING_FEATURE_HPP
