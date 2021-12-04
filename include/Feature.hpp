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
#include "Image.hpp"


struct Feature {
    static constexpr int fsa = 32; // feature axial size (in pixels)
    static constexpr int fsr = 128; // feature radial size (in pixels)
    static constexpr double frm = 1.1387886347566; // feature radius multiplier
    static constexpr double fmr = 64.0; // max radius ratio to firstRadius, frm^fsa

    Eigen::Matrix<float, fsa*6, fsr>    polar;
    double                              energy;
    Vec2f                               p;
    float                               firstRadius;

    Feature();
    Feature(const cv::Mat& img, const Vec2f& p, float firstRadius, float rotation=0.0f);
    Feature(const Image<Vec3f>& img, const Vec2f& p, float firstRadius, float rotation=0.0f);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    void computeDiffAndEnergy();
};


void visualizeFeature(Feature& feature, const std::string& windowName, int scale=1);


#endif //IMAGE_DEMORPHING_FEATURE_HPP
