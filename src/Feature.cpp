//
// Project: image_demorphing
// File: Feature.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "Feature.hpp"
#include "Utils.hpp"

#include <opencv2/highgui.hpp>


Feature::Feature() :
    polar   (Eigen::Matrix<float, fsa*6, fsr>::Zero()),
    energy  (0.0)
{
}

Feature::Feature(const cv::Mat& img, const Vec2f& p, float firstRadius)
{
    for (int i=0; i<Feature::fsr; ++i) {
        float angle = 2.0f*M_PI*(i/(float)Feature::fsr);
        float r = firstRadius;
        Vec2f dir(std::cos(angle), std::sin(angle));
        for (int j=0; j<Feature::fsa; ++j) {
            polar.block<3,1>(j*3,i) = sampleMatCubic<Vec3f>(img, p+dir*r);
            r *= Feature::frm;
        }
    }

    float avg = polar.block<fsa*3, fsr>(0,0).sum() / (fsa*3*fsr);
    polar.block<fsa*3, fsr>(0,0).noalias() -= avg * Eigen::Matrix<float, fsa*3, fsr>::Ones();
    energy = std::sqrt((double)polar.block<fsa*3, fsr>(0,0).array().square().sum() / (fsa*3*fsr));
    float sdInv = 1.0f / energy;
    polar.block<fsa*3, fsr>(0,0) *= sdInv;

    for (int i=0; i<Feature::fsr; ++i) {
        polar.block<Feature::fsa*3, 1>(fsa*3, i) =
            polar.block<Feature::fsa*3, 1>(0, (i+1)%Feature::fsr) -
                polar.block<Feature::fsa*3, 1>(0, i);
    }
}


void visualizeFeature(Feature& feature, const std::string& windowName)
{
    cv::Mat featureImg(Feature::fsr, Feature::fsa*2, CV_32FC3);
    featureImg.data = reinterpret_cast<unsigned char*>(feature.polar.data());
    cv::imshow(windowName, featureImg);
}
