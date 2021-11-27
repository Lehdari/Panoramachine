//
// Project: image_demorphing
// File: ImagePostProcessing.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ImagePostProcessing.hpp"


void brightnessContrast(cv::Mat& img, const Vec3f& brightness, const Vec3f& contrast)
{
    #pragma omp parallel for
    for (int j=0; j<img.rows; ++j) {
        auto* r = img.ptr<Vec3f>(j);
        for (int i=0; i<img.cols; ++i) {
            r[i] = (r[i].cwiseProduct(Vec3f::Ones()+contrast)+brightness)
                .cwiseMax(Vec3f::Zero()).cwiseMin(Vec3f::Ones());
        }
    }
}
