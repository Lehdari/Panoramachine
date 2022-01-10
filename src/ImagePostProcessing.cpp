//
// Project: panoramachine
// File: ImagePostProcessing.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "ImagePostProcessing.hpp"


void brightnessContrast(Image<Vec3f>& img, const Vec3f& brightness, const Vec3f& contrast)
{
    Vec3f contrast2 = contrast.cwiseProduct(contrast) + (Vec3f::Ones()*2.0f).cwiseProduct(contrast);
    Vec3f brightness2 = (Vec3f::Ones()*0.5f).cwiseProduct(brightness);

    for (auto& layer : img) {
        #pragma omp parallel for
        for (int j=0; j<layer.rows; ++j) {
            auto* r = layer.ptr<Vec3f>(j);
            for (int i=0; i<layer.cols; ++i) {
                r[i] = (Vec3f::Ones()*0.5f+((r[i]-Vec3f::Ones()*0.5f+brightness2)
                    .cwiseProduct(Vec3f::Ones()+contrast2))+brightness2)
                    .cwiseMax(Vec3f::Zero()).cwiseMin(Vec3f::Ones());
            }
        }
    }
}
