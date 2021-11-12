//
// Project: image_demorphing
// File: DistortImage.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "DistortImage.hpp"
#include "MorphTransform.hpp"
#include "Utils.hpp"
#include <random>


DistortedImage distortImage(const cv::Mat& image, const DistortSettings& settings)
{
    static std::default_random_engine rnd(1507715517);

    DistortedImage distortedImage;

    std::vector<MorphTransform> transforms;
    int nTransforms = settings.nMinTransforms + rnd()%(settings.nMaxTransforms-settings.nMinTransforms+1);
    for (int i=0; i<nTransforms; ++i) {
        transforms.emplace_back(MorphTransform::randomTransform(
            settings.maxPosition,
            settings.minDistance,
            settings.maxDistance,
            settings.maxRotation,
            settings.minScale,
            settings.maxScale,
            settings.maxTranslation
        ));
    }

    distortedImage.distorted = image.clone();
    distortedImage.backwardMap = cv::Mat(image.rows, image.cols, CV_32FC2);
    distortedImage.forwardMap = cv::Mat(image.rows, image.cols, CV_32FC2);

    for (int j=0; j<distortedImage.distorted.rows; ++j) {
        auto* rDistorted = distortedImage.distorted.ptr<Vec3f>(j);
        auto* rBackward = distortedImage.backwardMap.ptr<Vec2f>(j);
        //auto* rForward = distortedImage.backwardMap.ptr<Vec2f>(j);
        for (int i=0; i<distortedImage.distorted.cols; ++i) {
            Vec2f p(i+0.5f, j+0.5f);

            Vec2f p2 = p;
            for (auto& t : transforms)
                p2 = t * p2;

            rDistorted[i] = sampleMatCubic<Vec3f>(image, p2);
            rBackward[i] << p2-p;
        }
    }

    return distortedImage;
}
