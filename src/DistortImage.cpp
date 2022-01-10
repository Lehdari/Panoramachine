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
#include "KdTree.hpp"
#include "MathTypes.hpp"
#include <random>


DistortedImage distortImage(const cv::Mat& image, const DistortSettings& settings)
{
    static std::default_random_engine rnd(1507715517);

    DistortedImage distortedImage;

    std::vector<MorphTransform<double>> transforms;
    int nTransforms = settings.nMinTransforms + rnd()%(settings.nMaxTransforms-settings.nMinTransforms+1);
    for (int i=0; i<nTransforms; ++i) {
        transforms.emplace_back(MorphTransform<double>::randomTransform(
            Vec2d((double)image.cols, (double)image.rows),
            settings.minDistance,
            settings.maxDistance,
            settings.maxRotation,
            settings.minScale,
            settings.maxScale,
            settings.maxTranslation
        ));
    }

    cv::Mat distorted = image.clone();
    distortedImage.backwardMap = cv::Mat::zeros(image.rows, image.cols, CV_32FC2);
    distortedImage.forwardMap = cv::Mat::zeros(image.rows, image.cols, CV_32FC2);

    std::vector<Vec2d, Eigen::aligned_allocator<Vec2d>> forwardTargetVectors;
    forwardTargetVectors.reserve(image.cols*image.rows);
    KdTree<Vec2d, Vec2d> forwardPoints;

    #pragma omp parallel for schedule(static, 1)
    for (int j=0; j<distorted.rows; ++j) {
        auto* rDistorted = distorted.ptr<Vec3f>(j);
        auto* rBackward = distortedImage.backwardMap.ptr<Vec2f>(j);
        for (int i=0; i<distorted.cols; ++i) {
            Vec2d pTarget(i+0.5f, j+0.5f);

            Vec2d pSource = pTarget;
            for (auto& t : transforms)
                pSource = t * pSource;

            rDistorted[i] = sampleMatCubic<Vec3f>(image, pSource.cast<float>());
            rBackward[i] << (pSource-pTarget).cast<float>();

            #pragma omp critical
            {
                forwardTargetVectors.push_back(pTarget - pSource);
                forwardPoints.addPoint(pSource, &forwardTargetVectors.back());
            }
        }
    }

    forwardPoints.build();

    #pragma omp parallel for schedule(static, 1)
    for (int j=0; j<distortedImage.forwardMap.rows; ++j) {
        auto* rForward = distortedImage.forwardMap.ptr<Vec2f>(j);
        for (int i=0; i<distortedImage.forwardMap.cols; ++i) {
            Vec2d p(i+0.5f, j+0.5f);
            rForward[i] = forwardPoints.getNearest(p).second->cast<float>();
        }
    }

    distortedImage.distorted = Image<Vec3f>(std::move(distorted));

    return distortedImage;
}
