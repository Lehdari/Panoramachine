//
// Project: image_demorphing
// File: StitchImages.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "StitchImages.hpp"
#include "Feature.hpp"
#include "FeatureDetector.hpp"

#include <random>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#define RND ((rnd()%1000001)*0.000001)


cv::Mat stitchImages(const std::vector<cv::Mat>& images)
{
    constexpr int nFeatures = 100;

    std::default_random_engine rnd(1507715517);
    FeatureDetector<OptimizerStatic> detector;
    detector.loadWeights("../feature_detector_model");

    cv::Mat img1scaled, img2scaled, combinedScaled, stitch;
    cv::resize(images[0], img1scaled, cv::Size(0,0), 1.0/4.0, 1.0/4.0, cv::INTER_LINEAR);
    cv::resize(images[1], img2scaled, cv::Size(0,0), 1.0/4.0, 1.0/4.0, cv::INTER_LINEAR);
    gammaCorrect(img1scaled, 1.0f/2.2f);
    gammaCorrect(img2scaled, 1.0f/2.2f);
    cv::hconcat(img1scaled, img2scaled, combinedScaled);

    std::vector<Feature> features1;
    features1.reserve(nFeatures);
    std::vector<Feature> features2;
    features2.reserve(nFeatures);

    for (int i=0; i<nFeatures; ++i) {
        features1.emplace_back(images[0], Vec2f(RND*(images[0].cols-1.0e-8f), RND*(images[0].rows-1.0e-8f)), 2.0f);
        features2.emplace_back(images[1], Vec2f(RND*(images[1].cols-1.0e-8f), RND*(images[1].rows-1.0e-8f)), 2.0f);
    }

    stitch = combinedScaled.clone();

    cv::Mat featureDistances(nFeatures, nFeatures, CV_32F);
    for (int j=0; j<nFeatures; ++j) {
        auto* r = featureDistances.ptr<float>(j);
        for (int i=0; i<nFeatures; ++i) {
            r[i] = detector(features1[j], features2[i]);
        }
    }

    for (int j=0; j<nFeatures; ++j) {
        auto* r = featureDistances.ptr<float>(j);
        float minDistance = std::numeric_limits<float>::max();
        int minId = 0;
        for (int i=0; i<nFeatures; ++i) {
            if (r[i] < minDistance) {
                minDistance = r[i];
                minId = i;
            }
        }

        if (minDistance > 0.25f)
            continue;

        auto& p1 = features1[j].p;
        auto& p2 = features2[minId].p;
        cv::line(stitch,
            cv::Point(p1(0)/4.0, p1(1)/4.0),
            cv::Point(p1(0)/4.0+images[0].cols/4, p2(1)/4.0),
            cv::Scalar(1.0, 1.0, 1.0));
    }

    cv::imshow("stitch", stitch);
    cv::imshow("featureDistances", featureDistances);
    cv::waitKey();

    return cv::Mat(32, 32, CV_32FC3);
}
