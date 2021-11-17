//
// Project: image_demorphing
// File: TrainFeatureDetector.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "TrainFeatureDetector.hpp"
#include "Utils.hpp"
#include <opencv2/highgui.hpp>


#define RND ((rnd()%1000001)*0.000001)
#define RANGE(MIN, MAX) (decltype(MIN))(MIN+RND*(MAX-MIN))


std::default_random_engine TrainingImage::rnd(15077155157);


TrainingImage::TrainingImage(cv::Mat&& image,
    int nDistorted,
    const DistortSettings& minSettings,
    const DistortSettings& maxSettings)
:
    original    (std::move(image))
{

    distorted.reserve(nDistorted);

    for (int i=0; i<nDistorted; ++i) {
        DistortSettings settings {
            RANGE(minSettings.nMinTransforms, maxSettings.nMinTransforms),
            RANGE(minSettings.nMaxTransforms, maxSettings.nMaxTransforms),
            RANGE(minSettings.maxPosition, maxSettings.maxPosition),
            RANGE(minSettings.minDistance, maxSettings.minDistance),
            RANGE(minSettings.maxDistance, maxSettings.maxDistance),
            RANGE(minSettings.maxRotation, maxSettings.maxRotation),
            RANGE(minSettings.minScale, maxSettings.minScale),
            RANGE(minSettings.maxScale, maxSettings.maxScale),
            RANGE(minSettings.maxTranslation, maxSettings.maxTranslation)
        };

        distorted.push_back(std::move(distortImage(original, settings)));
    }
}


FeaturePair makeFeaturePair(const TrainingImages& trainingImages, float proximity)
{
    static std::default_random_engine rnd(1507715517);

    const cv::Mat* img1(nullptr);
    const cv::Mat* img2(nullptr);
    Vec2f p1, p2;

    if (proximity < 1.0e-8f) {
        // pick two samples from differing images
        int imgId1 = rnd()%trainingImages.size();
        int imgId2 = rnd()%trainingImages.size();
        while (imgId1 == imgId2)
            imgId2 = rnd()%trainingImages.size();

        img1 = &trainingImages[imgId1].original;
        img2 = &trainingImages[imgId2].original;
        p1 << RND*img1->cols, RND*img1->rows;
        p2 << RND*img2->cols, RND*img2->rows;
    }
    else {
        // pick sample from an original image and random respective distorted one
        int imgId1 = rnd()%trainingImages.size();
        img1 = &trainingImages[imgId1].original;
        int imgId2 = rnd()%trainingImages[imgId1].distorted.size();
        img2 = &trainingImages[imgId1].distorted[imgId2].distorted;
        p1 << RND*img1->cols, RND*img1->rows;
        p2 = p1 + trainingImages[imgId1].distorted[imgId2].forwardMap.at<Vec2f>((int)p1(1), (int)p1(0));
    }

    return std::make_pair(Feature(*img1, p1, 1.0f), Feature(*img2, p2, 1.0f));
}


void trainFeatureDetector()
{
    DistortSettings minSettings{
        10, 15,
        Vec2d(1023.0, 1023.0), // todo remove?
        64.0, 256.0,
        M_PI*0.0625,
        0.6, 0.9,
        Vec2d(32.0, 32.0)
    };

    DistortSettings maxSettings{
        15, 25,
        Vec2d(1023.0, 1023.0), // todo remove?
        256.0, 512.0,
        M_PI*0.125,
        0.85, 1.3,
        Vec2d(64.0, 64.0)
    };

    TrainingImages images;

    cv::Mat img = cv::imread(std::string(IMAGE_DEMORPHING_RES_DIR) + "lenna.exr",
        cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
    images.emplace_back(std::move(img), 2, minSettings, maxSettings);

    auto pair = makeFeaturePair(images, 1.0f);

    cv::imshow("original", images[0].original);
    cv::imshow("distorted1", images[0].distorted[0].distorted);
    cv::imshow("distorted2", images[0].distorted[1].distorted);

    visualizeFeature(pair.first, "f1");
    visualizeFeature(pair.second, "f2");
    cv::waitKey(0);
}
