//
// Project: panoramachine
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
#include "Momentum.hpp"

#include <random>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#define RND ((rnd()%1000001)*0.000001)
#define RNDE ((rnd()%1000000)*0.000001) // endpoint excluded


void drawFeatures(
    const cv::Mat& combinedScaled,
    std::vector<Feature>& features1,
    std::vector<Feature>& features2,
    Mat3f homography = Mat3f::Identity(),
    int wait = 0)
{
    cv::Mat img = combinedScaled.clone();
    for (int i=0; i<features1.size(); ++i) {
        auto& f1 = features1[i];
        auto& f2 = features2[i];
        Vec3f f1p1, f1p2;
        f1p1 << f1.p, 1.0f;
        f1p2 = homography * f1p1;
        f1p2 /= f1p2(2);
        cv::Point p1(f1.p(0) / 4.0f, f1.p(1) / 4.0f);
        cv::Point p2(combinedScaled.cols/2 + f2.p(0) / 4.0f, f2.p(1) / 4.0f);
        cv::Point p2h(combinedScaled.cols/2 + f1p2(0) / 4.0f, f1p2(1) / 4.0f);
        cv::circle(img, p1, f1.scale*4, cv::Scalar(1.0, 1.0, 1.0));
        cv::circle(img, p2, f2.scale*4, cv::Scalar(1.0, 1.0, 1.0));
        cv::line(img, p1, p2, cv::Scalar(1.0, 1.0, 1.0));
        cv::line(img, p1, p2h, cv::Scalar(0.0, 0.0, 1.0));
    }
    cv::imshow("stitch", img);
    cv::waitKey(wait);
}

void findMatchingFeatures(
    FeatureDetector<OptimizerStatic>& detector,
    Feature& f1,
    Feature& f2,
    const Image<Vec3f>& img1,
    const Image<Vec3f>& img2,
    float threshold = 0.5f)
{
    static std::default_random_engine rnd(1507715517);
    float scale = 0.0f;
    Vec2f diff(1.0f, 1.0f);
    while (diff.norm() > threshold) {
        Vec2f p1(RND * img1[0].cols, RND * img1[0].rows);
        Vec2f p2(RND * img2[0].cols, RND * img2[0].rows);
        scale = std::pow(2.0f, 4.0f + RND * 2.0f);
        f1 = Feature(img1, p1, scale);
        f2 = Feature(img2, p2, scale);
        diff = detector(f1, f2).block<2, 1>(0, 0);
    }
}

cv::Mat stitchImages(const std::vector<Image<Vec3f>>& images)
{
    FeatureDetector<OptimizerStatic> detector;
    detector.loadWeights("../feature_detector_model");

    cv::Mat img1scaled, img2scaled, combinedScaled, stitch;
    cv::resize(static_cast<cv::Mat>(images[0]), img1scaled, cv::Size(0,0), 1.0/4.0, 1.0/4.0, cv::INTER_LINEAR);
    cv::resize(static_cast<cv::Mat>(images[1]), img2scaled, cv::Size(0,0), 1.0/4.0, 1.0/4.0, cv::INTER_LINEAR);
    cv::hconcat(img1scaled, img2scaled, combinedScaled);

    std::vector<Feature> features1;
    std::vector<Feature> features2;
    std::vector<Momentum<Vec2f>> diffMomenta;

    constexpr int nFeatures = 16;
    constexpr double diffMomentum = 0.75;
    for (int f=0; f<nFeatures; ++f) {
        Feature f1, f2;
        findMatchingFeatures(detector, f1, f2, images[0], images[1], 0.25f);
        features1.push_back(std::move(f1));
        features2.push_back(std::move(f2));
        diffMomenta.emplace_back(diffMomentum, detector(f1, f2).block<2, 1>(0, 0));
    }

    std::vector<Vec2f> fp1, fp2;
    for (int i=0; i<nFeatures; ++i) {
        fp1.push_back(features1[i].p);
        fp2.push_back(features2[i].p);
    }
    Mat3f h = computeHomography(fp1, fp2);

    drawFeatures(combinedScaled, features1, features2, h);

    constexpr int nOptimizationSteps = 1024;
    constexpr float maxDiffScale = 1.0f;
    constexpr float maxVarianceScale = 1.0f;
    for (int e = 0; e < nOptimizationSteps; ++e) {
        float maxDiff = (1.0f - ((float)e/nOptimizationSteps)*0.5f)*maxDiffScale;
        float maxVariance = (1.0f - ((float)e/nOptimizationSteps)*0.75f)*maxVarianceScale;
        for (int f=0; f<nFeatures; ++f) {
            auto& f1 = features1[f];
            auto& f2 = features2[f];
            Vec2f diff = diffMomenta[f];
            Vec2f p1 = f1.p + diff * 0.25f * f1.scale * Feature::fmr;
            Vec2f p2 = f2.p - diff * 0.25f * f2.scale * Feature::fmr;
            if (diffMomenta[f].variance() > maxVariance ||
                f1.scale > 64.0f ||
                p1(0) < 0.0f || p1(0) > (float)images[0][0].cols ||
                p1(1) < 0.0f || p1(1) > (float)images[0][0].rows ||
                p2(0) < 0.0f || p2(0) > (float)images[1][0].cols ||
                p2(1) < 0.0f || p2(1) > (float)images[1][0].rows) {
                findMatchingFeatures(detector, f1, f2, images[0], images[1], maxDiff*0.5f);
                diffMomenta[f] = Momentum<Vec2f>(diffMomentum, detector(f1, f2).block<2, 1>(0, 0));
            }
            else {
                float newScale = std::max(f1.scale * (1.0f-maxVariance*0.4f+diffMomenta[f].variance()), 0.25f);
                f1 = Feature(images[0], p1, newScale);
                f2 = Feature(images[1], p2, newScale);
                diffMomenta[f](detector(f1, f2).block<2, 1>(0, 0));
            }
            fp1[f] = f1.p;
            fp2[f] = f2.p;
        }
        h = computeHomography(fp1, fp2);
        drawFeatures(combinedScaled, features1, features2, h, 20);
    }
    drawFeatures(combinedScaled, features1, features2, h);

    return cv::Mat(32, 32, CV_32FC3);
}
