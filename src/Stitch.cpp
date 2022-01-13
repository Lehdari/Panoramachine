//
// Project: panoramachine
// File: Stitch.cpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "Stitch.hpp"


#define RND ((rnd()%1000001)*0.000001)


std::default_random_engine Stitch::rnd(1507715517);


Stitch::Connection::Connection(
    Detector& detector,
    const Feature& f1, const Feature& f2,
    size_t img1, size_t img2) :
    img1    (img1),
    img2    (img2),
    p1      (f1.p),
    p2      (f2.p),
    scale1  (f1.scale),
    scale2  (f2.scale),
    diff    (detector(f1, f2).block<2,1>(0,0))
{
}

Stitch::Stitch(const std::vector<Image<Vec3f>>& images) :
    _images             (images),
    _nConnections       (_images.size()*4*8),
    _minDiff            (std::numeric_limits<float>::max()),
    _maxDiff            (0.0f)
{
    _detector.loadWeights("../feature_detector_model"); // TODO take weight directory as input
    for (size_t i=0; i<_nConnections; ++i) {
        size_t img1 = rnd()%_images.size();
        size_t img2 = rnd()%_images.size();
        while (img2 == img1)
            img2 = rnd()%_images.size();
        float scale = std::pow(2.0f, 4.0f+RND*2.0f);
        auto f1 = randomSampleFeature(_images[img1], scale);
        auto f2 = randomSampleFeature(_images[img2], scale);
        _connections.emplace_back(_detector, f1, f2, img1, img2);
        float diff = _connections.back().diff.norm();
        if (diff < _minDiff) _minDiff = diff;
        if (diff > _maxDiff) _maxDiff = diff;
        printf("%0.5f\n", _connections.back().diff.norm());
    }

    createVisualizationLayout();
    visualizeConnections(); // TODO remove
}

void Stitch::createVisualizationLayout()
{
    // TODO remove hardcoded visualization params and calculate them properly
    _visualizationParams.push_back(ImageVisualizationParams{0,0,0.25,0.25});
    _visualizationParams.push_back(ImageVisualizationParams{1008,0,0.25,0.25});
    _visualizationParams.push_back(ImageVisualizationParams{1008*2,0,0.25,0.25});
    _visualizationParams.push_back(ImageVisualizationParams{1008*3,0,0.25,0.25});

    int visualizationWidth = 4*1008;
    int visualizationHeight = 756;
    _visualization.create(cv::Size(visualizationWidth, visualizationHeight), CV_32FC3);
    cv::Mat resized;
    for (int i=0; i<_images.size(); ++i) {
        auto& params = _visualizationParams[i];
        cv::resize(_images[i][0], resized, cv::Size(0,0), params.xScale, params.yScale);
        cv::Mat vis_roi = _visualization(cv::Rect(params.xPos, params.yPos,
            _images[i][0].cols*params.xScale, _images[i][0].rows*params.yScale));
        resized.copyTo(vis_roi);
    }
}

void Stitch::visualizeConnections() const
{
    cv::Mat visCopy = _visualization.clone();
    for (auto& c : _connections) {
        auto& params1 = _visualizationParams[c.img1];
        auto& params2 = _visualizationParams[c.img2];
        cv::Point p1(params1.xPos + c.p1(0)*params1.xScale, params1.yPos + c.p1(1)*params1.yScale);
        cv::Point p2(params2.xPos + c.p2(0)*params2.xScale, params2.yPos + c.p2(1)*params2.yScale);
        float diffScaled = (c.diff.norm()-_minDiff)/(_maxDiff-_minDiff);
        cv::Scalar color(1.0-diffScaled, 1.0-diffScaled, 1.0-diffScaled);
        cv::circle(visCopy, p1, c.scale1*Feature::fmr*params1.xScale, color);
        cv::circle(visCopy, p2, c.scale2*Feature::fmr*params2.xScale, color);
        cv::line(visCopy, p1, p2, color);
    }
    cv::imshow("Connections", visCopy);
    cv::waitKey();
}

Feature Stitch::randomSampleFeature(const Image<Vec3f>& image, float scale)
{
    return Feature(image, Vec2f(image[0].cols*RND, image[0].rows*RND), scale);
}
