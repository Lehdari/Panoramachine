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


Stitch::Connection::Connection(Detector& detector, Feature& f1, Feature& f2, size_t img1, size_t img2) :
    img1    (img1),
    img2    (img2),
    p1      (f1.p),
    p2      (f2.p),
    scale1  (f1.scale),
    scale2  (f2.scale),
    energy1 (f1.getEnergy()),
    energy2 (f2.getEnergy()),
    diff    (detector(f1, f2).block<2,1>(0,0)),
    value   (0.0)
{
}

Stitch::Stitch(const std::vector<Image<Vec3f>>& images) :
    _images             (images),
    _nConnections       (_images.size()*4*8)
{
    _detector.loadWeights("../feature_detector_model"); // TODO take weight directory as input
    for (size_t i=0; i<_nConnections; ++i) {
        size_t img1 = rnd()%_images.size();
        size_t img2 = rnd()%_images.size();
        while (img2 == img1)
            img2 = rnd()%_images.size();
        float scale = std::pow(2.0f, 3.0f+RND*3.0f);
        auto f1 = randomSampleFeature(_images[img1], scale, 10);
        auto f2 = randomSampleFeature(_images[img2], scale, 10);
        _connections.emplace_back(_detector, f1, f2, img1, img2);
    }
    updateConnections();

    createVisualizationLayout();
    visualizeConnections(); // TODO remove
}

void Stitch::operator()()
{
    std::vector<Connection> connectionsCopy = _connections;
    for (int i=0; i<10000; ++i) {
        std::sort(_connections.begin(), _connections.end(),
            [](const Connection& c1, const Connection& c2){
            return c1.value > c2.value;
        });
        connectionsCopy = _connections;
        for (int j=0; j<_nConnections; ++j) {
            size_t img1, img2;
            Feature f1, f2;
            float scale;
            double replaceProbability = (j/(double)_nConnections)+0.2f*RND-0.1f;
            if (replaceProbability >= 0.9) {
                if (RND < 0.75) {
                    Connection *c = nullptr;
                    int jj = rnd()%_nConnections;
                    while (jj / (double) _nConnections > RND)
                        jj = rnd()%_nConnections;
                    c = &_connections[jj];
                    float dir1 = RND * 2.0f * M_PI;
                    float dir2 = RND * 2.0f * M_PI;
                    scale = std::clamp(c->scale1 * (float)std::pow(2.0f, -1.0f+RND*2.0f), 4.0f, 64.0f);;
                    float dis = 1.0f+3.0f*RND;
                    Vec2f p1 = c->p1 + Vec2f(cosf(dir1), sinf(dir1))*(c->scale1+scale)*Feature::fmr*dis;
                    Vec2f p2 = c->p2 + Vec2f(cosf(dir2), sinf(dir2))*(c->scale2+scale)*Feature::fmr*dis;
                    p1(0) = std::clamp(p1(0), 0.0f, (float)_images[c->img1][0].cols);
                    p1(1) = std::clamp(p1(1), 0.0f, (float)_images[c->img1][0].rows);
                    p2(0) = std::clamp(p2(0), 0.0f, (float)_images[c->img2][0].cols);
                    p2(1) = std::clamp(p2(1), 0.0f, (float)_images[c->img2][0].rows);
                    img1 = c->img1;
                    img2 = c->img2;
                    f1 = Feature(_images[img1], p1, scale);
                    f2 = Feature(_images[img2], p2, scale);
                }
                else {
                    img1 = rnd()%_images.size();
                    img2 = rnd()%_images.size();
                    while (img2 == img1)
                        img2 = rnd()%_images.size();
                    scale = std::pow(2.0f, 1.0f+RND*5.0f);
                    f1 = randomSampleFeature(_images[img1], scale, 10);
                    f2 = randomSampleFeature(_images[img2], scale, 10);
                }
            }
            else {
                // Update the connection towards smaller diff
                auto& c = _connections[j];
                double energySum = c.energy1+c.energy2; // less energy -> move more
                Vec2f p1 = c.p1 + c.diff*(c.energy2/energySum)*c.scale1*Feature::fmr;
                Vec2f p2 = c.p2 - c.diff*(c.energy1/energySum)*c.scale2*Feature::fmr;
                p1(0) = std::clamp(p1(0), 0.0f, (float)_images[c.img1][0].cols);
                p1(1) = std::clamp(p1(1), 0.0f, (float)_images[c.img1][0].rows);
                p2(0) = std::clamp(p2(0), 0.0f, (float)_images[c.img2][0].cols);
                p2(1) = std::clamp(p2(1), 0.0f, (float)_images[c.img2][0].rows);
                scale = c.scale1;
                img1 = c.img1;
                img2 = c.img2;
                f1 = Feature(_images[img1], p1, scale);
                f2 = Feature(_images[img2], p2, scale);
            }
            connectionsCopy[j] = Connection(_detector, f1, f2, img1, img2);
        }
        _connections.swap(connectionsCopy);
        updateConnections();

        visualizeConnections(15);
    }
    visualizeConnections();
}

void Stitch::updateConnections()
{
    // Find limits for diff and energy
    double minDiff = std::numeric_limits<double>::max();
    double maxDiff = -std::numeric_limits<double>::max();
    double minEnergy = std::numeric_limits<double>::max();
    double maxEnergy = -std::numeric_limits<double>::max();
    for (auto& c : _connections) {
        double diff = -c.diff.cast<double>().norm();
        if (diff < minDiff) minDiff = diff;
        if (diff > maxDiff) maxDiff = diff;
        double energy = std::min(c.energy1, c.energy2);
        if (energy < minEnergy) minEnergy = energy;
        if (energy > maxEnergy) maxEnergy = energy;
    }

    // Compute location cost and limits
    double minLocCost = std::numeric_limits<double>::max();
    double maxLocCost = -std::numeric_limits<double>::max();
    std::vector<double> locCosts(_nConnections, 0.0);
    for (int i=0; i<_nConnections; ++i) {
        double cost = 0.0;
        auto& p1i = _connections[i].p1;
        auto& p2i = _connections[i].p2;
        auto& img1i = _connections[i].img1;
        auto& img2i = _connections[i].img2;
        for (int j=0; j<_nConnections; ++j) {
            if (i==j) continue;
            auto& p1j = _connections[j].p1;
            auto& p2j = _connections[j].p2;
            float s1j = _connections[j].scale1*Feature::fmr;
            float s2j = _connections[j].scale2*Feature::fmr;
            auto& img1j = _connections[j].img1;
            auto& img2j = _connections[j].img2;
            if (img1i == img1j)
                cost += std::exp(-0.5*((p1i-p1j).squaredNorm()/(s1j*s1j))) / (s1j*std::sqrt(2.0*M_PI));
            if (img2i == img1j)
                cost += std::exp(-0.5*((p2i-p1j).squaredNorm()/(s1j*s1j))) / (s1j*std::sqrt(2.0*M_PI));
            if (img1i == img2j)
                cost += std::exp(-0.5*((p1i-p2j).squaredNorm()/(s2j*s2j))) / (s2j*std::sqrt(2.0*M_PI));
            if (img2i == img2j)
                cost += std::exp(-0.5*((p2i-p2j).squaredNorm()/(s2j*s2j))) / (s2j*std::sqrt(2.0*M_PI));
        }
        locCosts[i] = -cost;
        if (-cost < minLocCost) minLocCost = -cost;
        if (-cost > maxLocCost) maxLocCost = -cost;
    }

    _minValue = std::numeric_limits<float>::max();
    _maxValue = 0.0f;
    for (int i=0; i<_nConnections; ++i) {
        auto& c = _connections[i];
        double diff = -c.diff.cast<double>().norm();
        double energy = std::min(c.energy1, c.energy2);
        double locCost = locCosts[i];
        c.value = (diff-minDiff)/(maxDiff-minDiff)*(
            (energy-minEnergy)/(maxEnergy-minEnergy)+
            (locCost-minLocCost)/(maxLocCost-minLocCost));
        if (c.value < _minValue) _minValue = c.value;
        if (c.value > _maxValue) _maxValue = c.value;
    }
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

void Stitch::visualizeConnections(int delay) const
{
    cv::Mat visCopy = _visualization.clone();
    for (auto& c : _connections) {
        auto& params1 = _visualizationParams[c.img1];
        auto& params2 = _visualizationParams[c.img2];
        cv::Point p1(params1.xPos + c.p1(0)*params1.xScale, params1.yPos + c.p1(1)*params1.yScale);
        cv::Point p2(params2.xPos + c.p2(0)*params2.xScale, params2.yPos + c.p2(1)*params2.yScale);
        float valueScaled = (c.value-_minValue)/(_maxValue-_minValue);
        cv::Scalar color(valueScaled, valueScaled, valueScaled);
        cv::circle(visCopy, p1, c.scale1*Feature::fmr*params1.xScale, color);
        cv::circle(visCopy, p2, c.scale2*Feature::fmr*params2.xScale, color);
        cv::line(visCopy, p1, p2, color);
    }
    cv::imshow("Connections", visCopy);
    cv::waitKey(delay);
}

Feature Stitch::randomSampleFeature(const Image<Vec3f>& image, float scale, int energyIterations)
{
    Feature feature1(image, Vec2f(image[0].cols*RND, image[0].rows*RND), scale);
    Feature feature2;
    Feature* f1 = &feature1;
    Feature* f2 = &feature2;
    double e1 = f1->getEnergy();
    double e2;
    for (int i=0; i<energyIterations; ++i) {
        *f2 = Feature(image, Vec2f(image[0].cols*RND, image[0].rows*RND), scale);
        e2 = f2->getEnergy();
        if (e2 > e1) {
            std::swap(f1, f2);
            e1 = e2;
        }
    }
    return *f1;
}
