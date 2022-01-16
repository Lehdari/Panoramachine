//
// Project: panoramachine
// File: Stitch.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#ifndef PANORAMACHINE_STITCH_HPP
#define PANORAMACHINE_STITCH_HPP


#include "Image.hpp"
#include "Feature.hpp"
#include "FeatureDetector.hpp"

#include <vector>


class Stitch {
public:
    using Detector = FeatureDetector<OptimizerStatic>;

    struct Connection {
        size_t  img1;
        size_t  img2;
        Vec2f   p1;
        Vec2f   p2;
        float   scale1;
        float   scale2;
        double  energy1;
        double  energy2;
        Vec2f   diff;
        double  value;

        Connection(Detector& detector, Feature& f1, Feature& f2, size_t img1, size_t img2);

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    Stitch(const std::vector<Image<Vec3f>>& images);

    void operator()();

private:
    struct ImageVisualizationParams {
        int xPos;
        int yPos;
        double xScale;
        double yScale;
    };

    const std::vector<Image<Vec3f>>&        _images;
    std::vector<ImageVisualizationParams>   _visualizationParams;
    cv::Mat                                 _visualization;
    size_t                                  _nConnections;
    std::vector<Connection>                 _connections;
    Detector                                _detector;
    float                                   _minValue;
    float                                   _maxValue;

    void updateConnections();
    void createVisualizationLayout();
    void visualizeConnections(int delay = 0) const;

    static std::default_random_engine rnd;
    static Feature randomSampleFeature(const Image<Vec3f>& image, float scale);
};


#endif //PANORAMACHINE_STITCH_HPP
