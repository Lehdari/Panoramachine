//
// Project: image_demorphing
// File: Feature.hpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#ifndef IMAGE_DEMORPHING_FEATURE_HPP
#define IMAGE_DEMORPHING_FEATURE_HPP


#include <opencv2/core/mat.hpp>

#include "MathTypes.hpp"
#include "Image.hpp"


struct Feature {
    static constexpr int fsn = 128; // number of samples in feature
    static constexpr int fsd = 10; // feature sampling depth (in n. of layers)
    static constexpr double fmr = 16.0; // feature max radius

    using Polar = Eigen::Matrix<float, 9*fsd, fsn>;
    Polar   polar;
    Vec2f   p;
    float   scale;

    Feature();
    Feature(const Image<Vec3f>& img, const Vec2f& p, float scale);

    void writeToFile(std::ofstream& out) const;
    void readFromFile(std::ifstream& in);
    void writeToFile(const std::string& filename) const;
    void readFromFile(const std::string& filename);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    void sampleCircle(int firstColId, const Image<Vec3f>& img, const Vec2f& p, int n, float radius);
};


void visualizeFeature(Feature& feature, const std::string& windowName, int scale=1);


#endif //IMAGE_DEMORPHING_FEATURE_HPP
