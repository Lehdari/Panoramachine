//
// Project: panoramachine
// File: Feature.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "Feature.hpp"
#include "Utils.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


Feature::Feature() :
    polar   (Polar::Zero()),
    p       (0.0f, 0.0f),
    scale   (0.0f)
{
}

Feature::Feature(const Image<Vec3f>& img, const Vec2f& p, float scale) :
    polar   (Polar::Zero()),
    p       (p),
    scale   (scale)
{
    sampleCircle(0, img, p, 8, 1.0f*scale);
    sampleCircle(8, img, p, 8, 2.0f*scale);
    sampleCircle(16, img, p, 16, 4.0f*scale);
    sampleCircle(32, img, p, 32, 8.0f*scale);
    sampleCircle(64, img, p, 64, 16.0f*scale);
}

void Feature::writeToFile(std::ofstream& out) const
{
    out.write((char*) (&scale), sizeof(decltype(scale)));
    writeMatrixBinary(out, p);
    writeMatrixBinary(out, polar);
}

void Feature::readFromFile(std::ifstream& in)
{
    in.read((char*) (&scale), sizeof(decltype(scale)));
    readMatrixBinary(in, p);
    readMatrixBinary(in, polar);
}

void Feature::writeToFile(const std::string& filename) const
{
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    writeToFile(out);
    out.close();
}

void Feature::readFromFile(const std::string& filename)
{
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    readFromFile(in);
    in.close();
}

void Feature::sampleCircle(int firstColId, const Image<Vec3f>& img, const Vec2f& p, int n, float radius)
{
    const float angleStep = (2.0*M_PI)/n;
    for (int i=0; i<n; ++i) {
        Vec2f pp = p + Vec2f(std::cos(i*angleStep), std::sin(i*angleStep))*radius;
        for (int j=0; j<fsd; ++j) {
            polar.block<3,1>(j*9, firstColId+i) = img.sampleCubic(pp, j);
            polar.block<3,1>(j*9+3, firstColId+i) = img.sampleCubicXDeriv(pp, j);
            polar.block<3,1>(j*9+6, firstColId+i) = img.sampleCubicYDeriv(pp, j);
        }
    }
}

void visualizeFeature(Feature& feature, const std::string& windowName, int scale)
{
    cv::Mat featureImg(
        Feature::Polar::ColsAtCompileTime, Feature::Polar::RowsAtCompileTime/3,
        CV_32FC3, feature.polar.data());

    if (scale > 1) {
        cv::Mat featureImg2;
        cv::resize(featureImg, featureImg2, cv::Size(0, 0), scale, scale, cv::INTER_NEAREST);
        cv::imshow(windowName, featureImg2);
    }
    else
        cv::imshow(windowName, featureImg);
}
