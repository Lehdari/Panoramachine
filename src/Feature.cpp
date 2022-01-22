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
    scale   (0.0f),
    energy  (-1.0)
{
}

Feature::Feature(const Image<Vec3f>& img, const Vec2f& p, float scale) :
    polar   (Polar::Zero()),
    p       (p),
    scale   (scale),
    energy  (-1.0)
{
    sampleCircle(0, img, p, 8, 1.0f*scale);
    sampleCircle(8, img, p, 8, 2.0f*scale);
    sampleCircle(16, img, p, 16, 4.0f*scale);
    sampleCircle(32, img, p, 32, 8.0f*scale);
    sampleCircle(64, img, p, 64, 16.0f*scale);
}

double Feature::getEnergy()
{
    // Use decreasing weights for samples further from center
    static Eigen::Matrix<float, 6, fsn> weightMatrix = []() {
        Eigen::Matrix<float, 6, fsn> w;
        for (int i=0; i<8; ++i)
            w.block<6,1>(0,i) << 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f;
        for (int i=8; i<16; ++i)
            w.block<6,1>(0,i) << 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f;
        for (int i=16; i<32; ++i)
            w.block<6,1>(0,i) << 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f;
        for (int i=32; i<64; ++i)
            w.block<6,1>(0,i) << 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f;
        for (int i=64; i<128; ++i)
            w.block<6,1>(0,i) << 0.0625f, 0.0625f, 0.0625f, 0.0625f, 0.0625f, 0.0625f, 0.0625f, 0.0625f;
        return w;
    }();

    // Filtering relative to scale to avoid aliasing
    float sf = std::max(std::log2(scale), 0.0f);
    int s = (int)sf;

    if (s+1 >= Feature::fsd)
        return 0.0;

    if (energy < 0.0) {
        energy = 0.0;
        { // first layer to be sampled
            auto m = polar.block<6, fsn>(3 + 9 * s, 0);
            Eigen::Matrix<float, 6, fsn> mean = m.rowwise().mean().replicate<1, fsn>();
            energy += (1.0f-sf+s)*std::sqrt((m - mean).cwiseProduct(weightMatrix).array().square().sum());
        }
        for (int i = s+1; i < Feature::fsd; ++i) { // rest of the layers
            auto m = polar.block<6, fsn>(3 + 9*i, 0);
            Eigen::Matrix<float, 6, fsn> mean = m.rowwise().mean().replicate<1, fsn>();
            energy += std::sqrt((m-mean).cwiseProduct(weightMatrix).array().square().sum());
        }
    }

    energy = energy/(fsd-sf);

    return energy;
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
