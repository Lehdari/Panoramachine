//
// Project: image_demorphing
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
    polar       (Polar::Zero()),
    energy      (0.0),
    p           (0.0f, 0.0f),
    firstRadius (0.0f)
{
}

Feature::Feature(const cv::Mat& img, const Vec2f& p, float firstRadius, float rotation) :
    polar       (Polar::Zero()),
    energy      (0.0),
    p           (p),
    firstRadius (firstRadius)
{
    for (int i=0; i<Feature::fsr; ++i) {
        float angle = 2.0f*M_PI*(i/(float)Feature::fsr) + rotation;
        float r = firstRadius;
        Vec2f dir(std::cos(angle), std::sin(angle));
        for (int j=0; j<Feature::fsa; ++j) {
            polar.block<3,1>(j*4,i) = sampleMatCubic<Vec3f>(img, p+dir*r);
            r *= Feature::frm;
        }
    }

    computeDiffAndEnergy();
}

Feature::Feature(const Image<Vec3f>& img, const Vec2f& p, float firstRadius, float rotation) :
    polar       (Polar::Zero()),
    energy      (0.0),
    p           (p),
    firstRadius (firstRadius)
{
    constexpr float sampleDistanceFactor = (2.0*M_PI)/fsr;

    for (int i=0; i<Feature::fsr; ++i) {
        float angle = 2.0f*M_PI*(i/(float)Feature::fsr) + rotation;
        float r = firstRadius;
        Vec2f dir(std::cos(angle), std::sin(angle));
        for (int j=0; j<Feature::fsa; ++j) {
            polar.block<3,1>(j*4,i) = img(p+dir*r, r*sampleDistanceFactor);
            r *= Feature::frm;
        }
    }

    computeDiffAndEnergy();
}

void Feature::computeDiffAndEnergy()
{
    static Eigen::Matrix<float, 3, fsr> diffWeights = Vec3f(0.2f, 0.5f, 0.3f).replicate<1,fsr>();

    double avg = polar.sum() / (fsa*3*fsr);
    polar.noalias() -= avg * Polar::Ones();
    energy = std::sqrt(((double)(polar.array().square().sum()) - avg*avg*fsa*fsr) / (fsa*3*fsr));

    if (energy > 1.0e-8f) {
        float sdInv = 1.0f / energy;
        polar *= sdInv;
    }

    Polar polarShifted;
    polarShifted << polar.block<fsa*4, 1>(0,fsr-1), polar.block<fsa*4, fsr-1>(0,0);

    for (int i=0; i<Feature::fsa; ++i) {
        polar.block<1,fsr>(i*4+3, 0) = (polarShifted.block<3,fsr>(i*4, 0) - polar.block<3,fsr>(i*4, 0)).
            cwiseProduct(diffWeights).colwise().sum();
    }
}

void Feature::writeToFile(std::ofstream& out) const
{
    out.write((char*) (&energy), sizeof(decltype(energy)));
    out.write((char*) (&firstRadius), sizeof(decltype(firstRadius)));
    writeMatrixBinary(out, p);
    writeMatrixBinary(out, polar);
}

void Feature::readFromFile(std::ifstream& in)
{
    in.read((char*) (&energy), sizeof(decltype(energy)));
    in.read((char*) (&firstRadius), sizeof(decltype(firstRadius)));
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

void visualizeFeature(Feature& feature, const std::string& windowName, int scale)
{
    cv::Mat featureImg(Feature::fsr, Feature::fsa*2, CV_32FC3);

    for (int j=0; j<Feature::fsr; ++j) {
        auto* r = featureImg.ptr<Vec3f>(j);
        for (int i=0; i<Feature::fsa; ++i) {
            r[i] = Vec3f(0.5f, 0.5f, 0.5f) + feature.polar.block<3,1>(i*4,j)*0.5f;
            r[i+Feature::fsa] = Vec3f::Ones()*(0.5f + feature.polar(i*4+3,j)*0.5f);
        }
    }

    if (scale > 1)
        cv::resize(featureImg, featureImg, cv::Size(0,0), scale, scale, cv::INTER_NEAREST);

    cv::imshow(windowName, featureImg);
}
