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
    polar       (Eigen::Matrix<float, fsa*6, fsr>::Zero()),
    energy      (0.0),
    p           (0.0f, 0.0f),
    firstRadius (0.0f)
{
}

Feature::Feature(const cv::Mat& img, const Vec2f& p, float firstRadius, float rotation) :
    p           (p),
    firstRadius (firstRadius)
{
    for (int i=0; i<Feature::fsr; ++i) {
        float angle = 2.0f*M_PI*(i/(float)Feature::fsr) + rotation;
        float r = firstRadius;
        Vec2f dir(std::cos(angle), std::sin(angle));
        for (int j=0; j<Feature::fsa; ++j) {
            polar.block<3,1>(j*3,i) = sampleMatCubic<Vec3f>(img, p+dir*r);
            r *= Feature::frm;
        }
    }

    computeDiffAndEnergy();
}

Feature::Feature(const Image<Vec3f>& img, const Vec2f& p, float firstRadius, float rotation) :
    p           (p),
    firstRadius (firstRadius)
{
    constexpr float sampleDistanceFactor = (2.0*M_PI)/fsr;

    for (int i=0; i<Feature::fsr; ++i) {
        float angle = 2.0f*M_PI*(i/(float)Feature::fsr) + rotation;
        float r = firstRadius;
        Vec2f dir(std::cos(angle), std::sin(angle));
        for (int j=0; j<Feature::fsa; ++j) {
            polar.block<3,1>(j*3,i) = img(p+dir*r, r*sampleDistanceFactor);
            r *= Feature::frm;
        }
    }

    computeDiffAndEnergy();
}

void Feature::computeDiffAndEnergy()
{
    float avg = polar.block<fsa*3, fsr>(0,0).sum() / (fsa*3*fsr);
    polar.block<fsa*3, fsr>(0,0).noalias() -= avg * Eigen::Matrix<float, fsa*3, fsr>::Ones();
    energy = std::sqrt((double)polar.block<fsa*3, fsr>(0,0).array().square().sum() / (fsa*3*fsr));

    if (energy > 1.0e-8f) {
        float sdInv = 1.0f / energy;
        polar.block<fsa * 3, fsr>(0, 0) *= sdInv;
    }

    for (int i=0; i<Feature::fsr; ++i) {
        polar.block<Feature::fsa*3, 1>(fsa*3, i) =
            polar.block<Feature::fsa*3, 1>(0, (i+1)%Feature::fsr) -
                polar.block<Feature::fsa*3, 1>(0, i);
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
    decltype(feature.polar) polarNormalized = feature.polar*0.5f + 0.5f*decltype(feature.polar)::Ones();
    featureImg.data = reinterpret_cast<unsigned char*>(polarNormalized.data());

    if (scale > 1)
        cv::resize(featureImg, featureImg, cv::Size(0,0), scale, scale, cv::INTER_NEAREST);

    cv::imshow(windowName, featureImg);
}
