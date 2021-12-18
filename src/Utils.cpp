//
// Project: image_demorphing
// File: Utils.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "Utils.hpp"
#include <algorithm>
#include <opencv2/highgui.hpp>


namespace {

    Vec3f fullSaturation(float x)
    {
        x *= (6.0f/(2.0f*M_PI));
        return Vec3f(
            std::clamp(2.0f-std::abs(x-4.0f), 0.0f, 1.0f),
            std::clamp(2.0f-std::abs(x-2.0f), 0.0f, 1.0f),
            std::clamp(std::abs(x-3.0f)-1.0f, 0.0f, 1.0f));
    }

}

void gammaCorrect(cv::Mat& image, float gamma)
{
    for (int j=0; j<image.rows; ++j) {
        auto* r = image.ptr<float>(j);
        for (int i=0; i<image.cols*3; ++i) {
            r[i] = std::pow(r[i], gamma);
        }
    }
}

cv::Mat correctImage(const cv::Mat& image, const cv::Mat& correction)
{
    cv::Mat image2 = image.clone();

    for (int j=0; j<image.rows; ++j) {
        auto* rImage2 = image2.ptr<Vec3f>(j);
        auto* rCorrection = correction.ptr<Vec2f>(j);
        for (int i=0; i<image.cols; ++i) {
            Vec2f p(i+0.5f, j+0.5f);

            rImage2[i] = sampleMatCubic<Vec3f>(image, p + rCorrection[i]);
        }
    }

    return image2;
}

void show2ChannelImage(const std::string& windowName, const cv::Mat& image)
{
    cv::Mat image2(image.rows, image.cols, CV_32FC3);

    float maxNorm = 1.0e-8f;
    for (int j=0; j<image.rows; ++j) {
        auto* p = image.ptr<Vec2f>(j);
        auto* p2 = image2.ptr<Vec3f>(j);
        for (int i=0; i<image.cols; ++i) {
            p2[i] = fullSaturation(std::atan2(p[i](1), p[i](0))+M_PI);
            float norm = p[i].norm();
            if (norm > maxNorm)
                maxNorm = norm;
        }
    }

    for (int j=0; j<image.rows; ++j) {
        auto* p = image.ptr<Vec2f>(j);
        auto* p2 = image2.ptr<Vec3f>(j);
        for (int i=0; i<image.cols; ++i) {
            p2[i] *= p[i].norm() / maxNorm;
        }
    }

    cv::imshow(windowName, image2);
}

cv::Mat load2ChannelImage(const std::string& filename)
{
    cv::Mat img1 = cv::imread(filename, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
    cv::Mat img2(img1.rows, img1.cols, CV_32FC2);

    for (int j=0; j<img1.rows; ++j) {
        auto* r = img1.ptr<Vec3f>(j);
        auto* r2 = img2.ptr<Vec2f>(j);
        for (int i=0; i<img1.cols; ++i) {
            r2[i] = r[i].block<2,1>(0,0);
        }
    }

    return img2;
}
