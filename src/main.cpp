//
// Project: image_demorphing
// File: main.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "CorrectionAlgorithms.hpp"
#include "DistortImage.hpp"
#include "Utils.hpp"
#include <opencv2/highgui.hpp>
#include <random>


int main(void)
{
    std::string imageFileName = std::string(IMAGE_DEMORPHING_RES_DIR) + "lenna.exr";
    cv::Mat image = cv::imread(imageFileName, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

    DistortSettings settings{
        15, 25,
        Vec2d(image.cols, image.rows),
        128.0, 512.0,
        M_PI*0.125,
        0.85, 0.9,
        Vec2d(64.0, 64.0)
    };
    auto distortedImage = distortImage(image, settings);

    cv::imshow("distorted", distortedImage.distorted);
    show2ChannelImage("backwardMap", distortedImage.backwardMap);
    show2ChannelImage("forwardMap", distortedImage.forwardMap);
    cv::waitKey(10);

    createCorrection2(distortedImage.distorted, image);
    cv::waitKey(0);

    return 0;
}