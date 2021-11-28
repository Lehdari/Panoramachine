//
// Project: image_demorphing
// File: main.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "TrainFeatureDetector.hpp"
#include "StitchImages.hpp"
#include "Utils.hpp"
#include <opencv2/highgui.hpp>


//#define TRAINING


int main(void)
{
    #ifdef TRAINING
        trainFeatureDetector();
    #else
        // TODO replace with proper, argument-based image loading
        std::vector<std::string> filenames = {"../input/20210909_133243.jpg", "../input/20210909_133239.jpg"};
        //images.emplace_back(cv::imread("../input/20210909_133249.jpg"));
        //images.emplace_back(cv::imread("../input/20210909_133246.jpg"));

        std::vector<cv::Mat> images;

        for (auto& filename : filenames) {
            cv::Mat img;
            cv::imread("../input/20210909_133243.jpg").convertTo(img, CV_32FC3, 1/255.0);
            gammaCorrect(img, 2.2f);
            images.push_back(std::move(img));
        }
        auto result = stitchImages(images);
        cv::imwrite("../result.exr", result);
    #endif

    return 0;
}