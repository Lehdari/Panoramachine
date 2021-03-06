//
// Project: panoramachine
// File: main.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "TrainFeatureDetector.hpp"
#include "Stitch.hpp"
#include "Utils.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#include "Image.hpp"


//#define TRAINING


int main(void)
{
    #ifdef TRAINING
        trainFeatureDetector();
    #else
        // TODO replace with proper, argument-based image loading
        std::vector<std::string> filenames = {
            "../input/20210909_133249.jpg",
            "../input/20210909_133246.jpg",
            "../input/20210909_133243.jpg",
            "../input/20210909_133239.jpg"
        };

        std::vector<Image<Vec3f>> images;
        for (auto& filename : filenames) {
            cv::Mat img;
            cv::imread(filename).convertTo(img, CV_32FC3, 1/255.0);
            images.emplace_back(std::move(img));
        }
        Stitch stitch(images);
        stitch();
    #endif

    return 0;
}
