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
#include "KdTree.hpp"

#include <img_demorph_bm/ImageDemorphingBenchmark.hpp>
#include <img_demorph_bm/Utils.hpp>
#include <opencv2/highgui.hpp>
#include <random>


int main(void)
{
    std::string imageFileName = std::string(IMAGE_DEMORPHING_RES_DIR) + "lenna.exr";
    cv::Mat image = cv::imread(imageFileName, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

    gammaCorrect(image, 1.0f/2.2f);

    createCorrection2(image, image);
/*
    ImageDemorphingBenchmark bm(std::move(image));

    auto corr = createCorrection2(bm.getMorphedImage(), bm.getOriginalImage());

    auto imageCorrected = correctImage(bm.getMorphedImage(), corr);

    auto corr2 = corr.clone()*0.025f + 0.5f;
    cv::imshow("corr2", corr2);
    cv::imshow("imageCorrected", imageCorrected);
    cv::waitKey();

    printf("Error: %0.5f\n", bm.evaluate(corr));
*/
    return 0;
}