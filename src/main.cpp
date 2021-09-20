//
// Project: image_demorphing
// File: main.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include <img_demorph_bm/ImageDemorphingBenchmark.hpp>
#include <img_demorph_bm/Utils.hpp>
#include <opencv2/highgui.hpp>


cv::Mat createCorrection(const cv::Mat& source, const cv::Mat& target)
{
    cv::Mat corr = source.clone() * 0.0f;

    cv::Mat searchSizes = cv::Mat::ones(corr.rows, corr.cols, CV_32S) * 32;
    for (int j=0; j<corr.rows; ++j) {
        auto* rc = corr.ptr<Vec3f>(j);
        auto* rt = target.ptr<Vec3f>(j);
        auto* rz = searchSizes.ptr<int>(j);
        for (int i=0; i<corr.cols; ++i) {
            float minLoss = FLT_MAX;

            for (int jj=std::max(j-rz[i], 0); jj<std::min(j+rz[i]+1, corr.rows); ++jj) {
                for (int ii=std::max(i-rz[i], 0); ii<std::min(i+rz[i]+1, corr.cols); ++ii) {
                    float loss = (rt[i]-source.at<Vec3f>(jj, ii)).squaredNorm();

                    if (loss < minLoss) {
                        minLoss = loss;
                        rc[i] << (float)(ii-i), (float)(jj-j), 0.0f;
                    }
                }
            }
            //printf("(%4d, %4d): [%0.5f, %0.5f]\n", i, j, rc[i](0), rc[i](1));
        }
    }

    return corr;
}


int main(void)
{
    std::string imageFileName = std::string(IMAGE_DEMORPHING_RES_DIR) + "lenna.exr";
    cv::Mat image = cv::imread(imageFileName, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

    gammaCorrect(image, 1.0f/2.2f);

    ImageDemorphingBenchmark bm(std::move(image));

    auto corr = createCorrection(bm.getMorphedImage(), bm.getOriginalImage());

    auto imageCorrected = correctImage(bm.getMorphedImage(), corr);

    auto corr2 = corr.clone()*0.01f + 0.5f;
    cv::imshow("corr2", corr2);
    cv::imshow("imageCorrected", imageCorrected);
    cv::waitKey();

    printf("Error: %0.5f\n", bm.evaluate(corr));

    return 0;
}