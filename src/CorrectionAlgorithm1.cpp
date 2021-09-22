//
// Project: image_demorphing
// File: CorrectionAlgorithms.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "CorrectionAlgorithms.hpp"
#include <img_demorph_bm/ImageDemorphingBenchmark.hpp>


cv::Mat createCorrection1(const cv::Mat& source, const cv::Mat& target)
{
    constexpr float kernel[3][3] = {{0.0625f, 0.125f, 0.0625f}, {0.125f, 0.25f, 0.125f}, {0.0625f, 0.125f, 0.0625f}};
    constexpr float initialSearchSize = 64.0f;

    cv::Mat corr = source.clone() * 0.0f;
    cv::Mat corr2 = corr.clone();
    cv::Mat* c1;
    cv::Mat* c2;
    c1 = &corr;
    c2 = &corr2;
    cv::Mat searchSizes = cv::Mat::ones(corr.rows, corr.cols, CV_32F) * initialSearchSize;

    for (int k=0; k<100; ++k) {
        if (k>0) {
            for (int w=0; w<100-k; ++w) {
                #pragma omp parallel for
                for (int j=0; j<corr.rows; ++j) {
                    auto* rc = c2->ptr<Vec3f>(j);
                    for (int i=0; i<corr.cols; ++i) {
                        Vec3f nc(0.0f, 0.0f, 0.0f);

                        for (int jj=0; jj<3; ++jj) {
                            for (int ii=0; ii<3; ++ii) {
                                nc += kernel[jj][ii] * c1->at<Vec3f>(
                                    std::clamp(j+jj-1, 0, corr.rows-1), std::clamp(i+ii-1, 0, corr.cols-1));
                            }
                        }

                        rc[i] = nc;
                    }
                }
                std::swap(c1, c2);

                auto corr3 = c1->clone()*(0.5f/initialSearchSize) + 0.5f;
                //cv::imshow("corr3", corr3);
                //cv::waitKey(10);
            }
        }

        #pragma omp parallel for
        for (int j=0; j<corr.rows; ++j) {
            auto* rc = c1->ptr<Vec3f>(j);
            auto* rs = source.ptr<Vec3f>(j);
            auto* rt = target.ptr<Vec3f>(j);
            auto* rz = searchSizes.ptr<float>(j);
            for (int i=0; i<corr.cols; ++i) {
                float minLoss = FLT_MAX;

                bool neighbours[8] = {
                    j > 0 && i > 0,
                    j > 0,
                    j > 0 && i < corr.cols-1,
                    i > 0,
                    i < corr.cols-1,
                    j < corr.rows-1 && i > 0,
                    j < corr.rows-1,
                    j < corr.rows-1 && i < corr.cols-1
                };
                Vec2f neighbourCorrs[8] = {
                    neighbours[0] ? corr.at<Vec3f>(j-1, i-1).block<2,1>(0,0) : Vec2f(0.0f, 0.0f),
                    neighbours[1] ? corr.at<Vec3f>(j-1, i).block<2,1>(0,0) : Vec2f(0.0f, 0.0f),
                    neighbours[2] ? corr.at<Vec3f>(j-1, i+1).block<2,1>(0,0) : Vec2f(0.0f, 0.0f),
                    neighbours[3] ? corr.at<Vec3f>(j, i-1).block<2,1>(0,0) : Vec2f(0.0f, 0.0f),
                    neighbours[4] ? corr.at<Vec3f>(j, i+1).block<2,1>(0,0) : Vec2f(0.0f, 0.0f),
                    neighbours[5] ? corr.at<Vec3f>(j+1, i-1).block<2,1>(0,0) : Vec2f(0.0f, 0.0f),
                    neighbours[6] ? corr.at<Vec3f>(j+1, i).block<2,1>(0,0) : Vec2f(0.0f, 0.0f),
                    neighbours[7] ? corr.at<Vec3f>(j+1, i+1).block<2,1>(0,0) : Vec2f(0.0f, 0.0f),
                };
                float neighbourWeights[8] = {
                    neighbours[0] ? 0.5f * (1.7320508f - (rs[i] - source.at<Vec3f>(j-1, i-1)).norm()) : 0.0f,
                    neighbours[1] ? 1.0f * (1.7320508f - (rs[i] - source.at<Vec3f>(j-1, i)).norm()) : 0.0f,
                    neighbours[2] ? 0.5f * (1.7320508f - (rs[i] - source.at<Vec3f>(j-1, i+1)).norm()) : 0.0f,
                    neighbours[3] ? 1.0f * (1.7320508f - (rs[i] - source.at<Vec3f>(j, i-1)).norm()) : 0.0f,
                    neighbours[4] ? 1.0f * (1.7320508f - (rs[i] - source.at<Vec3f>(j, i+1)).norm()) : 0.0f,
                    neighbours[5] ? 0.5f * (1.7320508f - (rs[i] - source.at<Vec3f>(j+1, i-1)).norm()) : 0.0f,
                    neighbours[6] ? 1.0f * (1.7320508f - (rs[i] - source.at<Vec3f>(j+1, i)).norm()) : 0.0f,
                    neighbours[7] ? 0.5f * (1.7320508f - (rs[i] - source.at<Vec3f>(j+1, i+1)).norm()) : 0.0f
                };

                int i2 = i + (int)std::round(rc[i](0));
                int j2 = j + (int)std::round(rc[i](1));
                Vec3f nc(0.0f, 0.0f, 0.0f);

                int sz = std::ceil(rz[i]);
                for (int jj=std::max(j2-sz, 0); jj<std::min(j2+sz+1, corr.rows); ++jj) {
                    for (int ii=std::max(i2-sz, 0); ii<std::min(i2+sz+1, corr.cols); ++ii) {
                        Vec2f v((float)(ii-i), (float)(jj-j));
                        Vec2f v2((float)(ii-i2), (float)(jj-j2));

                        float loss = (rt[i]-source.at<Vec3f>(jj, ii)).squaredNorm();
                        //loss += (v.squaredNorm() / rz[i]*rz[i])*0.001f; // TODO weight factor

                        if (k > 0) {
                            for (int n = 0; n < 8; ++n) {
                                if (neighbours[n])
                                    loss += neighbourWeights[n] * (v - neighbourCorrs[n]).squaredNorm() * 0.01f;
                            }
                        }

                        if (loss < minLoss) {
                            minLoss = loss;
                            nc << v, 0.0f;
                        }
                    }
                }

                rc[i] = 0.5f*rc[i] + 0.5f*nc;
            }
        }

        #pragma omp parallel for
        for (int j=0; j<corr.rows; ++j) {
            auto* rc = c1->ptr<Vec3f>(j);
            auto* rz = searchSizes.ptr<float>(j);
            for (int i=0; i<corr.cols; ++i) {
                float maxCorrDiff = 0.0f;

                for (int jj=std::max(j-1, 0); jj<std::min(j+2, corr.rows); ++jj) {
                    for (int ii=std::max(i-1, 0); ii<std::min(i+2, corr.cols); ++ii) {
                        float corrDiff = (rc[i]-corr.at<Vec3f>(jj, ii)).norm();

                        if (corrDiff > maxCorrDiff)
                            maxCorrDiff = corrDiff;
                    }
                }

                rz[i] = std::max(0.95f*rz[i] + 0.15f*maxCorrDiff, 1.0f);
            }
        }
/*
        auto corr3 = c1->clone()*(0.5f/initialSearchSize) + 0.5f;
        cv::imshow("corr3", corr3);
        cv::Mat searchSizes2;
        searchSizes.convertTo(searchSizes2, CV_8U);
        cv::imshow("searchSizes2", searchSizes2);
        cv::waitKey(10);
        */
    }

    return corr;
}
