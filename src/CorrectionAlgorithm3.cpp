//
// Project: image_demorphing
// File: CorrectionAlgorithm3.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "CorrectionAlgorithms.hpp"
#include <img_demorph_bm/Utils.hpp>
#include <opencv2/highgui.hpp>


inline Vec3f samplePyramid(const std::vector<cv::Mat>& pyramid, const Vec2f& p)
{
    float sampleScale = 1.0f;
    float amplitudeScale = 1.0f;

    Vec3f sample(0.0f, 0.0f, 0.0f);
    for (auto& level : pyramid) {
        sample += sampleMatCubic(level, p*sampleScale)*amplitudeScale;
        sampleScale /= 2.0f;
        amplitudeScale *= 2.0f;
    }

    return sample;
}

float getError(const cv::Mat& source, const cv::Mat& target,
    const std::vector<cv::Mat>& pyramid, const Vec2f& p)
{
    return (sampleMatCubic(target, p) -
        sampleMatCubic(source, p + samplePyramid(pyramid, p).block<2,1>(0,0))).squaredNorm();
}

Vec3f getGradient(const cv::Mat& source, const cv::Mat& target,
    std::vector<cv::Mat>& pyramid, const Vec2f& p, Vec3f& c)
{
    constexpr float eps = 1.0e-1f;

    Vec3f g(0.0f, 0.0f, 0.0f);

    float e, c0;

    c0 = c(0);
    c(0) += eps;
    e = getError(source, target, pyramid, p);
    c(0) -= 2.0f*eps;
    e -= getError(source, target, pyramid, p);
    c(0) = c0;
    g(0) = e / (2.0f*eps);

    c0 = c(1);
    c(1) += eps;
    e = getError(source, target, pyramid, p);
    c(1) -= 2.0f*eps;
    e -= getError(source, target, pyramid, p);
    c(1) = c0;
    g(1) = e / (2.0f*eps);

    return g;
}

cv::Mat createCorrection3(const cv::Mat& source, const cv::Mat& target)
{
    constexpr float gdRate = 1.0e-1f;
    constexpr float gdMomentum = 0.95f;
    constexpr float gDiffusionRate = 0.1f;
    constexpr int nLevels = 8;
    constexpr float kernel[3][3] = {{0.0625f, 0.125f, 0.0625f}, {0.125f, 0.25f, 0.125f}, {0.0625f, 0.125f, 0.0625f}};

    int startLevel = 6;

    std::vector<cv::Mat> corrPyramid, gPyramid;
    corrPyramid.push_back(source.clone() * 0.0f);
    gPyramid.push_back(source.clone() * 0.0f);
    for (int i=0; i<nLevels-1; ++i) {
        corrPyramid.emplace_back(corrPyramid.back().rows/2+1, corrPyramid.back().cols/2+1, CV_32FC3, 0.0f);
        gPyramid.emplace_back(gPyramid.back().rows/2+1, gPyramid.back().cols/2+1, CV_32FC3, 0.0f);
        //printf("%d %d\n", i, gPyramid.back().rows);
    }
    cv::Mat corr = source.clone() * 0.0f;
    cv::Mat corrImg = source.clone();
    cv::Mat diffImg = cv::abs(corrImg-target);

    for (int k=0; k<200; ++k) {
        printf("k: %d\n", k);

        //#pragma omp parallel for
        for (int j = 0; j < source.rows; ++j) {
            for (int i = 0; i < source.cols; ++i) {
                Vec2f p(i, j);
                for (int l=startLevel; l<nLevels; ++l) {
                    float lf = 1.0f / (1 << l);
                    Vec3f& c = corrPyramid[l].at<Vec3f>((int)(p(1) * lf + 0.5f), (int)(p(0) * lf + 0.5f));
                    Vec3f& g = gPyramid[l].at<Vec3f>((int)(p(1) * lf + 0.5f), (int)(p(0) * lf + 0.5f));
                    Vec3f gg = getGradient(source, target, corrPyramid, p, c);
                    g += gg * lf*lf;
                }
            }
        }

        // diffusion
        for (int l=startLevel; l<nLevels; ++l) {
            auto& level = gPyramid[l];
            cv::Mat level2 = level.clone();
            for (int j = 0; j < level.rows; ++j) {
                auto *r2 = level2.ptr<Vec3f>(j);
                for (int i = 0; i < level.cols; ++i) {
                    Vec3f nc(0.0f, 0.0f, 0.0f);

                    for (int jj = 0; jj < 3; ++jj) {
                        for (int ii = 0; ii < 3; ++ii) {
                            nc += kernel[jj][ii] * level.at<Vec3f>(
                                std::clamp(j + jj - 1, 0, level.rows - 1), std::clamp(i + ii - 1, 0, level.cols - 1));
                        }
                    }

                    r2[i] = gDiffusionRate*nc + (1.0f-gDiffusionRate)*r2[i];
                }
            }
            cv::swap(level, level2);
        }

        for (int l=startLevel; l<nLevels; ++l) {
            auto& level = corrPyramid[l];
            float lf = 1.0f / (1 << l);

            #pragma omp parallel for
            for (int j = 0; j < level.rows; ++j) {
                for (int i = 0; i < level.cols; ++i) {
                    Vec2f p(i, j);
                    Vec3f& c = level.at<Vec3f>(j, i);
                    c -= gPyramid[l].at<Vec3f>(j, i) * gdRate * lf;
                }
            }
        }

        for (auto& level : gPyramid)
            level *= gdMomentum;

        #pragma omp parallel for
        for (int j = 0; j < source.rows; ++j) {
            auto *cr = corr.ptr<Vec3f>(j);
            for (int i = 0; i < source.cols; ++i) {
                cr[i] = samplePyramid(corrPyramid, Vec2f(i, j));
            }
        }

        corrImg = correctImage(source, corr);
        diffImg = cv::abs(corrImg-target);
        cv::imshow("corr", corr*0.1f + 0.5f);
        cv::imshow("corrected", corrImg);
        cv::imshow("diff", diffImg);
        cv::waitKey(20);
    }

    return corr;
}