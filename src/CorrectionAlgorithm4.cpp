//
// Project: image_demorphing
// File: CorrectionAlgorithm4.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "CorrectionAlgorithms.hpp"
#include <img_demorph_bm/Utils.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


float getError(const cv::Mat& source, const cv::Mat& target, const cv::Mat& corr, const Vec2f& p)
{
    return (sampleMatCubic(target, p) -
        sampleMatCubic(source, p + sampleMatCubic(corr, p).block<2,1>(0,0))).squaredNorm();
}

Vec3f getGradient(const cv::Mat& source, const cv::Mat& target, const cv::Mat& corr, const Vec2f& p, Vec3f& c)
{
    constexpr float eps = 1.0e-2f;

    Vec3f g(0.0f, 0.0f, 0.0f);

    float e, c0;

    c0 = c(0);
    c(0) += eps;
    e = getError(source, target, corr, p);
    c(0) -= 2.0f*eps;
    e -= getError(source, target, corr, p);
    c(0) = c0;
    g(0) = e / (2.0f*eps);

    c0 = c(1);
    c(1) += eps;
    e = getError(source, target, corr, p);
    c(1) -= 2.0f*eps;
    e -= getError(source, target, corr, p);
    c(1) = c0;
    g(1) = e / (2.0f*eps);

    return g;
}

cv::Mat createCorrection4(const cv::Mat& source, const cv::Mat& target)
{
    float gdRate = 512.0f;
    constexpr float gdMomentum = 0.95f;

    cv::Mat corr = source.clone() * 0.0f;
    cv::Mat grad = source.clone() * 0.0f;
    cv::Mat corrImg = source.clone();
    cv::Mat diffImg = cv::abs(corrImg-target);
    cv::Mat sourceBlurred, targetBlurred, diffBlurred;

    int blur = 256;

    for (int k=0; k<6; ++k) {
        cv::GaussianBlur(target, targetBlurred, cv::Size(blur+1, blur+1), 0);
        cv::GaussianBlur(source, sourceBlurred, cv::Size(blur+1, blur+1), 0);

        double maxGradSum = 0.0;
        double gradSum = 0.1;
        double prevGradSum = gradSum;
        double gradSumDiff = 0.1;

        while (gradSum > maxGradSum*0.25 && (gradSumDiff < 0.0 || gradSumDiff > maxGradSum*0.0005)) {
            for (int j = 0; j < sourceBlurred.rows; ++j) {
                auto *cr = corr.ptr<Vec3f>(j);
                auto *gr = grad.ptr<Vec3f>(j);
                for (int i = 0; i < sourceBlurred.cols; ++i) {
                    gr[i] += getGradient(sourceBlurred, targetBlurred, corr, Vec2f(i, j), cr[i]);
                }
            }

            for (int j = 0; j < sourceBlurred.rows; ++j) {
                auto *cr = corr.ptr<Vec3f>(j);
                auto *gr = grad.ptr<Vec3f>(j);
                for (int i = 0; i < sourceBlurred.cols; ++i) {
                    cr[i] -= gr[i] * gdRate;
                }
            }

            cv::imshow("grad", grad * -50.0f + 0.5f);

            gradSum = cv::sum(cv::abs(grad))[0];
            if (maxGradSum < gradSum) maxGradSum = gradSum;
            gradSumDiff = prevGradSum-gradSum;
            prevGradSum = gradSum;
            printf("k: %d gradSum: %0.10f maxGradSum: %0.10f gradSumDiff: %0.10f\n",
                k, gradSum, maxGradSum, gradSumDiff);

            cv::GaussianBlur(grad, grad, cv::Size(int(blur/8)*2+1, int(blur/8)*2+1), 0);
            cv::GaussianBlur(corr, corr, cv::Size(int(blur/4)*2+1, int(blur/4)*2+1), 0);

            corrImg = correctImage(source, corr);
            diffImg = cv::abs(corrImg - target);
            cv::imshow("corr", corr * 0.025f + 0.5f);
            cv::imshow("corrected", corrImg);
            cv::imshow("diff", diffImg);
            cv::waitKey(20);

            grad *= gdMomentum;
        }

        blur /= 2;
        gdRate /= 2.0;
    }

    return corr;
}