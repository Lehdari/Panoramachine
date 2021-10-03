//
// Project: image_demorphing
// File: CorrectionAlgorithm2.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "CorrectionAlgorithms.hpp"
#include <img_demorph_bm/Utils.hpp>
#include <opencv2/highgui.hpp>
#include <fftw3.h>
#include <mutex>


struct Feature {
    cv::Mat polar;
    cv::Mat amplitude;
    cv::Mat phase;
};

class FFTWPlan {
public:
    FFTWPlan(int n) :
        in      ((fftw_complex *) fftw_malloc(sizeof(fftw_complex) * n)),
        out     ((fftw_complex *) fftw_malloc(sizeof(fftw_complex) * n)),
        _plan   (fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE))
    {}

    FFTWPlan(const FFTWPlan&) = default;
    FFTWPlan(FFTWPlan&&) = default;

    ~FFTWPlan()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        fftw_destroy_plan(_plan);
        fftw_free(in);
        fftw_free(out);
    }

    void execute()
    {
        fftw_execute(_plan);
    }

    fftw_complex* const in;
    fftw_complex* const out;

private:
    fftw_plan   _plan;

    static std::mutex   _mutex;
};

std::mutex FFTWPlan::_mutex;


void fourierTransform(const Vec3f* f, Vec3f* va, Vec3f* vp, int n)
{
    static std::vector<FFTWPlan> plans = [&](){
        std::vector<FFTWPlan> plans;
        plans.reserve(omp_get_num_threads());
        for (int i=0; i<omp_get_num_threads(); ++i)
            plans.emplace_back(n);
        return plans;
    }();

    auto& p = plans[omp_get_thread_num()];
    for (int c=0; c<3; ++c) {
        for (int i=0; i<n; ++i) {
            p.in[i][0] = f[i](c);
            p.in[i][1] = 0.0;
        }

        p.execute();

        for (int i=0; i<n; ++i) {
            va[i](c) = std::log(std::sqrt(p.out[i][0]*p.out[i][0] + p.out[i][1]*p.out[i][1]));
            vp[i](c) = 0.5f+0.5f*std::atan2(p.out[i][1], p.out[i][0]);
        }
    }
}

void computeFeature(const cv::Mat& img, int x, int y, Feature& feature)
{
    constexpr int fsa = 32; // feature axial size
    constexpr int fsr = 128; // feature radial size
    constexpr float frf = 1.0f; // first feature radius
    constexpr float frm = 1.189207115f; // feature radius multiplier

    feature.polar = cv::Mat(fsa, fsr, CV_32FC3);
    feature.amplitude = cv::Mat(fsa, fsr, CV_32FC3);
    feature.phase = cv::Mat(fsa, fsr, CV_32FC3);

    float r = frf;
    for (int j=0; j<fsa; ++j) {
        auto* f = feature.polar.ptr<Vec3f>(j);
        for (int i=0; i<fsr; ++i) {
            float angle = 2.0f*M_PI*(i/(float)fsr);
            f[i] = sampleMatCubic(img, Vec2f(x+r*std::cos(angle), y+r*std::sin(angle)));
        }

        fourierTransform(f, feature.amplitude.ptr<Vec3f>(j), feature.phase.ptr<Vec3f>(j), fsr);
        r *= frm;
    }
}

void clickCallBack(int event, int x, int y, int flags, void* imagePtr)
{
    auto& img = *static_cast<const cv::Mat*>(imagePtr);

    Feature feature;
    computeFeature(img, x, y, feature);

    cv::imshow("feature.polar", feature.polar);
    cv::imshow("feature.amplitude", feature.amplitude);
    cv::imshow("feature.phase", feature.phase);
    cv::waitKey(20);
}

cv::Mat createCorrection2(const cv::Mat& source, const cv::Mat& target)
{
    cv::Mat corr = source.clone() * 0.0f;
/*
    cv::namedWindow("source");
    cv::setMouseCallback("source", clickCallBack, (void*)&source);

    cv::imshow("source", source);
    cv::waitKey();
    */

    //std::vector<Feature> features(source.rows * source.cols);

    cv::Mat fImage = source.clone() * 0.0f;

    float featureMax = 0.0f;

    #pragma omp parallel for
    for (int j=0; j<corr.rows; ++j) {
        auto* rf = fImage.ptr<Vec3f>(j);
        for (int i=0; i<corr.cols; ++i) {
            //auto& feature = features[corr.cols*j + i];
            Feature feature;
            computeFeature(source, i, j, feature);

            for (int f=0; f<32; ++f) {
                rf[i](0) += (32-f)*feature.amplitude.at<Vec3f>(f, 1)(0);
                rf[i](1) += (32-f)*feature.amplitude.at<Vec3f>(f, 2)(0);
                rf[i](2) += (32-f)*feature.amplitude.at<Vec3f>(f, 4)(0);
            }

            if (rf[i].maxCoeff() > featureMax)
                featureMax = rf[i].maxCoeff();
        }
        printf("j: %d\n", j);
    }

    if (featureMax > 0.0f)
        fImage /= featureMax;

    cv::imshow("source", source);
    cv::imshow("fImage", fImage);
    cv::waitKey();
    cv::imwrite("fImage.exr", fImage);

    return corr;
}