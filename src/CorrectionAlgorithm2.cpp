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
#include <random>
#include <iostream> // TODO temp
#include <KdTree.hpp>


struct Feature {
    static constexpr int fsa = 32; // feature axial size
    static constexpr int fsr = 128; // feature radial size
    static constexpr float frf = 1.0f; // first feature radius
    static constexpr float frm = 1.189207115f; // feature radius multiplier
#if 0
    cv::Mat polar;
    cv::Mat amplitude;
    cv::Mat phase;
#endif
    Eigen::Matrix<float, fsa*6, fsr/2>  flattenedFft;
    Eigen::Matrix<float, 2, fsr/2>      intermediate;
    Vec2f                               projected;

    std::vector<Vec2f>                  targets;
    std::vector<Vec2f>                  antiTargets;

    using PMatrix1 = Eigen::Matrix<float, 2, fsa*6>;
    using PMatrix2 = Eigen::Matrix<float, fsr/2, 1>;

    static PMatrix1 projectionMatrix1;
    static PMatrix2 projectionMatrix2;

    void update(void)
    {
        intermediate = Feature::projectionMatrix1 * flattenedFft;
        projected = intermediate * Feature::projectionMatrix2;
    }
};

Eigen::Matrix<float, 2, Feature::fsa*6> Feature::projectionMatrix1 = Eigen::Matrix<float, 2, Feature::fsa*6>::Random();
Eigen::Matrix<float, Feature::fsr/2, 1> Feature::projectionMatrix2 = Eigen::Matrix<float, Feature::fsr/2, 1>::Random();

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


void fourierTransform(const Vec3f* f, std::complex<double> fftRow[Feature::fsr/2][3])
{
    static std::vector<FFTWPlan> plans = [&](){
        std::vector<FFTWPlan> plans;
        plans.reserve(omp_get_num_threads());
        for (int i=0; i<omp_get_num_threads(); ++i)
            plans.emplace_back(Feature::fsr);
        return plans;
    }();

    auto& p = plans[omp_get_thread_num()];
    for (int c=0; c<3; ++c) {
        for (int i=0; i<Feature::fsr; ++i) {
            p.in[i][0] = f[i](c);
            p.in[i][1] = 0.0;
        }

        p.execute();

        for (int i=0; i<Feature::fsr/2; ++i) {
            fftRow[i][c] = *reinterpret_cast<std::complex<double>*>(&p.out[i]);
        }
    }
}

inline __attribute__((always_inline)) double phaseDiff(
    const std::complex<double>& a, const std::complex<double>& b)
{
    double diff = std::atan2(b.imag(), b.real()) - std::atan2(a.imag(), a.real());
    if (diff < -M_PI)
        diff += 2.0*M_PI;
    if (diff > M_PI)
        diff -= 2.0*M_PI;
    return diff;
}

Feature computeFeature(const cv::Mat& img, int x, int y)
{
    Feature feature;
    cv::Mat polar = cv::Mat(Feature::fsa, Feature::fsr, CV_32FC3);
#if 0
    feature.amplitude = cv::Mat(Feature::fsa, Feature::fsr/2, CV_32FC3);
    feature.phase = cv::Mat(Feature::fsa, Feature::fsr/2, CV_32FC3);
#endif

    std::complex<double> fft[Feature::fsa][Feature::fsr/2][3];

    float r = Feature::frf;
    for (int j=0; j<Feature::fsa; ++j) {
        auto* f = polar.ptr<Vec3f>(j);
        for (int i=0; i<Feature::fsr; ++i) {
            float angle = 2.0f*M_PI*(i/(float)Feature::fsr);
            f[i] = sampleMatCubic(img, Vec2f(x+r*std::cos(angle), y+r*std::sin(angle)));
        }

        fourierTransform(f, fft[j]);

        r *= Feature::frm;
    }

    for (int j=0; j<Feature::fsa; ++j) {
#if 0
        auto* pa = feature.amplitude.ptr<Vec3f>(j);
        auto* pp = feature.phase.ptr<Vec3f>(j);
#endif
        for (int i=0; i<Feature::fsr/2; ++i) {
#if 0
            pa[i] = Vec3f(
                std::log(std::abs(fft[j][i][0])),
                std::log(std::abs(fft[j][i][1])),
                std::log(std::abs(fft[j][i][2])));
            if (i < Feature::fsr/2-1) {
                pp[i] = Vec3f(
                    0.5f + M_1_PI * phaseDiff(fft[j][i][0], fft[j][i + 1][0]),
                    0.5f + M_1_PI * phaseDiff(fft[j][i][1], fft[j][i + 1][1]),
                    0.5f + M_1_PI * phaseDiff(fft[j][i][2], fft[j][i + 1][2]));
            }
            else {
                pp[i] = Vec3f(0.0f, 0.0f, 0.0f);
            }
#endif

            for (int c=0; c<3; ++c) {
                feature.flattenedFft(j+c*Feature::fsa, i) = std::abs(fft[j][i][c]);
                if (i < Feature::fsr/2-1)
                    feature.flattenedFft(j+(c+3)*Feature::fsa, i) =
                        phaseDiff(fft[j][i][0], fft[j][i+1][0]);
                else
                    feature.flattenedFft(j+(c+3)*Feature::fsa, i) = 0.0f;
            }
        }
    }

    feature.update();

    //printf("projected: [ %0.5f %0.5f ]\n", feature.projected(0), feature.projected(1));

    return feature;
}

void clickCallBack(int event, int x, int y, int flags, void* imagePtr)
{
    auto& img = *static_cast<const cv::Mat*>(imagePtr);

    Feature feature = computeFeature(img, x, y);
#if 0
    cv::imshow("feature.polar", feature.polar);
    cv::imshow("feature.amplitude", feature.amplitude);
    cv::imshow("feature.phase", feature.phase);
#endif
}

cv::Mat createCorrection2(const cv::Mat& source, const cv::Mat& target)
{
    cv::Mat corr = source.clone() * 0.0f;
#if 0
    cv::namedWindow("source");
    cv::setMouseCallback("source", clickCallBack, (void*)&source);

    cv::imshow("source", source);
    cv::waitKey();

#elif 1

    std::default_random_engine rnd(1507715517);

    cv::Mat projectedFeatures = source.clone() * 0.0f;

    constexpr int nFeatures = 10000;
    std::vector<Feature> features;
    for (int i=0; i<nFeatures; ++i) {
        int sx = rnd() % source.cols;
        int sy = rnd() % source.rows;

        features.emplace_back(computeFeature(source, sx, sy));
    }

    for (int o=0; o<10000; ++o) {
        printf("o: %d\n", o);
        projectedFeatures *= 0.0f;

        for (auto& feature: features) {
            int px = source.cols / 2 + feature.projected(0);
            int py = source.rows / 2 + feature.projected(1);
            if (px < 0 || px >= source.cols || py < 0 || py >= source.rows)
                continue;

            projectedFeatures.at<Vec3f>(py, px) = Vec3f(1.0f, 1.0f, 1.0f);
        }

        KdTree<Vec2f> tree;
        for (auto& feature : features) {
            feature.update();
            feature.targets.clear();
            feature.antiTargets.clear();
            tree.addPoint(feature.projected);
        }
        tree.build();

        Vec2f globalCentroid(0.0f, 0.0f);
        for (auto& feature : features) {
            globalCentroid += feature.projected;
        }
        globalCentroid /= nFeatures;

        constexpr int neighbourhoodSize = 10;
        std::vector<const Vec2f*> nearest;
        for (auto& feature : features) {
            tree.getKNearest(feature.projected, neighbourhoodSize, nearest);

            Vec2f centroid(0.0f, 0.0f);
            for (auto& p : nearest)
                centroid += *p;
            centroid /= neighbourhoodSize;

            feature.targets.push_back(centroid);
            feature.targets.emplace_back(0.0f, 0.0f);
            feature.antiTargets.emplace_back(globalCentroid);
        }

        // Gradient descent
        constexpr double gdRate = 1.0e-5 / nFeatures;

        Feature::PMatrix1 gm1 = Feature::PMatrix1::Zero();
        Feature::PMatrix2 gm2 = Feature::PMatrix2::Zero();
        for (auto& feature: features) {
            for (auto& fTarget : feature.targets) {
                gm2 += feature.intermediate.transpose() * (feature.projected - fTarget);
                gm1 += ((feature.projected - fTarget) *
                    (feature.flattenedFft * Feature::projectionMatrix2).transpose());
            }
            for (auto& fAntiTarget : feature.antiTargets) {
                gm2 += feature.intermediate.transpose() * (fAntiTarget - feature.projected);
                gm1 += (fAntiTarget - feature.projected) *
                    (feature.flattenedFft * Feature::projectionMatrix2).transpose();
            }
        }

        Feature::projectionMatrix1 -= gm1 * gdRate;
        Feature::projectionMatrix2 -= gm2 * gdRate;

        cv::imshow("projectedFeatures", projectedFeatures);
        cv::waitKey(20);
    }

    cv::imshow("projectedFeatures", projectedFeatures);
    cv::waitKey();

#elif

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
#endif
    return corr;
}