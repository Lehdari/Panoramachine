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
#include <random>
#include <KdTree.hpp>


struct Feature {
    static constexpr int fsa = 4; // feature axial size
    static constexpr int fsr = 128; // feature radial size
    static constexpr int is1 = 16; // intermediate size 1
    static constexpr float frf = 1.0f; // first feature radius
    static constexpr float frm = 1.189207115f; // feature radius multiplier

    Eigen::Matrix<float, fsa*3, fsr>    polar;
    Eigen::Matrix<float, fsa*3+1, fsr>  polarDiff;
    Eigen::Matrix<float, is1, fsr>      intermediate1;
    Eigen::Matrix<float, is1+1, 1>      intermediate2;
    Vec2f                               projected;

    Vec2f                               dir;

    using PMatrix1 = Eigen::Matrix<float, is1, fsa*3+1>;
    using PMatrix2 = Eigen::Matrix<float, 2, is1+1>;

    static PMatrix1 projectionMatrix1;
    static PMatrix2 projectionMatrix2;

    static PMatrix1 projectionMatrix1Gradient;
    static PMatrix2 projectionMatrix2Gradient;

    void update(void)
    {
        intermediate1 = (Feature::projectionMatrix1 * polarDiff).array().tanh();
        intermediate2.block<is1,1>(0,0) = intermediate1.rowwise().sum();
        intermediate2(is1, 0) = 1.0f;
        projected = Feature::projectionMatrix2 * intermediate2;
    }

    void computeGradients(const Vec2f& g)
    {
        Feature::projectionMatrix2Gradient += g * intermediate2.transpose();

        constexpr float fsrInv = 1.0f / fsr;
        decltype(intermediate1) intermediateGradient =
            (decltype(intermediate1)::Ones() - (intermediate1.array().pow(2.0)).matrix()).cwiseProduct(
            (Feature::projectionMatrix2.block<2,is1>(0,0).transpose() * g * fsrInv).replicate<1,fsr>());

        Feature::projectionMatrix1Gradient += intermediateGradient * polarDiff.transpose();
    }

    static constexpr float gdMomentum = 0.9f;
    static void applyGradients(float gdRate)
    {
        projectionMatrix1 -= projectionMatrix1Gradient * gdRate;
        projectionMatrix2 -= projectionMatrix2Gradient * gdRate;

        projectionMatrix1Gradient *= Feature::gdMomentum;
        projectionMatrix2Gradient *= Feature::gdMomentum;

        //projectionMatrix1Gradient = Feature::PMatrix1::Zero();
        //projectionMatrix2Gradient = Feature::PMatrix2::Zero();
    }
};

Feature::PMatrix1 Feature::projectionMatrix1 = Feature::PMatrix1::Random()*100.0f;
Feature::PMatrix2 Feature::projectionMatrix2 = Feature::PMatrix2::Random()*1.0f;

Feature::PMatrix1 Feature::projectionMatrix1Gradient = Feature::PMatrix1::Zero();
Feature::PMatrix2 Feature::projectionMatrix2Gradient = Feature::PMatrix2::Zero();


Feature computeFeature(const cv::Mat& img, int x, int y)
{
    Feature feature;
    float r = Feature::frf;
    for (int j=0; j<Feature::fsa; ++j) {
        for (int i=0; i<Feature::fsr; ++i) {
            float angle = 2.0f*M_PI*(i/(float)Feature::fsr);
            feature.polar.block<3,1>(j*3,i) = sampleMatCubic(img, Vec2f(x+r*std::cos(angle), y+r*std::sin(angle)));
        }
        r *= Feature::frm;
    }

    for (int i=0; i<Feature::fsr; ++i) {
        feature.polarDiff.block<Feature::fsa*3, 1>(0, i) =
            (feature.polar.block<Feature::fsa*3, 1>(0, (i+1)%Feature::fsr) -
            feature.polar.block<Feature::fsa*3, 1>(0, i));
        feature.polarDiff(Feature::fsa*3, i) = 1.0f;
    }

    feature.update();
    return feature;
}

cv::Mat createCorrection2(const cv::Mat& source, const cv::Mat& target)
{
    cv::Mat corr = source.clone() * 0.0f;

    std::default_random_engine rnd(1507715517);
    Feature::projectionMatrix1.block<Feature::is1,1>(0,Feature::fsa*3) = Eigen::Matrix<float, Feature::is1,1>::Zero();
    Feature::projectionMatrix2.block<2,1>(0,Feature::is1) = Eigen::Matrix<float, 2,1>::Zero();

    cv::Mat projectedFeatures = source.clone() * 0.0f;

    constexpr int nFeatures = 1000;
    std::vector<Feature> features;
    for (int i=0; i<nFeatures; ++i) {
        int sx = rnd() % source.cols;
        int sy = rnd() % source.rows;

        features.emplace_back(computeFeature(source, sx, sy));
    }

    for (int o=0; o<100000; ++o) {
        printf("o: %d\n", o);
        projectedFeatures *= 0.0f;

        for (auto& feature: features) {
            int px = (source.cols / 2)*(1.0f + feature.projected(0));
            int py = (source.rows / 2)*(1.0f + feature.projected(1));
            if (px < 0 || px >= source.cols || py < 0 || py >= source.rows)
                continue;

            projectedFeatures.at<Vec3f>(py, px) = Vec3f(1.0f, 1.0f, 1.0f);
        }

        KdTree<Vec2f> tree;
        for (auto& feature : features) {
            feature.update();
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

            feature.dir = feature.projected - centroid;
            feature.dir += (globalCentroid - feature.projected)*0.5f;
            feature.dir += feature.projected - feature.projected.normalized()*0.5f;
        }

        // Gradient descent
        constexpr double gdRate = 1.0e-6;

        for (auto& feature: features) {
                feature.computeGradients(feature.dir);
        }

        Feature::applyGradients(gdRate);


        cv::imshow("projectedFeatures", projectedFeatures);
        cv::waitKey(20);
    }

    cv::imshow("projectedFeatures", projectedFeatures);
    cv::waitKey();

    return corr;
}