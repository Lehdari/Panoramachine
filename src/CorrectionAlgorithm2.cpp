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
#include "Utils.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <KdTree.hpp>
#include <NeuralNetwork.hpp>

#if 0
#define RND ((rnd()%1000001)*0.000001)


template <typename T>
using AlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;


struct Feature {
    static constexpr int fsa = 32; // feature axial size
    static constexpr int fsr = 128; // feature radial size
    static constexpr int is1 = 16; // intermediate size 1
    static constexpr float frm = 1.1387886347566f; // feature radius multiplier

    Eigen::Matrix<float, fsa*3, fsr>    polar;
    Eigen::Matrix<float, fsa*3, fsr>    polarDiff;
    Vec2f                               projected;

    Vec2f                               dir;


    using Layer1 = LayerMerge<float, ActivationTanh, fsa*3, fsr, 32>;
    using Layer2 = LayerMerge<float, ActivationTanh, 32, fsr/2, 32>;
    using Layer3 = LayerMerge<float, ActivationTanh, 32, fsr/4, 32>;
    using Layer4 = LayerMerge<float, ActivationTanh, 32, fsr/8, 32>;
    using Layer5 = LayerMerge<float, ActivationTanh, 32, fsr/16, 32>;
    using Layer6 = LayerMerge<float, ActivationTanh, 32, fsr/32, 32>;
    using Layer7 = LayerMerge<float, ActivationTanh, 32, fsr/64, 32>;
    using Layer8 = LayerDense<float, ActivationTanh, 32, 16>;
    using Layer9 = LayerDense<float, ActivationTanh, 16, 8>;
    using Layer10 = LayerDense<float, ActivationTanh, 8, 2>;

    static Layer1 layer1;
    static Layer2 layer2;
    static Layer3 layer3;
    static Layer4 layer4;
    static Layer5 layer5;
    static Layer6 layer6;
    static Layer7 layer7;
    static Layer8 layer8;
    static Layer9 layer9;
    static Layer10 layer10;

    Feature(const cv::Mat& img, int x, int y, float firstRadius)
    {
        float r = firstRadius;
        for (int j=0; j<Feature::fsa; ++j) {
            for (int i=0; i<Feature::fsr; ++i) {
                float angle = 2.0f*M_PI*(i/(float)Feature::fsr);
                polar.block<3,1>(j*3,i) = sampleMatCubic<Vec3f>(img, Vec2f(x+r*std::cos(angle), y+r*std::sin(angle)));
            }
            r *= Feature::frm;
        }

#if 1
        for (int i=0; i<Feature::fsr; ++i) {
            polarDiff.block<Feature::fsa*3, 1>(0, i) =
                polar.block<Feature::fsa*3, 1>(0, (i+1)%Feature::fsr) -
                    polar.block<Feature::fsa*3, 1>(0, i);
        }
#else
        polarDiff = polar;
#endif

        update();
    }

    void update(void)
    {
        projected = layer10(layer9(layer8(layer7(layer6(layer5(layer4(layer3(layer2(layer1(polar))))))))));
    }

    void computeGradients(const Vec2f& g)
    {
        layer1.backpropagate(
        layer2.backpropagate(
        layer3.backpropagate(
        layer4.backpropagate(
        layer5.backpropagate(
        layer6.backpropagate(
        layer7.backpropagate(
        layer8.backpropagate(
        layer9.backpropagate(
        layer10.backpropagate(g))))))))));
    }

    static void applyGradients(float gdRate, float momentum = 0.9f, float momentum2 = 0.999f)
    {
        layer1.applyGradients(gdRate, momentum, momentum2);
        layer2.applyGradients(gdRate, momentum, momentum2);
        layer3.applyGradients(gdRate, momentum, momentum2);
        layer4.applyGradients(gdRate, momentum, momentum2);
        layer5.applyGradients(gdRate, momentum, momentum2);
        layer6.applyGradients(gdRate, momentum, momentum2);
        layer7.applyGradients(gdRate, momentum, momentum2);
        layer8.applyGradients(gdRate, momentum, momentum2);
        layer9.applyGradients(gdRate, momentum, momentum2);
        layer10.applyGradients(gdRate, momentum, momentum2);
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

Feature::Layer1 Feature::layer1(1.0);
Feature::Layer2 Feature::layer2(1.0);
Feature::Layer3 Feature::layer3(1.0);
Feature::Layer4 Feature::layer4(1.0);
Feature::Layer5 Feature::layer5(1.0);
Feature::Layer6 Feature::layer6(1.0);
Feature::Layer7 Feature::layer7(1.0);
Feature::Layer8 Feature::layer8(1.0);
Feature::Layer9 Feature::layer9(1.0);
Feature::Layer10 Feature::layer10(1.0);


cv::Point cvTransform(const cv::Mat& image, const Vec2f& p)
{
    return cv::Point((image.cols/2)*(1.0+p(0)), (image.rows/2)*(1.0+p(1)));
}


struct Cluster {
    Vec2f centroid = Vec2f(0.0f, 0.0f);
    Vec2f target = Vec2f(0.0f, 0.0f);
    std::vector<Feature*> features;

    void updateCentroid(void)
    {
        centroid << 0.0f, 0.0f;
        for (auto& feature : features)
            centroid += feature->projected;
        centroid /= features.size();
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// sample features with uniformly increasing energy levels
void sampleFeatures(const cv::Mat& source, int nFeatureClusters,
    AlignedVector<Feature>& features, AlignedVector<Cluster>& featureClusters)
{
#define SAMPLE_FEATURES_VISUALIZATION 1

    constexpr int sampleMultiplier = 256; // number of initial samples w.r.t. n. of feature clusters
    struct FeatureSample {
        Vec2i   p;
        float   energy;
        Feature f;
    };

    features.clear();
    features.reserve(nFeatureClusters*3);
    featureClusters.clear();

#if SAMPLE_FEATURES_VISUALIZATION
    cv::Mat sourceCopy = source.clone();
#endif

    static std::default_random_engine rnd(1507715517);
    std::vector<FeatureSample>  featureSamples;
    float energyMin = FLT_MAX;
    float energyMax = 0.0f;
    for (int i=0; i<nFeatureClusters*sampleMultiplier; ++i) {
        Vec2i p(rnd()%source.cols, rnd()%source.rows);
        featureSamples.push_back(FeatureSample{
            p, 0.0f,
            Feature(source, p(0), p(1), 2.0f)
        });
        auto& fs = featureSamples.back();
        fs.energy = fs.f.polarDiff.array().abs().sum();

        if (fs.energy < energyMin)
            energyMin = fs.energy;
        if (fs.energy > energyMax)
            energyMax = fs.energy;
    }

    std::sort(featureSamples.begin(), featureSamples.end(),
        [](const FeatureSample& a, const FeatureSample& b){
        return a.energy < b.energy;
    });

    std::vector<FeatureSample*>  featureSamplesFiltered;
    float filterEnergyStep = (energyMax-energyMin)/nFeatureClusters;
    float filterEnergy = filterEnergyStep/2.0f;
    FeatureSample* fsLast = nullptr;
    for (auto& fs : featureSamples) {
        if (fs.energy > filterEnergy) {
            if (fsLast != nullptr && fs.energy-filterEnergy < filterEnergy-fsLast->energy)
                featureSamplesFiltered.emplace_back(&fs);
            else
                featureSamplesFiltered.emplace_back(fsLast);

            filterEnergy += filterEnergyStep;
        }
        fsLast = &fs;
    }

    for (auto* fs : featureSamplesFiltered) {
        featureClusters.emplace_back();
        features.push_back(fs->f);
        featureClusters.back().features.push_back(&features.back());
        features.emplace_back(source, fs->p(0), fs->p(1), 2.0f/Feature::frm);
        featureClusters.back().features.push_back(&features.back());
        features.emplace_back(source, fs->p(0), fs->p(1), 2.0f*Feature::frm);
        featureClusters.back().features.push_back(&features.back());
    }

    int fId = 0;
    for (auto& f : features) {
        f.update();
        std::stringstream fWindowName;
        fWindowName << "feature" << fId++;
        cv::Mat featureImg(Feature::fsr, Feature::fsa, CV_32FC3);
        featureImg.data = reinterpret_cast<unsigned char*>(f.polar.data());
        cv::imshow(fWindowName.str(), featureImg);
    }
    for (auto& fc : featureClusters) {
        fc.updateCentroid();
        fc.target = fc.centroid;
    }

#if SAMPLE_FEATURES_VISUALIZATION
    for (auto* fs : featureSamplesFiltered) {
        float e = (fs->energy-energyMin)/(energyMax-energyMin);
        cv::circle(sourceCopy, cv::Point(fs->p(0), fs->p(1)), 3, cv::Scalar(e, e, e), cv::FILLED);
    }

    cv::imshow("FeaturePositions", sourceCopy);
    cv::waitKey(20);
#endif
}


cv::Mat createCorrection2(const cv::Mat& source, const cv::Mat& target)
{
    constexpr int nFeatureClusters = 4;
    AlignedVector<Feature> features;
    AlignedVector<Cluster> featureClusters;

    cv::Mat corr = source.clone() * 0.0f;
    cv::Mat projectedFeatures = source.clone() * 0.0f;

    for (int o=0; o<10000000; ++o) {
        printf("o: %d\n", o);
        projectedFeatures *= 0.0f;

        if (o%1000 == 0)
            sampleFeatures(source, nFeatureClusters, features, featureClusters);

        auto t1 = std::chrono::high_resolution_clock::now();

        for (auto& feature: features) {
            feature.update();
        }

        auto t2 = std::chrono::high_resolution_clock::now();

        Vec2f globalCentroid(0.0f, 0.0f);
        for (auto& feature : features) {
            globalCentroid += feature.projected;
        }
        globalCentroid /= features.size();

        auto t3 = std::chrono::high_resolution_clock::now();

        cv::circle(projectedFeatures, cvTransform(projectedFeatures, globalCentroid),
            4.0, cv::Scalar(0.0, 0.0, 1.0), 1.0);

        for (auto& cluster : featureClusters) {
            cluster.updateCentroid();
        }

        auto t4 = std::chrono::high_resolution_clock::now();

        std::vector<std::pair<const Vec2f*, Cluster*>> nearest;
        double error = 0.0;
        for (auto& cluster : featureClusters) {
            constexpr float margin = 0.75f;
            for (auto* f : cluster.features) {
                f->dir << 0.0f, 0.0f;

                for (auto& cluster2 : featureClusters) {
                    if (&cluster == &cluster2)
                        continue;
                    Vec2f cd = cluster2.centroid - f->projected;
                    float cdl = cd.norm();
                    if (cdl > 0.0f && cdl < margin)
                        f->dir += (cd/cdl);
                }

                for (auto* f2 : cluster.features) {
                    if (f == f2)
                        continue;
                    f->dir -= (f2->projected - f->projected).normalized();
                }

                // normalize direction towards unit circle
                float projLen = f->projected.norm()+1.0e-8f;
                Vec2f normProj = f->projected/projLen;
                Vec2f dirRadial = (f->dir.dot(normProj))*normProj;
                Vec2f dirAxial = f->dir-dirRadial;
                f->dir = dirAxial + (1.0f-projLen)*dirRadial;

                printf("[ %10.5f, %10.5f, %10.5f, %10.5f ]\n",
                    f->projected(0), f->projected(1), f->dir(0), f->dir(1));

                cv::line(projectedFeatures, cvTransform(projectedFeatures, f->projected),
                    cvTransform(projectedFeatures, f->projected - f->dir*0.1), cv::Scalar(0.35, 0.35, 0.0));
                cv::line(projectedFeatures, cvTransform(projectedFeatures, f->projected),
                    cvTransform(projectedFeatures, cluster.centroid), cv::Scalar(0.15, 0.0, 0.2));
                cv::circle(projectedFeatures, cvTransform(projectedFeatures, f->projected),
                    3.0, cv::Scalar(0.35, 0.35, 0.0), 1.0);
            }

            cv::circle(projectedFeatures, cvTransform(projectedFeatures, cluster.centroid),
                3.0, cv::Scalar(0.75, 0.0, 1.0), 1.0);
#if 0
            cv::circle(projectedFeatures, cvTransform(projectedFeatures, cluster.target),
                3.0, cv::Scalar(1.0, 0.0, 0.0), 1.0);
#endif
        }
        printf("error: %0.10f\n", error);

        auto t5 = std::chrono::high_resolution_clock::now();

        // Gradient descent
        constexpr double gdRate = 1.0e-6;

        //#pragma omp parallel for
        for (auto& feature: features) {
            feature.update();
            feature.computeGradients(feature.dir);
        }

        Feature::applyGradients(gdRate);

        auto t6 = std::chrono::high_resolution_clock::now();


        printf("%8.3f %8.3f %8.3f %8.3f %8.3f\n",
            std::chrono::duration<double, std::milli>(t2-t1).count(),
            std::chrono::duration<double, std::milli>(t3-t2).count(),
            std::chrono::duration<double, std::milli>(t4-t3).count(),
            std::chrono::duration<double, std::milli>(t5-t4).count(),
            std::chrono::duration<double, std::milli>(t6-t5).count());

        cv::imshow("projectedFeatures", projectedFeatures);
        cv::waitKey(10);
    }

    cv::imshow("projectedFeatures", projectedFeatures);
    cv::waitKey();

    return corr;
}
#endif