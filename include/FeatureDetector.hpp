//
// Project: image_demorphing
// File: FeatureDetector.hpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#ifndef IMAGE_DEMORPHING_FEATUREDETECTOR_HPP
#define IMAGE_DEMORPHING_FEATUREDETECTOR_HPP


#include "TrainFeatureDetector.hpp"
#include "NeuralNetwork.hpp"


template <template <typename> class T_Optimizer>
class FeatureDetector {
private:
    using Layer1 = LayerConv<float, ActivationReLU, T_Optimizer, 9*Feature::fsd, Feature::fsn, 64>;
    using Layer2 = LayerConv<float, ActivationReLU, T_Optimizer, 64, Feature::fsn, 64>;
    using Layer3 = LayerMerge<float, ActivationReLU, T_Optimizer, Feature::fsn*2, 64, 64>;
    using Layer4 = LayerMerge<float, ActivationReLU, T_Optimizer, 64, 32, 64>;
    using Layer5 = LayerMerge<float, ActivationReLU, T_Optimizer, 64, 16, 64>;
    using Layer6 = LayerMerge<float, ActivationReLU, T_Optimizer, 64, 8, 64>;
    using Layer7 = LayerMerge<float, ActivationReLU, T_Optimizer, 64, 4, 64>;
    using Layer8 = LayerMerge<float, ActivationReLU, T_Optimizer, 64, 2, 128>;
    using Layer9 = LayerDense<float, ActivationReLU, T_Optimizer, 128, 128>;
    using Layer10 = LayerDense<float, ActivationReLU, T_Optimizer, 128, 64>;
    using Layer11 = LayerDense<float, ActivationLinear, T_Optimizer, 64, 3>;
    using LastLayer = Layer11;

    Layer1  _layer1a, _layer1b;
    Layer2  _layer2a, _layer2b;
    Layer3  _layer3;
    Layer4  _layer4;
    Layer5  _layer5;
    Layer6  _layer6;
    Layer7  _layer7;
    Layer8  _layer8;
    Layer9  _layer9;
    Layer10 _layer10;
    Layer11 _layer11;

    std::vector<typename Layer2::Output, Eigen::aligned_allocator<typename Layer2::Output>>         _v1;
    std::vector<typename Layer2::Output, Eigen::aligned_allocator<typename Layer2::Output>>         _v2;
    std::vector<typename Layer3::Input, Eigen::aligned_allocator<typename Layer3::Input>>           _v3;
    std::vector<typename LastLayer::Output, Eigen::aligned_allocator<typename LastLayer::Output>>   _g;
    std::vector<typename Layer3::Input, Eigen::aligned_allocator<typename Layer3::Input>>           _g3;

public:
    FeatureDetector(double dropoutRate = 0.0);

    double trainBatch(const TrainingBatch& batch);
    void saveWeights(const std::string& directory);
    void loadWeights(const std::string& directory);
    void printInfo();

    typename LastLayer::Output operator()(const Feature& f1, const Feature& f2);

    static typename LastLayer::Output gradient(
        const typename LastLayer::Output& pred, const typename LastLayer::Output& label);

private:
    typename LastLayer::Output trainingForward(const Feature& f1, const Feature& f2);
    float trainingPass(const Feature& f1, const Feature& f2, const typename LastLayer::Output& label);
};


#include "FeatureDetector.inl"


#endif //IMAGE_DEMORPHING_FEATUREDETECTOR_HPP
