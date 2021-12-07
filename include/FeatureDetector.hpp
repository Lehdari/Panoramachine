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
public:
    FeatureDetector();

    double trainBatch(const TrainingBatch& batch);
    void saveWeights(const std::string& directory);
    void loadWeights(const std::string& directory);

    float operator()(const Feature& f1, const Feature& f2);

private:
    using Layer1 = LayerMerge<float, ActivationReLU, T_Optimizer, Feature::fsa*4, Feature::fsr, 64>;
    using Layer2 = LayerMerge<float, ActivationReLU, T_Optimizer, 64, Feature::fsr/2, 64>;
    using Layer3 = LayerMerge<float, ActivationReLU, T_Optimizer, 64, Feature::fsr/4, 64>;
    using Layer4 = LayerMerge<float, ActivationReLU, T_Optimizer, 64, Feature::fsr/8, 64>;
    using Layer5 = LayerMerge<float, ActivationReLU, T_Optimizer, 64, Feature::fsr/16, 64>;
    using Layer6 = LayerMerge<float, ActivationReLU, T_Optimizer, 64, Feature::fsr/32, 64>;
    using Layer7 = LayerMerge<float, ActivationReLU, T_Optimizer, 64, Feature::fsr/64, 128>;
    using Layer8 = LayerDense<float, ActivationTanh, T_Optimizer, 128, 64>;
    using Layer9 = LayerDense<float, ActivationTanh, T_Optimizer, 64, 32>;
    using LastLayer = Layer9;

    Layer1  _layer1a, _layer1b;
    Layer2  _layer2a, _layer2b;
    Layer3  _layer3a, _layer3b;
    Layer4  _layer4a, _layer4b;
    Layer5  _layer5a, _layer5b;
    Layer6  _layer6a, _layer6b;
    Layer7  _layer7a, _layer7b;
    Layer8  _layer8a, _layer8b;
    Layer9  _layer9a, _layer9b;

    typename LastLayer::Output _v1;
    typename LastLayer::Output _v2; // feature vectors from a and b branches
    typename LastLayer::Output _diff; // v1-v2
    typename LastLayer::Output _g;

    float trainingPass(const Feature& f1, const Feature& f2, float targetDiff);
};

#include "FeatureDetector.inl"


#endif //IMAGE_DEMORPHING_FEATUREDETECTOR_HPP
