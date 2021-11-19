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


class FeatureDetector {
public:
    FeatureDetector();

    double trainBatch(const TrainingBatch& batch);


private:
    using Layer1 = LayerMerge<float, ActivationReLU, OptimizerAdam, Feature::fsa*6, Feature::fsr, 64>;
    using Layer2 = LayerMerge<float, ActivationReLU, OptimizerAdam, 64, Feature::fsr/2, 32>;
    using Layer3 = LayerMerge<float, ActivationReLU, OptimizerAdam, 32, Feature::fsr/4, 32>;
    using Layer4 = LayerMerge<float, ActivationReLU, OptimizerAdam, 32, Feature::fsr/8, 32>;
    using Layer5 = LayerMerge<float, ActivationReLU, OptimizerAdam, 32, Feature::fsr/16, 32>;
    using Layer6 = LayerMerge<float, ActivationReLU, OptimizerAdam, 32, Feature::fsr/32, 32>;
    using Layer7 = LayerMerge<float, ActivationReLU, OptimizerAdam, 32, Feature::fsr/64, 32>;
    using Layer8 = LayerDense<float, ActivationReLU, OptimizerAdam, 32, 32>;
    using Layer9 = LayerDense<float, ActivationReLU, OptimizerAdam, 32, 32>;
    using Layer10 = LayerDense<float, ActivationTanh, OptimizerAdam, 32, 16>;
    using Layer11 = LayerDense<float, ActivationTanh, OptimizerAdam, 16, 1>;

    Layer1  _layer1a, _layer1b;
    Layer2  _layer2a, _layer2b;
    Layer3  _layer3a, _layer3b;
    Layer4  _layer4a, _layer4b;
    Layer5  _layer5a, _layer5b;
    Layer6  _layer6a, _layer6b;
    Layer7  _layer7a, _layer7b;
    Layer8  _layer8a, _layer8b;
    Layer9  _layer9a, _layer9b;
    Layer10 _layer10a, _layer10b;
    Layer11 _layer11;


    float trainingPass(const Feature& f1, const Feature& f2, float targetSimilarity);
};


#endif //IMAGE_DEMORPHING_FEATUREDETECTOR_HPP
