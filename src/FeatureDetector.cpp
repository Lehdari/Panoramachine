//
// Project: image_demorphing
// File: FeatureDetector.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "FeatureDetector.hpp"


FeatureDetector::FeatureDetector() :
    _layer1a(0.1f, ActivationReLU(0.01f)), _layer1b(ActivationReLU(0.01f), _layer1a.getOptimizerPtr()),
    _layer2a(0.1f, ActivationReLU(0.01f)), _layer2b(ActivationReLU(0.01f), _layer2a.getOptimizerPtr()),
    _layer3a(0.1f, ActivationReLU(0.01f)), _layer3b(ActivationReLU(0.01f), _layer3a.getOptimizerPtr()),
    _layer4a(0.1f, ActivationReLU(0.01f)), _layer4b(ActivationReLU(0.01f), _layer4a.getOptimizerPtr()),
    _layer5a(0.1f, ActivationReLU(0.01f)), _layer5b(ActivationReLU(0.01f), _layer5a.getOptimizerPtr()),
    _layer6a(0.1f, ActivationReLU(0.01f)), _layer6b(ActivationReLU(0.01f), _layer6a.getOptimizerPtr()),
    _layer7a(0.1f, ActivationReLU(0.01f)), _layer7b(ActivationReLU(0.01f), _layer7a.getOptimizerPtr()),
    _layer8a(0.1f, ActivationReLU(0.01f)), _layer8b(ActivationReLU(0.01f), _layer8a.getOptimizerPtr()),
    _layer9a(0.1f, ActivationReLU(0.01f)), _layer9b(ActivationReLU(0.01f), _layer9a.getOptimizerPtr()),
    _layer10a(0.1f, ActivationTanh()), _layer10b(ActivationTanh(), _layer10a.getOptimizerPtr())
{
}

double FeatureDetector::trainBatch(const TrainingBatch& batch)
{
    int n=0;
    double loss = 0.0;
    for (auto* entry : batch) {
        loss += trainingPass(entry->f1, entry->f2, entry->diff);
        ++n;
    }
    loss /= n;

    constexpr float learningRate = 0.001f;
    constexpr float momentum = 0.9f;
    constexpr float momentum2 = 0.999f;
    constexpr float weightDecay = 0.01f;
    _layer1a.getOptimizer()->applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer2a.getOptimizer()->applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer3a.getOptimizer()->applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer4a.getOptimizer()->applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer5a.getOptimizer()->applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer6a.getOptimizer()->applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer7a.getOptimizer()->applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer8a.getOptimizer()->applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer9a.getOptimizer()->applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer10a.getOptimizer()->applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    // no need to call for b-layers since the weights are shared

    return loss;
}

float FeatureDetector::operator()(const Feature& f1, const Feature& f2)
{
    _v1 = _layer10a(_layer9a(_layer8a(_layer7a(_layer6a(_layer5a(_layer4a(_layer3a(_layer2a(_layer1a(f1.polar))))))))));
    _v2 = _layer10b(_layer9b(_layer8b(_layer7b(_layer6b(_layer5b(_layer4b(_layer3b(_layer2b(_layer1b(f2.polar))))))))));
    _diff = _v1-_v2;
    return _diff.norm();
}

float FeatureDetector::trainingPass(const Feature& f1, const Feature& f2, float targetDiff)
{
    // forward propagate
    float diffPrediction = operator()(f1, f2);

    // loss
    float loss = diffPrediction - targetDiff;
    if (targetDiff > 0.9 && loss > 0.0)
        loss = 0.0;
    else {

        // backpropagate
        constexpr float epsilon = 1.0e-8f; // epsilon to prevent division by zero
        _g = (loss*_diff)/(diffPrediction+epsilon); // initial gradient over distance function

        _layer1a.backpropagate(
        _layer2a.backpropagate(
        _layer3a.backpropagate(
        _layer4a.backpropagate(
        _layer5a.backpropagate(
        _layer6a.backpropagate(
        _layer7a.backpropagate(
        _layer8a.backpropagate(
        _layer9a.backpropagate(
        _layer10a.backpropagate(_g))))))))));

        _layer1b.backpropagate(
        _layer2b.backpropagate(
        _layer3b.backpropagate(
        _layer4b.backpropagate(
        _layer5b.backpropagate(
        _layer6b.backpropagate(
        _layer7b.backpropagate(
        _layer8b.backpropagate(
        _layer9b.backpropagate(
        _layer10b.backpropagate(-_g)))))))))); // apply negative gradient to b branch (negated to form diff)
    }

    return loss*loss;
}
