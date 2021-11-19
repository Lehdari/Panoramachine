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
    _layer10a(0.1f, ActivationTanh()), _layer10b(ActivationTanh(), _layer10a.getOptimizerPtr()),
    _layer11(0.1f)
{
}

double FeatureDetector::trainBatch(const TrainingBatch& batch)
{
    int n=0;
    double loss = 0.0;
    for (auto* entry : batch) {
        loss += trainingPass(entry->f1, entry->f2, entry->similarity);
        ++n;
    }
    loss /= n;

    _layer1a.getOptimizer()->applyGradients<float>();
    _layer2a.getOptimizer()->applyGradients<float>();
    _layer3a.getOptimizer()->applyGradients<float>();
    _layer4a.getOptimizer()->applyGradients<float>();
    _layer5a.getOptimizer()->applyGradients<float>();
    _layer6a.getOptimizer()->applyGradients<float>();
    _layer7a.getOptimizer()->applyGradients<float>();
    _layer8a.getOptimizer()->applyGradients<float>();
    _layer9a.getOptimizer()->applyGradients<float>();
    _layer10a.getOptimizer()->applyGradients<float>();
    _layer11.getOptimizer()->applyGradients<float>();
    // no need to call for b-layers since the weights are shared

    return loss;
}

float FeatureDetector::trainingPass(const Feature& f1, const Feature& f2, float targetSimilarity)
{
    // forward propagate
    auto v1 = _layer10a(_layer9a(_layer8a(_layer7a(_layer6a(_layer5a(_layer4a(_layer3a(_layer2a(_layer1a(f1.polar))))))))));
    auto v2 = _layer10b(_layer9b(_layer8b(_layer7b(_layer6b(_layer5b(_layer4b(_layer3b(_layer2b(_layer1b(f2.polar))))))))));
    auto v3 = v1-v2;
    float similarityPrediction = _layer11(v3)(0,0);

    // loss
    float loss = similarityPrediction - targetSimilarity;
    loss *= loss;

    // backpropagate
    Layer11::Output g;
    g << similarityPrediction - targetSimilarity; // initial loss gradient
    auto l11g = _layer11.backpropagate(g); // gradient after the layer 11

    _layer1a.backpropagate(
    _layer2a.backpropagate(
    _layer3a.backpropagate(
    _layer4a.backpropagate(
    _layer5a.backpropagate(
    _layer6a.backpropagate(
    _layer7a.backpropagate(
    _layer8a.backpropagate(
    _layer9a.backpropagate(
    _layer10a.backpropagate(l11g))))))))));

    _layer1b.backpropagate(
    _layer2b.backpropagate(
    _layer3b.backpropagate(
    _layer4b.backpropagate(
    _layer5b.backpropagate(
    _layer6b.backpropagate(
    _layer7b.backpropagate(
    _layer8b.backpropagate(
    _layer9b.backpropagate(
    _layer10b.backpropagate(-l11g)))))))))); // apply negative gradient to b branch (negated to form v3)

    return loss;
}
