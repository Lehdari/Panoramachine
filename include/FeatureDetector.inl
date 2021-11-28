//
// Project: image_demorphing
// File: FeatureDetector.inl
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//


template <template <typename> class T_Optimizer>
FeatureDetector<T_Optimizer>::FeatureDetector() :
    _layer1a(0.1f, ActivationReLU(0.01f)), _layer1b(ActivationReLU(0.01f), _layer1a.getOptimizerPtr()),
    _layer2a(0.1f, ActivationReLU(0.01f)), _layer2b(ActivationReLU(0.01f), _layer2a.getOptimizerPtr()),
    _layer3a(0.1f, ActivationReLU(0.01f)), _layer3b(ActivationReLU(0.01f), _layer3a.getOptimizerPtr()),
    _layer4a(0.1f, ActivationReLU(0.01f)), _layer4b(ActivationReLU(0.01f), _layer4a.getOptimizerPtr()),
    _layer5a(0.1f, ActivationReLU(0.01f)), _layer5b(ActivationReLU(0.01f), _layer5a.getOptimizerPtr()),
    _layer6a(0.1f, ActivationReLU(0.01f)), _layer6b(ActivationReLU(0.01f), _layer6a.getOptimizerPtr()),
    _layer7a(0.1f, ActivationTanh()), _layer7b(ActivationTanh(), _layer7a.getOptimizerPtr()),
    _layer8a(0.1f, ActivationTanh()), _layer8b(ActivationTanh(), _layer8a.getOptimizerPtr())
{
}

template <template <typename> class T_Optimizer>
double FeatureDetector<T_Optimizer>::trainBatch(const TrainingBatch& batch)
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
    _layer1a.getOptimizer()->template applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer2a.getOptimizer()->template applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer3a.getOptimizer()->template applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer4a.getOptimizer()->template applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer5a.getOptimizer()->template applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer6a.getOptimizer()->template applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer7a.getOptimizer()->template applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer8a.getOptimizer()->template applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    // no need to call for b-layers since the weights are shared

    return loss;
}

template <template <typename> class T_Optimizer>
void FeatureDetector<T_Optimizer>::saveWeights(const std::string& directory)
{
    _layer1a.getOptimizer()->saveWeights(directory + "/layer1.bin");
    _layer2a.getOptimizer()->saveWeights(directory + "/layer2.bin");
    _layer3a.getOptimizer()->saveWeights(directory + "/layer3.bin");
    _layer4a.getOptimizer()->saveWeights(directory + "/layer4.bin");
    _layer5a.getOptimizer()->saveWeights(directory + "/layer5.bin");
    _layer6a.getOptimizer()->saveWeights(directory + "/layer6.bin");
    _layer7a.getOptimizer()->saveWeights(directory + "/layer7.bin");
    _layer8a.getOptimizer()->saveWeights(directory + "/layer8.bin");
}

template <template <typename> class T_Optimizer>
void FeatureDetector<T_Optimizer>::loadWeights(const std::string& directory)
{
    _layer1a.getOptimizer()->loadWeights(directory + "/layer1.bin");
    _layer2a.getOptimizer()->loadWeights(directory + "/layer2.bin");
    _layer3a.getOptimizer()->loadWeights(directory + "/layer3.bin");
    _layer4a.getOptimizer()->loadWeights(directory + "/layer4.bin");
    _layer5a.getOptimizer()->loadWeights(directory + "/layer5.bin");
    _layer6a.getOptimizer()->loadWeights(directory + "/layer6.bin");
    _layer7a.getOptimizer()->loadWeights(directory + "/layer7.bin");
    _layer8a.getOptimizer()->loadWeights(directory + "/layer8.bin");
}

template <template <typename> class T_Optimizer>
float FeatureDetector<T_Optimizer>::operator()(const Feature& f1, const Feature& f2)
{
    _v1 = _layer8a(_layer7a(_layer6a(_layer5a(_layer4a(_layer3a(_layer2a(_layer1a(f1.polar))))))));
    _v2 = _layer8b(_layer7b(_layer6b(_layer5b(_layer4b(_layer3b(_layer2b(_layer1b(f2.polar))))))));
    _diff = _v1-_v2;
    return _diff.norm();
}

template <template <typename> class T_Optimizer>
float FeatureDetector<T_Optimizer>::trainingPass(const Feature& f1, const Feature& f2, float targetDiff)
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
        _layer8a.backpropagate(_g))))))));

        _layer1b.backpropagate(
        _layer2b.backpropagate(
        _layer3b.backpropagate(
        _layer4b.backpropagate(
        _layer5b.backpropagate(
        _layer6b.backpropagate(
        _layer7b.backpropagate(
        _layer8b.backpropagate(-_g)))))))); // apply negative gradient to b branch (negated to form diff)
    }

    return loss*loss;
}
