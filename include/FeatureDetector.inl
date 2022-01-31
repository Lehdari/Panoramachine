//
// Project: panoramachine
// File: FeatureDetector.inl
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//


template <template <typename> class T_Optimizer>
FeatureDetector<T_Optimizer>::FeatureDetector(double dropoutRate) :
    _layer1a(0.01f, ActivationReLU(0.01f), dropoutRate), _layer1b(ActivationReLU(0.01f), _layer1a.getOptimizerPtr()),
    _layer2a(0.01f, ActivationReLU(0.01f), dropoutRate), _layer2b(ActivationReLU(0.01f), _layer2a.getOptimizerPtr()),
    _layer3(0.01f, ActivationReLU(0.01f), dropoutRate),
    _layer4(0.01f, ActivationReLU(0.01f), dropoutRate/2.0f),
    _layer5(0.01f, ActivationReLU(0.01f), dropoutRate/2.0f),
    _layer6(0.01f, ActivationReLU(0.01f), dropoutRate/2.0f),
    _layer7(0.01f, ActivationReLU(0.01f), dropoutRate/4.0f),
    _layer8(0.01f, ActivationReLU(0.01f), dropoutRate/4.0f),
    _layer9(0.01f, ActivationReLU(0.01f), dropoutRate/4.0f),
    _layer10(0.01f, ActivationReLU(0.01f)),
    _layer11(0.01f, ActivationLinear()),
    _v1(omp_get_max_threads(), decltype(_v1)::value_type::Zero()),
    _v2(omp_get_max_threads(), decltype(_v2)::value_type::Zero()),
    _v3(omp_get_max_threads(), decltype(_v3)::value_type::Zero()),
    _g(omp_get_max_threads(), decltype(_g)::value_type::Zero()),
    _g3(omp_get_max_threads(), decltype(_g3)::value_type::Zero())
{
}

template <template <typename> class T_Optimizer>
double FeatureDetector<T_Optimizer>::trainBatch(
    const FeatureDataset::ConstIterator& begin, const FeatureDataset::ConstIterator& end)
{
    double loss = 0.0;
    #pragma omp parallel for
    for (FeatureDataset::ConstIterator it = begin; it != end; ++it) {
        auto& entry = *it;
        double l = trainingPass(*entry.f1, *entry.f2, *entry.label);
        #pragma omp critical
        loss += l;
    }
    loss /= std::distance(begin, end);

    constexpr float learningRate = 0.001f;
    constexpr float momentum = 0.9f;
    constexpr float momentum2 = 0.99f;
    constexpr float weightDecay = 0.0005f;

    _layer1a.getOptimizer()->template applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer2a.getOptimizer()->template applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer3.getOptimizer()->template applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer4.getOptimizer()->template applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer5.getOptimizer()->template applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer6.getOptimizer()->template applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer7.getOptimizer()->template applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer8.getOptimizer()->template applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer9.getOptimizer()->template applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer10.getOptimizer()->template applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    _layer11.getOptimizer()->template applyGradients<float>(learningRate, momentum, momentum2, weightDecay);
    // no need to call for b-layers since the weights are shared

    return loss;
}

template <template <typename> class T_Optimizer>
void FeatureDetector<T_Optimizer>::saveWeights(const std::string& directory)
{
    _layer1a.getOptimizer()->saveWeights(directory + "/layer1.bin");
    _layer2a.getOptimizer()->saveWeights(directory + "/layer2.bin");
    _layer3.getOptimizer()->saveWeights(directory + "/layer3.bin");
    _layer4.getOptimizer()->saveWeights(directory + "/layer4.bin");
    _layer5.getOptimizer()->saveWeights(directory + "/layer5.bin");
    _layer6.getOptimizer()->saveWeights(directory + "/layer6.bin");
    _layer7.getOptimizer()->saveWeights(directory + "/layer7.bin");
    _layer8.getOptimizer()->saveWeights(directory + "/layer8.bin");
    _layer9.getOptimizer()->saveWeights(directory + "/layer9.bin");
    _layer10.getOptimizer()->saveWeights(directory + "/layer10.bin");
    _layer11.getOptimizer()->saveWeights(directory + "/layer11.bin");
}

template <template <typename> class T_Optimizer>
void FeatureDetector<T_Optimizer>::loadWeights(const std::string& directory)
{
    _layer1a.getOptimizer()->loadWeights(directory + "/layer1.bin");
    _layer2a.getOptimizer()->loadWeights(directory + "/layer2.bin");
    _layer3.getOptimizer()->loadWeights(directory + "/layer3.bin");
    _layer4.getOptimizer()->loadWeights(directory + "/layer4.bin");
    _layer5.getOptimizer()->loadWeights(directory + "/layer5.bin");
    _layer6.getOptimizer()->loadWeights(directory + "/layer6.bin");
    _layer7.getOptimizer()->loadWeights(directory + "/layer7.bin");
    _layer8.getOptimizer()->loadWeights(directory + "/layer8.bin");
    _layer9.getOptimizer()->loadWeights(directory + "/layer9.bin");
    _layer10.getOptimizer()->loadWeights(directory + "/layer10.bin");
    _layer11.getOptimizer()->loadWeights(directory + "/layer11.bin");
}

template <template <typename> class T_Optimizer>
void FeatureDetector<T_Optimizer>::printInfo()
{
    printf("layer1 max w abs: %0.5f\n", _layer1a.getOptimizer()->w.cwiseAbs().maxCoeff());
    printf("layer2 max w abs: %0.5f\n", _layer2a.getOptimizer()->w.cwiseAbs().maxCoeff());
    printf("layer3 max w abs: %0.5f\n", _layer3.getOptimizer()->w.cwiseAbs().maxCoeff());
    printf("layer4 max w abs: %0.5f\n", _layer4.getOptimizer()->w.cwiseAbs().maxCoeff());
    printf("layer5 max w abs: %0.5f\n", _layer5.getOptimizer()->w.cwiseAbs().maxCoeff());
    printf("layer6 max w abs: %0.5f\n", _layer6.getOptimizer()->w.cwiseAbs().maxCoeff());
    printf("layer7 max w abs: %0.5f\n", _layer7.getOptimizer()->w.cwiseAbs().maxCoeff());
    printf("layer8 max w abs: %0.5f\n", _layer8.getOptimizer()->w.cwiseAbs().maxCoeff());
    printf("layer9 max w abs: %0.5f\n", _layer9.getOptimizer()->w.cwiseAbs().maxCoeff());
    printf("layer10 max w abs: %0.5f\n", _layer10.getOptimizer()->w.cwiseAbs().maxCoeff());
    printf("layer11 max w abs: %0.5f\n", _layer11.getOptimizer()->w.cwiseAbs().maxCoeff());
}

template <template <typename> class T_Optimizer>
typename FeatureDetector<T_Optimizer>::LastLayer::Output
    FeatureDetector<T_Optimizer>::operator()(const Feature& f1, const Feature& f2)
{
    _v1[omp_get_thread_num()] = _layer2a(_layer1a(f1.polar));
    _v2[omp_get_thread_num()] = _layer2b(_layer1b(f2.polar));
    _v3[omp_get_thread_num()] << _v1[omp_get_thread_num()].transpose(), _v2[omp_get_thread_num()].transpose();
    return _layer11(_layer10(_layer9(_layer8(_layer7(_layer6(_layer5(_layer4(_layer3(
        _v3[omp_get_thread_num()])))))))));
}

template <template <typename> class T_Optimizer>
typename FeatureDetector<T_Optimizer>::LastLayer::Output
    FeatureDetector<T_Optimizer>::gradient(
    const typename LastLayer::Output& pred, const typename LastLayer::Output& label)
{
    typename LastLayer::Output g = pred - label;
    if (label.template block<2,1>(0,0).norm() > 1.0f) {
        // move outside the unit circle in case the prediction is inside it
        if (pred.template block<2,1>(0,0).norm() > 1.0f)
            g.template block<2,1>(0,0) << 0.0f, 0.0f;
        else
            g.template block<2,1>(0,0) =
                pred.template block<2,1>(0,0) - pred.template block<2,1>(0,0).normalized() * 2.0f;
    }
    return g;
}


template <template <typename> class T_Optimizer>
typename FeatureDetector<T_Optimizer>::LastLayer::Output
    FeatureDetector<T_Optimizer>::trainingForward(const Feature& f1, const Feature& f2)
{
    _v1[omp_get_thread_num()] =
        _layer2a.trainingForward(
        _layer1a.trainingForward(f1.polar));
    _v2[omp_get_thread_num()] =
        _layer2b.trainingForward(
        _layer1b.trainingForward(f2.polar));
    _v3[omp_get_thread_num()] << _v1[omp_get_thread_num()].transpose(), _v2[omp_get_thread_num()].transpose();
    return _layer11.trainingForward(
        _layer10.trainingForward(
        _layer9.trainingForward(
        _layer8.trainingForward(
        _layer7.trainingForward(
        _layer6.trainingForward(
        _layer5.trainingForward(
        _layer4.trainingForward(
        _layer3.trainingForward(
        _v3[omp_get_thread_num()]
        )))))))));
}

template <template <typename> class T_Optimizer>
float FeatureDetector<T_Optimizer>::trainingPass(
    const Feature& f1, const Feature& f2, const typename LastLayer::Output& label)
{
    // forward propagate
    typename LastLayer::Output pred = trainingForward(f1, f2);

    // backpropagate
    _g[omp_get_thread_num()] = gradient(pred, label);
#if 0
    printf("pred: %12.8f %12.8f %12.8f label: %12.8f %12.8f %12.8f grad: %12.8f %12.8f %12.8f\n",
        pred(0), pred(1), pred(2), label(0), label(1), label(2),
        _g[omp_get_thread_num()](0), _g[omp_get_thread_num()](1), _g[omp_get_thread_num()](2));
#endif
    _g3[omp_get_thread_num()] =
    _layer3.backpropagate(
    _layer4.backpropagate(
    _layer5.backpropagate(
    _layer6.backpropagate(
    _layer7.backpropagate(
    _layer8.backpropagate(
    _layer9.backpropagate(
    _layer10.backpropagate(
    _layer11.backpropagate(_g[omp_get_thread_num()])))))))));

    _layer1a.backpropagate(
    _layer2a.backpropagate(_g3[omp_get_thread_num()].template block<Feature::fsn, 64>(0,0).transpose()));
    _layer1b.backpropagate(
    _layer2b.backpropagate(_g3[omp_get_thread_num()].template block<Feature::fsn, 64>(Feature::fsn,0).transpose()));

    return _g[omp_get_thread_num()].array().square().sum();
}
