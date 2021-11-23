//
// Project: image_demorphing
// File: NeuralNetwork.hpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#ifndef IMAGE_DEMORPHING_NEURALNETWORK_HPP
#define IMAGE_DEMORPHING_NEURALNETWORK_HPP


#include "MathTypes.hpp"
#include <iostream> // TODO temp


// Hyperbolic tangent activation
struct ActivationTanh
{
    template <typename T>
    inline __attribute__((always_inline)) auto activation(const T& x)
    {
        return x.array().tanh();
    }

    template <typename T>
    inline __attribute__((always_inline)) auto gradient(const T& x)
    {
        return T::Ones() - x.array().tanh().square().matrix();
    }
};

// Linear activation function
struct ActivationLinear
{
    template <typename T>
    inline __attribute__((always_inline)) auto activation(const T& x)
    {
        return x;
    }

    template <typename T>
    inline __attribute__((always_inline)) auto gradient(const T& x)
    {
        return T::Ones();
    }
};

// (Leaky) ReLU activation function
struct ActivationReLU
{
    ActivationReLU(float alpha = 0.0f) :
        _alpha  (alpha),
        _x      ((2.0f/(1.0f-_alpha))-1.0f),
        _y      ((1.0f-_alpha)/2.0f)
    {}

    template <typename T>
    inline __attribute__((always_inline)) auto activation(const T& x)
    {
        return x.cwiseMax(x*_alpha);
    }

    template <typename T>
    inline __attribute__((always_inline)) auto gradient(const T& x)
    {
        return (x.array().sign().matrix() + _x*T::Ones())*_y;
    }

private:
    float   _alpha;
    float   _x;
    float   _y;
};


template <typename T_Weights>
struct OptimizerAdam
{
    template <typename T_Scalar>
    OptimizerAdam(T_Scalar initAmplitude = 1.0) :
        w   (T_Weights::Random() * initAmplitude),
        wg  (T_Weights::Zero()),
        wm  (T_Weights::Zero()),
        wv  (T_Weights::Zero()),
        t   (0)
    {
        w.template block<T_Weights::RowsAtCompileTime,1>(0,T_Weights::ColsAtCompileTime-1) =
            Eigen::Matrix<T_Scalar,T_Weights::RowsAtCompileTime,1>::Zero();
    }

    template <typename T_Scalar>
    inline void applyGradients(
        T_Scalar learningRate = 0.001,
        T_Scalar momentum = 0.9,
        T_Scalar momentum2 = 0.999,
        T_Scalar weightDecay = 0.01)
    {
        const T_Scalar epsilon = 1.0e-8;
        ++t;

        wm = momentum*wm + (1.0-momentum)*wg; // update first moment
        wv = momentum*wv + (1.0-momentum2)*(wg.array().square().matrix()); // update second moment

        T_Scalar alpha = learningRate * (std::sqrt(1.0 - std::pow(momentum2, (T_Scalar)t)) /
            (1.0 - std::pow(momentum, (T_Scalar)t)));

        w -= alpha * wm.template cwiseProduct((wv.cwiseSqrt()+T_Weights::Ones()*epsilon).cwiseInverse());

        if (weightDecay >= epsilon) // weight decay
            w.noalias() = w*(1.0-weightDecay);

        wg = T_Weights::Zero();
    }

    T_Weights   w;
    T_Weights   wg; // gradient
    T_Weights   wm; // first moment
    T_Weights   wv; // second moment
    int         t; // timestep index

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


template <typename T_Scalar, typename T_Activation, template <typename> class T_Optimizer,
    int T_InputRows, int T_OutputRows>
class LayerDense
{
public:
    using Input = Eigen::Matrix<T_Scalar, T_InputRows, 1>;
    using Output = Eigen::Matrix<T_Scalar, T_OutputRows, 1>;
    using InputExtended = Eigen::Matrix<T_Scalar, T_InputRows+1, 1>;
    using Weights = Eigen::Matrix<T_Scalar, T_OutputRows, T_InputRows+1>;

    LayerDense(T_Scalar initAmplitude = 1.0,
        T_Activation&& activation = T_Activation()) :
        _activation (activation),
        _optimizer  (std::make_shared<T_Optimizer<Weights>>(initAmplitude)),
        _input      (InputExtended::Ones())
    {
    }

    LayerDense(
        T_Activation&& activation,
        const std::shared_ptr<T_Optimizer<Weights>>& optimizer) :
        _activation (activation),
        _optimizer  (optimizer),
        _input      (InputExtended::Ones())
    {
    }

    inline Output operator()(const Input& x)
    {
        _input.template block<T_InputRows,1>(0,0) = x;
        _weighed = _optimizer->w * _input;
        return _activation.activation(_weighed);
    }

    inline Input backpropagate(const Output& g)
    {
        Output ag = _activation.gradient(_weighed).cwiseProduct(g);
        _optimizer->wg += ag * _input.transpose();
        return _optimizer->w.template block<T_OutputRows, T_InputRows>(0,0).transpose() * ag;
    }

    T_Optimizer<Weights>* getOptimizer()
    {
        return _optimizer.get();
    }

    const std::shared_ptr<T_Optimizer<Weights>>& getOptimizerPtr()
    {
        return _optimizer;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    T_Activation                            _activation;
    std::shared_ptr<T_Optimizer<Weights>>   _optimizer;
    InputExtended                           _input; // last input
    Output                                  _weighed; // weights * input
};


template <typename T_Scalar, typename T_Activation, template <typename> class T_Optimizer,
    int T_InputRows, int T_InputCols, int T_OutputRows>
class LayerMerge
{
public:
    using Input = Eigen::Matrix<T_Scalar, T_InputRows, T_InputCols>;
    using Output = Eigen::Matrix<T_Scalar, T_OutputRows, T_InputCols/2>;
    using InputModified = Eigen::Matrix<T_Scalar, 2*T_InputRows+1, T_InputCols/2>;
    using Weights = Eigen::Matrix<T_Scalar, T_OutputRows, 2*T_InputRows+1>;

    LayerMerge(T_Scalar initAmplitude = 1.0,
        T_Activation&& activation = T_Activation()) :
        _activation (activation),
        _optimizer  (std::make_shared<T_Optimizer<Weights>>(initAmplitude)),
        _input      (InputModified::Ones())
    {
    }

    LayerMerge(
        T_Activation&& activation,
        const std::shared_ptr<T_Optimizer<Weights>>& optimizer) :
        _activation (activation),
        _optimizer  (optimizer),
        _input      (InputModified::Ones())
    {
    }

    inline Output operator()(const Input& x)
    {
        _input.template block<2*T_InputRows,T_InputCols/2>(0,0) =
            Eigen::Map<const Eigen::Matrix<T_Scalar, 2*T_InputRows, T_InputCols/2>>(x.data());
        _weighed = _optimizer->w * _input;
        return _activation.activation(_weighed);
    }

    inline Input backpropagate(const Output& g)
    {
        Output ag = _activation.gradient(_weighed).cwiseProduct(g);
        _optimizer->wg += ag * _input.transpose();
        Eigen::Matrix<T_Scalar, 2*T_InputRows, T_InputCols/2> igModified =
            _optimizer->w.template block<T_OutputRows, 2*T_InputRows>(0,0).transpose() * ag;
        Input ig = Eigen::Map<Eigen::Matrix<T_Scalar, T_InputRows,T_InputCols>>(igModified.data());
        return ig;
    }

    T_Optimizer<Weights>* getOptimizer()
    {
        return _optimizer.get();
    }

    const std::shared_ptr<T_Optimizer<Weights>>& getOptimizerPtr()
    {
        return _optimizer;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    T_Activation                            _activation;
    std::shared_ptr<T_Optimizer<Weights>>   _optimizer;
    InputModified                           _input; // last input
    Output                                  _weighed; // weights * input
};


#endif //IMAGE_DEMORPHING_NEURALNETWORK_HPP
