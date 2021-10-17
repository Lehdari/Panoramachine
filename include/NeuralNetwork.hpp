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


#include <Eigen/Dense>
#include <iostream> // TODO temp


// Hyperbolic tangent activation
struct ActivationTanh
{
    template <typename T>
    inline __attribute__((always_inline)) static auto activation(const T& x)
    {
        return x.array().tanh();
    }

    template <typename T>
    inline __attribute__((always_inline)) static auto gradient(const T& x)
    {
        return T::Ones() - x.array().tanh().square().matrix();
    }
};

// Linear activation function
struct ActivationLinear
{
    template <typename T>
    inline __attribute__((always_inline)) static auto activation(const T& x)
    {
        return x;
    }

    template <typename T>
    inline __attribute__((always_inline)) static auto gradient(const T& x)
    {
        return T::Ones();
    }
};

// Linear activation function
struct ActivationReLU
{
    template <typename T>
    inline __attribute__((always_inline)) static auto activation(const T& x)
    {
        return x.cwiseMax(T::Zero());
    }

    template <typename T>
    inline __attribute__((always_inline)) static auto gradient(const T& x)
    {
        return x.array().sign().matrix()*0.5 + T::Ones()*0.5;
    }
};


// Layer CRTP base class
template <typename T_Derived, typename T_Scalar, typename T_Activation,
    int T_InputRows, int T_InputCols, int T_OutputRows, int T_OutputCols>
class LayerBase
{
public:
    using Input = Eigen::Matrix<T_Scalar, T_InputRows, T_InputCols>;
    using Output = Eigen::Matrix<T_Scalar, T_OutputRows, T_OutputCols>;

    // Forward propagation
    inline __attribute__((always_inline)) Output operator()(const Input& x)
    {
        return static_cast<T_Derived*>(this)->operator()(x);
    }

    // Backpropagation
    inline __attribute__((always_inline)) Input backpropagate(const Output& g)
    {
        return static_cast<T_Derived*>(this)->backpropagate(g);
    }

    // Apply gradients
    inline __attribute__((always_inline)) void applyGradients(T_Scalar learningRate, T_Scalar momentum)
    {
        return static_cast<T_Derived*>(this)->applyGradients(learningRate, momentum);
    }
};


template <typename T_Scalar, typename T_Activation,
    int T_InputRows, int T_OutputRows>
class LayerDense : public LayerBase<LayerDense<
    T_Scalar, T_Activation, T_InputRows, T_OutputRows>,
    T_Scalar, T_Activation, T_InputRows, 1, T_OutputRows, 1>
{
public:
    using Base = LayerBase<LayerDense<
        T_Scalar, T_Activation, T_InputRows, T_OutputRows>,
        T_Scalar, T_Activation, T_InputRows, 1, T_OutputRows, 1>;
    using Input = typename Base::Input;
    using Output = typename Base::Output;
    using InputExtended = Eigen::Matrix<T_Scalar, T_InputRows+1, 1>;
    using Weights = Eigen::Matrix<T_Scalar, T_OutputRows, T_InputRows+1>;

    LayerDense(T_Scalar initAmplitude = 1) :
        _input  (InputExtended::Ones()),
        _w      (Weights::Random() * initAmplitude),
        _wg     (Weights::Zero())
    {
    }

    inline Output operator()(const Input& x)
    {
        _input.template block<T_InputRows,1>(0,0) = x;
        return T_Activation::activation(_w * _input);
    }

    inline Input backpropagate(const Output& g)
    {
        Output ag = T_Activation::gradient(g).cwiseProduct(g);
        _wg += ag * _input.transpose();
        return _w.template block<T_OutputRows, T_InputRows>(0,0).transpose() * ag;
    }

    inline void applyGradients(T_Scalar learningRate, T_Scalar momentum = 0.0)
    {
        _w -= _wg*learningRate;
        _wg *= momentum;

        if (_w.array().abs().maxCoeff() > 1000.0)
            _w *= 0.9999;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    InputExtended   _input;
    Weights         _w;
    Weights         _wg;
};


template <typename T_Scalar, typename T_Activation,
    int T_InputRows, int T_InputCols, int T_OutputRows>
class LayerMerge : public LayerBase<LayerMerge<
    T_Scalar, T_Activation, T_InputRows, T_InputCols, T_OutputRows>,
    T_Scalar, T_Activation, T_InputRows, T_InputCols, T_OutputRows, T_InputCols/2>
{
public:
    using Base = LayerBase<LayerMerge<
        T_Scalar, T_Activation, T_InputRows, T_InputCols, T_OutputRows>,
        T_Scalar, T_Activation, T_InputRows, T_InputCols, T_OutputRows, T_InputCols/2>;
    using Input = typename Base::Input;
    using Output = typename Base::Output;
    using InputModified = Eigen::Matrix<T_Scalar, 2*T_InputRows+1, T_InputCols/2>;
    using Weights = Eigen::Matrix<T_Scalar, T_OutputRows, 2*T_InputRows+1>;

    LayerMerge(T_Scalar initAmplitude = 1) :
        _input  (InputModified::Ones()),
        _w      (Weights::Random() * initAmplitude),
        _wg     (Weights::Zero())
    {
    }

    inline Output operator()(const Input& x)
    {
        _input.template block<T_InputRows,T_InputCols/2>(0,0) =
            x.template block<T_InputRows,T_InputCols/2>(0,0);
        _input.template block<T_InputRows,T_InputCols/2>(T_InputRows,0) =
            x.template block<T_InputRows,T_InputCols/2>(0,T_InputCols/2);
        return T_Activation::activation(_w * _input);
    }

    inline Input backpropagate(const Output& g)
    {
        Output ag = T_Activation::gradient(g).cwiseProduct(g);
        _wg += ag * _input.transpose();
        //_wg = ag * _input.transpose();
        auto igModified = _w.template block<T_OutputRows, 2*T_InputRows>(0,0).transpose() * ag;
        Input ig;
        ig <<
            igModified.template block<T_InputRows, T_InputCols/2>(0,0),
            igModified.template block<T_InputRows, T_InputCols/2>(T_InputRows,0);
        return ig;
    }

    inline void applyGradients(T_Scalar learningRate, T_Scalar momentum = 0.0)
    {
        _w -= _wg*learningRate;
        _wg *= momentum;

        if (_w.array().abs().maxCoeff() > 1000.0)
            _w *= 0.9999;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    InputModified   _input;
    Weights         _w;
    Weights         _wg;
};


#endif //IMAGE_DEMORPHING_NEURALNETWORK_HPP
