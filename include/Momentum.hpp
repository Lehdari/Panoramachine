//
// Project: image_demorphing
// File: Momentum.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#ifndef IMAGE_DEMORPHING_MOMENTUM_HPP
#define IMAGE_DEMORPHING_MOMENTUM_HPP


#include <Eigen/Dense>


template <typename T>
class Momentum {
public:
    Momentum(double momentum, const T& v = T()) :
        _momentum   (momentum),
        _variance   (0.0*T()),
        _v          (v)
    {}

    const T& operator()(const T& v)
    {
        T variance = std::pow(v-_v, 2.0);
        _variance = _momentum*_variance + (1.0-_momentum)*variance;
        _v = _momentum*_v + (1.0-_momentum)*v;
        return _v;
    }

    operator const T&() const
    {
        return _v;
    }

    const T& variance() const
    {
        return std::sqrt(_variance);
    }

private:
    double      _momentum;
    T           _variance;
    T           _v;
};


template <typename T_Scalar, int Rows, int Cols>
class Momentum<Eigen::Matrix<T_Scalar, Cols, Rows>> {
private:
    using T = Eigen::Matrix<T_Scalar, Cols, Rows>;

public:
    Momentum(T_Scalar momentum, const T& v = T()) :
        _momentum   (momentum),
        _variance   (0.0*T()),
        _v          (v)
    {}

    const T& operator()(const T& v)
    {
        T variance = (v-_v).array().square().matrix();
        _variance = _momentum*_variance + (1.0-_momentum)*variance;
        _v = _momentum*_v + (1.0-_momentum)*v;
        return _v;
    }

    operator const T&() const
    {
        return _v;
    }

    T_Scalar variance() const
    {
        return _variance.cwiseSqrt().norm();
    }

private:
    T_Scalar    _momentum;
    T           _variance;
    T           _v;
};



#endif //IMAGE_DEMORPHING_MOMENTUM_HPP
