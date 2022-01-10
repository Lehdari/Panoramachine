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
#include "Utils.hpp"

#include <omp.h>


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


template <template <typename> class T_Optimizer, typename T_Weights>
struct Optimizer {
    template <typename T_Scalar>
    Optimizer(T_Scalar initAmplitude = 1.0, double dropoutRate = 0.0) :
        w           (T_Weights::Random() * initAmplitude),
        dropoutRate (dropoutRate)
    {}

    void saveWeights(const std::string& filename)
    {
        writeMatrixBinary(filename, w);
    }

    void loadWeights(const std::string& filename)
    {
        readMatrixBinary(filename, w);
    }

    double      dropoutRate;
    T_Weights   w;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <typename T_Weights>
struct OptimizerAdam : public Optimizer<OptimizerAdam, T_Weights>
{
    template <typename T_Scalar>
    OptimizerAdam(T_Scalar initAmplitude = 1.0, double dropoutRate = 0.0) :
        Optimizer<OptimizerAdam, T_Weights>(initAmplitude, dropoutRate),
        wgt (omp_get_max_threads(), T_Weights::Zero()),
        wg  (T_Weights::Zero()),
        wm  (T_Weights::Zero()),
        wv  (T_Weights::Zero()),
        t   (0)
    {
        this->w.template block<T_Weights::RowsAtCompileTime,1>(0,T_Weights::ColsAtCompileTime-1) =
            Eigen::Matrix<T_Scalar,T_Weights::RowsAtCompileTime,1>::Zero();
    }

    inline void addGradientParallel(const T_Weights& g)
    {
        wgt[omp_get_thread_num()] += g;
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

        for (auto& wgtt : wgt)
            wg += wgtt;

        wm.noalias() = momentum*wm + (1.0-momentum)*wg; // update first moment
        wv.noalias() = momentum2*wv + (1.0-momentum2)*(wg.array().square().matrix()); // update second moment

        T_Scalar alpha = learningRate *
            (std::sqrt(1.0 - std::pow(momentum2, (T_Scalar)t)) /
            (1.0 - std::pow(momentum, (T_Scalar)t)));

        this->w -= alpha * wm.cwiseProduct((wv.cwiseSqrt()+T_Weights::Ones()*epsilon).cwiseInverse());

        if (weightDecay >= epsilon) // weight decay
            this->w.noalias() = this->w*(1.0-weightDecay);

        for (auto& wgtt : wgt)
            wgtt = T_Weights::Zero();
        wg = T_Weights::Zero();
    }

    std::vector<T_Weights, Eigen::aligned_allocator<T_Weights>> wgt;
    T_Weights   wg; // gradient
    T_Weights   wm; // first moment
    T_Weights   wv; // second moment
    int         t; // timestep index

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <typename T_Weights>
struct OptimizerStatic : public Optimizer<OptimizerStatic, T_Weights>
{
    template <typename T_Scalar>
    OptimizerStatic(T_Scalar initAmplitude = 1.0, double dropoutRate = 0.0) :
        Optimizer<OptimizerStatic, T_Weights>(initAmplitude)
    {
        this->w.template block<T_Weights::RowsAtCompileTime,1>(0,T_Weights::ColsAtCompileTime-1) =
            Eigen::Matrix<T_Scalar,T_Weights::RowsAtCompileTime,1>::Zero();
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


template <typename T_Matrix>
T_Matrix& random(float amplitude)
{
    static std::vector<std::default_random_engine> rnd = [](){
        std::vector<std::default_random_engine> rnd;
        rnd.reserve(omp_get_max_threads());
        for (int i=0; i<omp_get_max_threads(); ++i)
            rnd.template emplace_back(1507+715517*i);
        return rnd;
    }();
    static std::vector<T_Matrix, Eigen::aligned_allocator<T_Matrix>> m(omp_get_max_threads(), T_Matrix::Zero());
    for (int j=0; j<T_Matrix::ColsAtCompileTime; ++j) {
        for (int i=0; i<T_Matrix::RowsAtCompileTime; ++i) {
            m[omp_get_thread_num()](i, j) = ((int)rnd[omp_get_thread_num()]()%2000001-1000000)*0.000001f*amplitude;
        }
    }
    return m[omp_get_thread_num()];
}

template <typename T_Matrix>
T_Matrix& dropoutMask(float zeroRate)
{
    static std::vector<T_Matrix, Eigen::aligned_allocator<T_Matrix>> m(omp_get_max_threads(), T_Matrix::Zero());
    m[omp_get_thread_num()] = (random<T_Matrix>(0.5f)+T_Matrix::Ones()*(1.0f-zeroRate)).array().round().matrix();
    return m[omp_get_thread_num()];
}


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
        T_Activation&& activation = T_Activation(),
        double dropoutRate = 0.0) :
        _activation (activation),
        _optimizer  (std::make_shared<T_Optimizer<Weights>>(initAmplitude, dropoutRate)),
        _input      (omp_get_max_threads(), InputExtended::Ones()),
        _weighed    (omp_get_max_threads(), Output::Ones())
    {
    }

    LayerDense(
        T_Activation&& activation,
        const std::shared_ptr<T_Optimizer<Weights>>& optimizer) :
        _activation (activation),
        _optimizer  (optimizer),
        _input      (omp_get_max_threads(), InputExtended::Ones()),
        _weighed    (omp_get_max_threads(), Output::Ones())
    {
    }

    inline Output operator()(const Input& x)
    {
        _input[omp_get_thread_num()].template block<T_InputRows,1>(0,0) = x*(1.0f-(float)_optimizer->dropoutRate);
        _weighed[omp_get_thread_num()] = _optimizer->w * _input[omp_get_thread_num()];
        return _activation.activation(_weighed[omp_get_thread_num()]);
    }

    inline Output trainingForward(const Input& x)
    {
        _input[omp_get_thread_num()].template block<T_InputRows,1>(0,0) = x.
            template cwiseProduct(dropoutMask<Input>((float)_optimizer->dropoutRate));
        _weighed[omp_get_thread_num()] = _optimizer->w * _input[omp_get_thread_num()];
        return _activation.activation(_weighed[omp_get_thread_num()]);
    }

    inline Input backpropagate(const Output& g)
    {
        Output ag = _activation.gradient(_weighed[omp_get_thread_num()]).cwiseProduct(g);
        _optimizer->addGradientParallel(ag * _input[omp_get_thread_num()].transpose());
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
    std::vector<InputExtended, Eigen::aligned_allocator<InputExtended>> _input; // last input
    std::vector<Output, Eigen::aligned_allocator<Output>>               _weighed; // weights * input
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
        T_Activation&& activation = T_Activation(),
        double dropoutRate = 0.0) :
        _activation (activation),
        _optimizer  (std::make_shared<T_Optimizer<Weights>>(initAmplitude, dropoutRate)),
        _input      (omp_get_max_threads(), InputModified::Ones()),
        _weighed    (omp_get_max_threads(), Output::Ones())
    {
    }

    LayerMerge(
        T_Activation&& activation,
        const std::shared_ptr<T_Optimizer<Weights>>& optimizer) :
        _activation (activation),
        _optimizer  (optimizer),
        _input      (omp_get_max_threads(), InputModified::Ones()),
        _weighed    (omp_get_max_threads(), Output::Ones())
    {
    }

    inline Output operator()(const Input& x)
    {
        Input xx = x*(1.0f-(float)_optimizer->dropoutRate);
        _input[omp_get_thread_num()].template block<2*T_InputRows,T_InputCols/2>(0,0) =
            Eigen::Map<const Eigen::Matrix<T_Scalar, 2*T_InputRows, T_InputCols/2>>(xx.data());
        _weighed[omp_get_thread_num()] = _optimizer->w * _input[omp_get_thread_num()];
        return _activation.activation(_weighed[omp_get_thread_num()]);
    }

    inline Output trainingForward(const Input& x)
    {
        Input xx = x.template cwiseProduct(dropoutMask<Input>((float)_optimizer->dropoutRate));
        _input[omp_get_thread_num()].template block<2*T_InputRows,T_InputCols/2>(0,0) =
            Eigen::Map<const Eigen::Matrix<T_Scalar, 2*T_InputRows, T_InputCols/2>>(xx.data());
        _weighed[omp_get_thread_num()] = _optimizer->w * _input[omp_get_thread_num()];
        return _activation.activation(_weighed[omp_get_thread_num()]);
    }

    inline Input backpropagate(const Output& g)
    {
        Output ag = _activation.gradient(_weighed[omp_get_thread_num()]).cwiseProduct(g);
        _optimizer->addGradientParallel(ag * _input[omp_get_thread_num()].transpose());
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
    std::vector<InputModified, Eigen::aligned_allocator<InputModified>> _input; // last input
    std::vector<Output, Eigen::aligned_allocator<Output>>               _weighed; // weights * input
};

template <typename T_Scalar, typename T_Activation, template <typename> class T_Optimizer,
    int T_InputRows, int T_InputCols, int T_OutputRows>
class LayerConv
{
public:
    using Input = Eigen::Matrix<T_Scalar, T_InputRows, T_InputCols>;
    using Output = Eigen::Matrix<T_Scalar, T_OutputRows, T_InputCols>;
    using InputExtended = Eigen::Matrix<T_Scalar, T_InputRows+1, T_InputCols>;
    using Weights = Eigen::Matrix<T_Scalar, T_OutputRows, T_InputRows+1>;

    LayerConv(T_Scalar initAmplitude = 1.0,
        T_Activation&& activation = T_Activation(),
        double dropoutRate = 0.0) :
        _activation (activation),
        _optimizer  (std::make_shared<T_Optimizer<Weights>>(initAmplitude, dropoutRate)),
        _input      (omp_get_max_threads(), InputExtended::Ones()),
        _weighed    (omp_get_max_threads(), Output::Ones())
    {
    }

    LayerConv(
        T_Activation&& activation,
        const std::shared_ptr<T_Optimizer<Weights>>& optimizer) :
        _activation (activation),
        _optimizer  (optimizer),
        _input      (omp_get_max_threads(), InputExtended::Ones()),
        _weighed    (omp_get_max_threads(), Output::Ones())
    {
    }

    inline Output operator()(const Input& x)
    {
        _input[omp_get_thread_num()].template block<T_InputRows,T_InputCols>(0,0) =
            x*(1.0f-(float)_optimizer->dropoutRate);
        _weighed[omp_get_thread_num()] = _optimizer->w * _input[omp_get_thread_num()];
        return _activation.activation(_weighed[omp_get_thread_num()]);
    }

    inline Output trainingForward(const Input& x)
    {
        _input[omp_get_thread_num()].template block<T_InputRows,T_InputCols>(0,0) = x.
            template cwiseProduct(dropoutMask<Input>((float)_optimizer->dropoutRate));
        _weighed[omp_get_thread_num()] = _optimizer->w * _input[omp_get_thread_num()];
        return _activation.activation(_weighed[omp_get_thread_num()]);
    }

    inline Input backpropagate(const Output& g)
    {
        Output ag = _activation.gradient(_weighed[omp_get_thread_num()]).cwiseProduct(g);
        _optimizer->addGradientParallel(ag * _input[omp_get_thread_num()].transpose());
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
    std::vector<InputExtended, Eigen::aligned_allocator<InputExtended>> _input; // last input
    std::vector<Output, Eigen::aligned_allocator<Output>>               _weighed; // weights * input
};


#endif //IMAGE_DEMORPHING_NEURALNETWORK_HPP
