//
// Project: image_demorphing
// File: Image.inl
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "Utils.hpp"


template<typename T>
Image<T>::Image()
{
    _layers.emplace_back(cv::Mat());
}

template <typename T>
Image<T>::Image(cv::Mat&& image)
{
    _layers.clear();
    _layers.emplace_back(image.clone());

    while (_layers.back().rows > 1 || _layers.back().cols > 1) {
        _layers.emplace_back(downscaleLayer(_layers.back()));
    }
}

template <typename T>
Image<T>::Image(const cv::Mat& image) :
    Image(image.clone())
{
}

template<typename T>
const T& Image<T>::operator()(int x, int y) const
{
    return _layers[0].at<T>(y, x);
}

template<typename T>
T Image<T>::operator()(const Vec2f& p, float r) const
{
    // linear-interpolated mip cubic sampling
    float layerSmooth = std::log2(r);
    int layer = (int)layerSmooth;
    float layerInterpolate = layer >= 0 ? layerSmooth-layer : 0.0f;
    layer = std::max(layer, 0);
    return (1.0f-layerInterpolate)*sampleMatCubic<T>(_layers[layer], p/(1 << layer)) +
        layerInterpolate* sampleMatCubic<T>(_layers[layer+1], p/(2 << layer));
}

template<typename T>
cv::Mat& Image<T>::operator[](int layer)
{
    return _layers[layer];
}

template<typename T>
Image<T>::operator cv::Mat() const
{
    return _layers[0];
}

template<typename T>
Image<T> Image<T>::clone() const
{
    Image<T> imgClone;
    imgClone._layers.clear();

    for (auto& layer : _layers)
        imgClone._layers.template emplace_back(layer.clone());

    return imgClone;
}

template<typename T>
std::vector<cv::Mat>::iterator Image<T>::begin()
{
    return _layers.begin();
}

template<typename T>
std::vector<cv::Mat>::iterator Image<T>::end()
{
    return _layers.end();
}

template<typename T>
cv::Mat Image<T>::downscaleLayer(const cv::Mat& layer)
{
    cv::Mat downscaled(layer.rows/2+layer.rows%2, layer.cols/2+layer.cols%2, layer.type());

    for (int j=0; j<downscaled.rows; ++j) {
        auto* r = downscaled.ptr<T>(j);
        for (int i=0; i<downscaled.cols; ++i) {
            int ii = i*2;
            int jj = j*2;
            int nSamples = 1;

            r[i] = layer.at<T>(jj, ii);
            if (ii+1 < layer.cols) {
                r[i] += layer.at<T>(jj, ii+1);
                ++nSamples;
            }

            if (jj+1 < layer.rows) {
                r[i] += layer.at<T>(jj+1, ii);
                ++nSamples;
                if (ii+1 < layer.cols) {
                    r[i] += layer.at<T>(jj+1, ii+1);
                    ++nSamples;
                }
            }

            r[i] /= nSamples;
        }
    }

    return downscaled;
}

