//
// Project: image_demorphing
// File: Image.hpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#ifndef IMAGE_DEMORPHING_IMAGE_HPP
#define IMAGE_DEMORPHING_IMAGE_HPP


#include <opencv2/core/mat.hpp>

#include "MathTypes.hpp"


template <typename T>
class Image {
public:
    Image();
    Image(cv::Mat&& image);
    Image(const cv::Mat& image);

    // direct pixel access
    const T& operator()(int x, int y) const;

    // cubic interpolation
    T operator()(const Vec2f& p, float r=0.0f) const;

    T sampleCubic(const Vec2f& p, int layer=0) const;
    T sampleCubicXDeriv(const Vec2f& p, int layer=0) const;
    T sampleCubicYDeriv(const Vec2f& p, int layer=0) const;

    // layer access
    cv::Mat& operator[](int layer);

    // implicit conversion to cv::Mat returns the first layer
    operator cv::Mat() const;

    Image<T> clone() const;

    std::vector<cv::Mat>::iterator begin();
    std::vector<cv::Mat>::iterator end();

private:
    std::vector<cv::Mat>    _layers;

    static cv::Mat downscaleLayer(const cv::Mat& layer);
};


#include "Image.inl"


#endif //IMAGE_DEMORPHING_IMAGE_HPP
