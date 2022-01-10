//
// Project: panoramachine
// File: Utils.hpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#ifndef PANORAMACHINE_UTILS_HPP
#define PANORAMACHINE_UTILS_HPP


#include "MathTypes.hpp"
#include "Image.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <fstream>


Vec3f sampleMatCubic(const cv::Mat& m, const Vec2f& p);
void gammaCorrect(cv::Mat& image, float gamma);
cv::Mat correctImage(const cv::Mat& image, const cv::Mat& correction);
void show2ChannelImage(const std::string& windowName, const cv::Mat& image);
cv::Mat load2ChannelImage(const std::string& filename);

// Map from vector/scalar types to OpenCV pixel formats
struct PixelFormatMap { template <typename U> static constexpr int format = 0; };
template <> constexpr int PixelFormatMap::format<float> = CV_32FC1;
template <> constexpr int PixelFormatMap::format<Vec2f> = CV_32FC2;
template <> constexpr int PixelFormatMap::format<Vec3f> = CV_32FC3;


#include "Utils.inl"


#endif //PANORAMACHINE_UTILS_HPP
