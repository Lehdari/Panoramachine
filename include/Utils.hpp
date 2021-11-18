//
// Project: image_demorphing
// File: Utils.hpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#ifndef IMAGE_DEMORPHING_UTILS_HPP
#define IMAGE_DEMORPHING_UTILS_HPP


#include "MathTypes.hpp"
#include <opencv2/core/mat.hpp>
#include <fstream>


Vec3f sampleMatCubic(const cv::Mat& m, const Vec2f& p);
void gammaCorrect(cv::Mat& image, float gamma);
cv::Mat correctImage(const cv::Mat& image, const cv::Mat& correction);
void show2ChannelImage(const std::string& windowName, const cv::Mat& image);


#include "Utils.inl"


#endif //IMAGE_DEMORPHING_UTILS_HPP
