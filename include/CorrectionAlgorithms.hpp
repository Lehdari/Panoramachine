//
// Project: image_demorphing
// File: CorrectionAlgorithms.hpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#ifndef IMAGE_DEMORPHING_CORRECTIONALGORITHMS_HPP
#define IMAGE_DEMORPHING_CORRECTIONALGORITHMS_HPP


#include <opencv2/core/mat.hpp>


cv::Mat createCorrection1(const cv::Mat& source, const cv::Mat& target);


#endif //IMAGE_DEMORPHING_CORRECTIONALGORITHMS_HPP
