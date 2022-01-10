//
// Project: panoramachine
// File: CorrectionAlgorithms.hpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#ifndef PANORAMACHINE_CORRECTIONALGORITHMS_HPP
#define PANORAMACHINE_CORRECTIONALGORITHMS_HPP


#include <opencv2/core/mat.hpp>


cv::Mat createCorrection2(const cv::Mat& source, const cv::Mat& target);
cv::Mat createCorrection4(const cv::Mat& source, const cv::Mat& target);


#endif //PANORAMACHINE_CORRECTIONALGORITHMS_HPP
