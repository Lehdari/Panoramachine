//
// Project: image_demorphing
// File: StitchImages.hpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#ifndef IMAGE_DEMORPHING_STITCHIMAGES_HPP
#define IMAGE_DEMORPHING_STITCHIMAGES_HPP


#include <vector>
#include <opencv2/core/mat.hpp>

#include "Image.hpp"


cv::Mat stitchImages(const std::vector<Image<Vec3f>>& images);


#endif //IMAGE_DEMORPHING_STITCHIMAGES_HPP
