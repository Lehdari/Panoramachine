//
// Project: panoramachine
// File: ImagePostProcessing.hpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#ifndef PANORAMACHINE_IMAGEPOSTPROCESSING_HPP
#define PANORAMACHINE_IMAGEPOSTPROCESSING_HPP


#include "MathTypes.hpp"
#include "Image.hpp"


void brightnessContrast(Image<Vec3f>& img, const Vec3f& brightness, const Vec3f& contrast);


#endif //PANORAMACHINE_IMAGEPOSTPROCESSING_HPP
