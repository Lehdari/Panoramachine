//
// Project: image_demorphing
// File: TrainFeatureDetector.hpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#ifndef IMAGE_DEMORPHING_TRAINFEATUREDETECTOR_HPP
#define IMAGE_DEMORPHING_TRAINFEATUREDETECTOR_HPP


#include <vector>
#include <random>

#include <DistortImage.hpp>
#include <Feature.hpp>


// Original image with multiple disroted ones, used for training entry generation
struct TrainingImage {
    cv::Mat                     original;
    std::vector<DistortedImage> distorted;

    TrainingImage(cv::Mat&& image,
        int nDistorted,
        const DistortSettings& minSettings,
        const DistortSettings& maxSettings);

private:
    static std::default_random_engine rnd;
};

using TrainingImages = std::vector<TrainingImage>;


struct TrainingEntry {
    Feature f1;
    Feature f2;
    float   similarity;
};

using TrainingData = std::vector<TrainingEntry>;


TrainingEntry makeTrainingEntry(const TrainingImages& trainingImages, float similarity=0.0f);

void trainFeatureDetector();


#endif //IMAGE_DEMORPHING_TRAINFEATUREDETECTOR_HPP
