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

#include "DistortImage.hpp"
#include "Feature.hpp"
#include "Image.hpp"


// Original image with multiple disroted ones, used for training entry generation
struct TrainingImage {
    Image<Vec3f>                original;
    std::vector<DistortedImage> distorted;

    void create(Image<Vec3f>&& image,
        int nDistorted,
        const DistortSettings& minSettings,
        const DistortSettings& maxSettings);

    void write(const std::string& stem) const;
    bool read(const std::string& stem, int nDistorted);

private:
    static std::default_random_engine rnd;
};

using TrainingImages = std::vector<TrainingImage>;


struct TrainingEntry {
    Feature f1;
    Feature f2;
    Vec3f   label;  // pos diff in x/y (divided by feature max radius * f1 scale), scale diff (f2 scale / f1 scale)

    void writeToFile(const std::string& filename) const;
    void readFromFile(const std::string& filename);
};

using TrainingData = std::vector<TrainingEntry>;
using TrainingBatch = std::vector<const TrainingEntry*>;


TrainingEntry makeTrainingEntry(const TrainingImages& trainingImages, float similarity=0.0f);

void trainFeatureDetector();


#endif //IMAGE_DEMORPHING_TRAINFEATUREDETECTOR_HPP
