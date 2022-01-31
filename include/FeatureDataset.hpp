//
// Project: panoramachine
// File: FeatureDataset.hpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#ifndef PANORAMACHINE_FEATUREDATASET_HPP
#define PANORAMACHINE_FEATUREDATASET_HPP


#include "Image.hpp"
#include "Feature.hpp"

#include <random>


class FeatureDataset {
public:
    using Label = Vec3f;

    struct Entry {
        Feature*    f1;
        Feature*    f2;
        Label*      label;
    };

    using Iterator = std::vector<Entry>::iterator;
    using ConstIterator = std::vector<Entry>::const_iterator;

    FeatureDataset(const std::vector<Image<Vec3f>>& images);

    void construct(size_t size);
    void generateNewEntries(const ConstIterator& begin, const ConstIterator& end, double replaceProbability);
    void shuffle(const Iterator& begin, const Iterator& end);

    Iterator begin();
    Iterator end();
    ConstIterator begin() const;
    ConstIterator end() const;

    void writeToDirectory(const std::string& directory);
    void readFromDirectory(const std::string& directory);

private:
    static std::default_random_engine   _rnd;

    const std::vector<Image<Vec3f>>     _images;
    std::vector<Feature>                _features1;
    std::vector<Feature>                _features2;
    std::vector<Label>                  _labels;
    std::vector<Entry>                  _entries;

    void createRandomPositiveEntry(const Entry& entry, double diff = 0.0);
    void createRandomNegativeEntry(const Entry& entry);

    struct DistortSettings {
        Vec2f   p;
        float   featureRadius;
        float   scale;
        float   rotation;
    };

    static Image<Vec3f> distortImage(
        const Image<Vec3f>& image,
        const DistortSettings& settings,
        Mat3f* t = nullptr);
};


#endif //PANORAMACHINE_FEATUREDATASET_HPP
