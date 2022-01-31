//
// Project: panoramachine
// File: TrainFeatureDetector.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "TrainFeatureDetector.hpp"
#include "FeatureDataset.hpp"
#include "Utils.hpp"
#include "FeatureDetector.hpp"
#include <iomanip>
#include <filesystem>
#include <opencv2/imgproc.hpp>


#define RND ((rnd()%1000000)*0.000001)
#define RANGE(MIN, MAX) (decltype(MIN))(MIN+RND*(MAX-MIN))


std::vector<Image<Vec3f>> readImages(const std::string& datasetImagesDirectory)
{
    std::vector<Image<Vec3f>> images;
    for (const auto & entry : std::filesystem::directory_iterator(datasetImagesDirectory)) {
        printf("Processing %s...\n", entry.path().string().c_str());
        std::string imagePath = datasetImagesDirectory + std::string("/") + entry.path().string();
        images.push_back(readImage<Vec3f>(imagePath));
    }
    return images;
}

void trainFeatureDetector()
{
    constexpr size_t datasetSize = 8192;
    constexpr size_t batchSize = 256;
    constexpr int nNewEntries = 32; // number of new entries per epoch
    constexpr int nTrainingBatches = 16;
    constexpr int nEvaluationBatches = 16;
    constexpr int batchesInEpoch = nTrainingBatches + nEvaluationBatches;
    constexpr int nEpochs = 100000;

    std::default_random_engine rnd(1507715517);

    //TrainingData trainingData;
    const std::string trainingSourceImagesDirectory = "../training_data";
    const std::string datasetImagesDirectory = "../temp/images";
    const std::string datasetDirectory = "../temp/dataset";

    auto images = readImages(trainingSourceImagesDirectory);
    FeatureDataset dataset(images);
    if (!std::filesystem::exists("../temp/dataset")) {
        // generate dataset in case it does not yet exist
        printf("Generaring new dataset...\n");
        dataset.construct(datasetSize);
        dataset.writeToDirectory(datasetDirectory);
    }
    else {
        printf("Loading dataset...\n");
        dataset.readFromDirectory(datasetDirectory);
    }

    FeatureDetector<OptimizerAdam> detector(0.25);
    detector.loadWeights("../feature_detector_model");
    double minEvaluationLoss = std::numeric_limits<double>::max();

    FeatureDataset::ConstIterator newEntriesBegin = dataset.begin();
    FeatureDataset::ConstIterator newEntriesEnd = newEntriesBegin + nNewEntries;
    for (int e=0; e<nEpochs; ++e) {
        FeatureDataset::ConstIterator batchBegin = dataset.begin();
        FeatureDataset::ConstIterator batchEnd = batchBegin + batchSize;

        auto t1 = std::chrono::high_resolution_clock::now();
        double trainingLoss = 0.0;
        for (int i=0; i<nTrainingBatches; ++i) {
            trainingLoss += detector.trainBatch(batchBegin, batchEnd);

            batchBegin = batchEnd;
            batchEnd = batchBegin + batchSize;
            assert(batchEnd <= dataset.end());
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        trainingLoss /= nTrainingBatches;

        double evaluationLoss = 0.0;
        for (int i=0; i<nEvaluationBatches; ++i) {
            for (int j = 0; j < batchSize; ++j) {
                auto& entry = *(batchBegin+j);
                Vec3f g = FeatureDetector<OptimizerAdam>::gradient(
                    detector(*entry.f1, *entry.f2), *entry.label);
                evaluationLoss += g.array().square().sum();
            }

            batchBegin = batchEnd;
            batchEnd = batchBegin + batchSize;
            assert(batchEnd <= dataset.end());
        }

        evaluationLoss /= batchSize*nEvaluationBatches;

        printf("Epoch %d finished, trainingLoss: %13.10f, evaluationLoss: %13.10f, best: %13.10f time: %ld\n",
            e, trainingLoss, evaluationLoss, minEvaluationLoss,
            std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());

        detector.printInfo();

        if (evaluationLoss < minEvaluationLoss) {
            detector.saveWeights("../feature_detector_model");
            printf("record evaluationLoss, saving\n");
            minEvaluationLoss = evaluationLoss;
        }

        // Generate new entries and shuffle the dataset
        dataset.generateNewEntries(newEntriesBegin, newEntriesEnd, 1.0);
        newEntriesBegin = newEntriesEnd; // advance the range
        newEntriesEnd = newEntriesBegin + nNewEntries;
        if (newEntriesEnd > dataset.end()) {
            newEntriesBegin = dataset.begin();
            newEntriesEnd = newEntriesBegin + nNewEntries;
        }
        dataset.shuffle(dataset.begin(), dataset.begin()+(batchSize*nTrainingBatches));
    }
}
