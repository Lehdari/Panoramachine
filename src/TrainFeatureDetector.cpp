//
// Project: image_demorphing
// File: TrainFeatureDetector.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "TrainFeatureDetector.hpp"
#include "Utils.hpp"
#include "FeatureDetector.hpp"
#include <opencv2/highgui.hpp>
#include <iomanip>


#define RND ((rnd()%1000001)*0.000001)
#define RANGE(MIN, MAX) (decltype(MIN))(MIN+RND*(MAX-MIN))


std::default_random_engine TrainingImage::rnd(15077155157);


TrainingImage::TrainingImage(cv::Mat&& image,
    int nDistorted,
    const DistortSettings& minSettings,
    const DistortSettings& maxSettings)
:
    original    (std::move(image))
{

    distorted.reserve(nDistorted);

    for (int i=0; i<nDistorted; ++i) {
        DistortSettings settings {
            RANGE(minSettings.nMinTransforms, maxSettings.nMinTransforms),
            RANGE(minSettings.nMaxTransforms, maxSettings.nMaxTransforms),
            RANGE(minSettings.minDistance, maxSettings.minDistance),
            RANGE(minSettings.maxDistance, maxSettings.maxDistance),
            RANGE(minSettings.maxRotation, maxSettings.maxRotation),
            RANGE(minSettings.minScale, maxSettings.minScale),
            RANGE(minSettings.maxScale, maxSettings.maxScale),
            RANGE(minSettings.maxTranslation, maxSettings.maxTranslation)
        };

        distorted.push_back(std::move(distortImage(original, settings)));
    }
}


inline __attribute__((always_inline)) float entrySimilarityDistance(float similarity) {
    return similarity > 0.9f ? (1.0f-similarity)*20.0f : 2.0f*std::pow(Feature::frm, (1.0f-similarity)*50.0f);
}

TrainingEntry makeTrainingEntry(const TrainingImages& trainingImages, float similarity)
{
    static std::default_random_engine rnd(1507715517);

    const cv::Mat* img1(nullptr);
    const cv::Mat* img2(nullptr);
    Vec2f p1, p2;

    if (similarity < 1.0e-8f) {
        // pick two samples from differing images
        int imgId1 = rnd()%trainingImages.size();
        int imgId2 = rnd()%trainingImages.size();
        while (imgId1 == imgId2)
            imgId2 = rnd()%trainingImages.size();

        img1 = &trainingImages[imgId1].original;
        img2 = &trainingImages[imgId2].original;
        p1 << RND*img1->cols, RND*img1->rows;
        p2 << RND*img2->cols, RND*img2->rows;
    }
    else {
        // pick sample from an original image and random respective distorted one
        int imgId1 = rnd()%trainingImages.size();
        img1 = &trainingImages[imgId1].original;
        int imgId2 = rnd()%trainingImages[imgId1].distorted.size();
        img2 = &trainingImages[imgId1].distorted[imgId2].distorted;
        p1 << RND*img1->cols, RND*img1->rows;
        p2 = p1 + trainingImages[imgId1].distorted[imgId2].forwardMap.at<Vec2f>((int)p1(1), (int)p1(0));

        float angle = 2.0f*M_PI*RND;
        Vec2f ddir(std::cos(angle), std::sin(angle));
        p2 += ddir*entrySimilarityDistance(similarity);
    }

    return { Feature(*img1, p1, 2.0f), Feature(*img2, p2, 2.0f), std::clamp(similarity*2.0f-1.0f, -0.999f, 0.999f) };
}

TrainingData generateDataset(int datasetSize)
{
    DistortSettings minSettings{
        10, 15,
        128.0, 256.0,
        M_PI*0.0625,
        0.6, 0.9,
        Vec2d(32.0, 32.0)
    };

    DistortSettings maxSettings{
        15, 25,
        256.0, 512.0,
        M_PI*0.125,
        0.85, 1.3,
        Vec2d(64.0, 64.0)
    };

    TrainingImages images;
    images.emplace_back(cv::imread(std::string(IMAGE_DEMORPHING_RES_DIR) + "mountains1.exr",
        cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH), 2, minSettings, maxSettings);
    images.emplace_back(cv::imread(std::string(IMAGE_DEMORPHING_RES_DIR) + "lenna.exr",
        cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH), 2, minSettings, maxSettings);

#if 0
    cv::imshow("original", images[0].original);
    cv::imshow("distorted1", images[0].distorted[0].distorted);
    cv::imshow("distorted2", images[0].distorted[1].distorted);
#endif

    std::default_random_engine rnd(1507715517);

    TrainingData trainingData;
    trainingData.reserve(datasetSize);
    for (int i=0; i<datasetSize; ++i) {
        float similarity = rnd()%2 == 0 ? 0.0f : 0.5f+RND*0.5f;
        trainingData.emplace_back(makeTrainingEntry(images, similarity));

#if 0
        visualizeFeature(trainingData.back().f1, "f1");
        visualizeFeature(trainingData.back().f2, "f2");
        cv::waitKey(0);
#endif
    }

    return trainingData;
}

void saveDataset(const TrainingData& data)
{
    for (int i=0; i<data.size(); ++i) {
        std::stringstream filename1, filename2, filenameSimilarity;
        filename1 << "temp/feature_" << std::setfill('0') << std::setw(4) << i << "_f1.bin";
        filename2 << "temp/feature_" << std::setfill('0') << std::setw(4) << i << "_f2.bin";
        filenameSimilarity << "temp/feature_" << std::setfill('0') << std::setw(4) << i << "_similarity.txt";

        writeMatrixBinary(filename1.str(), data[i].f1.polar);
        writeMatrixBinary(filename2.str(), data[i].f2.polar);

        std::ofstream similarityFile;
        similarityFile.open(filenameSimilarity.str());
        similarityFile << data[i].similarity << std::endl;
        similarityFile.close();
    }
}

TrainingData loadDataset(int datasetSize)
{
    TrainingData trainingData;
    trainingData.reserve(datasetSize);
    for (int i=0; i<datasetSize; ++i) {
        trainingData.emplace_back();
        std::stringstream filename1, filename2, filenameSimilarity;
        filename1 << "temp/feature_" << std::setfill('0') << std::setw(4) << i << "_f1.bin";
        filename2 << "temp/feature_" << std::setfill('0') << std::setw(4) << i << "_f2.bin";
        filenameSimilarity << "temp/feature_" << std::setfill('0') << std::setw(4) << i << "_similarity.txt";

        readMatrixBinary(filename1.str(), trainingData.back().f1.polar);
        readMatrixBinary(filename2.str(), trainingData.back().f2.polar);

        std::ifstream similarityFile;
        similarityFile.open(filenameSimilarity.str());
        similarityFile >> trainingData.back().similarity;
        similarityFile.close();
    }
    return trainingData;
}

TrainingBatch sampleTrainingBatch(const TrainingData& data, int trainingDatasetSize, int batchSize)
{
    TrainingBatch batch;
    batch.reserve(batchSize);

    static size_t p = 0;
    for (int i=0; i<batchSize; ++i) {
        batch.push_back(&data[p]);

        if (++p >= trainingDatasetSize)
            p=0;
    }

    return batch;
}

void trainFeatureDetector()
{
    constexpr int datasetSize = 1024;
    constexpr int trainingDatasetSize = (datasetSize/8)*7;
    constexpr int evaluationDatasetSize = datasetSize/8;
    constexpr int batchSize = 32;
    constexpr int batchesInEpoch = trainingDatasetSize / batchSize;
    constexpr int nEpochs = 100000;

#if 0
    auto trainingData = generateDataset(datasetSize);
    saveDataset(trainingData);
#else
    auto trainingData = loadDataset(datasetSize);
#endif

    FeatureDetector detector;

    for (int e=0; e<nEpochs; ++e) {
        auto t1 = std::chrono::high_resolution_clock::now();

        double trainingLoss = 0.0;
        for (int i=0; i<batchesInEpoch; ++i) {
            auto batch = sampleTrainingBatch(trainingData, trainingDatasetSize, batchSize);
            trainingLoss += detector.trainBatch(batch);
        }

        auto t2 = std::chrono::high_resolution_clock::now();

        double evaluationLoss = 0.0;
        for (int i=trainingDatasetSize; i<datasetSize; ++i) {
            double l = detector(trainingData[i].f1, trainingData[i].f2)-trainingData[i].similarity;
            l *= l;
            evaluationLoss += l;
        }

        printf("Epoch %d finished, trainingLoss: %13.10f, evaluationLoss: %13.10f, time: %ld\n",
            e, trainingLoss/batchesInEpoch, evaluationLoss/evaluationDatasetSize,
            std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
    }
}
