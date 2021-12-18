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
#include "ImagePostProcessing.hpp"
#include <opencv2/highgui.hpp>
#include <iomanip>
#include <filesystem>


#define RND ((rnd()%1000001)*0.000001)
#define RANGE(MIN, MAX) (decltype(MIN))(MIN+RND*(MAX-MIN))


std::default_random_engine TrainingImage::rnd(15077155157);


TrainingImage::TrainingImage(Image<Vec3f>&& image,
    int nDistorted,
    const DistortSettings& minSettings,
    const DistortSettings& maxSettings)
:
    original    (std::move(image))
{
    distorted.reserve(nDistorted);

    // apply more transforms for images with more area
    double nTransformsMultiplier = (original[0].rows*original[0].cols) / (1024.0*1024.0);

    for (int i=0; i<nDistorted; ++i) {
        printf("Creating distorted image %d/%d...\n", i+1, nDistorted);
        DistortSettings settings {
            RANGE((int)(minSettings.nMinTransforms*nTransformsMultiplier),
                (int)(maxSettings.nMinTransforms*nTransformsMultiplier)),
            RANGE((int)(minSettings.nMaxTransforms*nTransformsMultiplier),
                (int)(maxSettings.nMaxTransforms*nTransformsMultiplier)),
            RANGE(minSettings.minDistance, maxSettings.minDistance),
            RANGE(minSettings.maxDistance, maxSettings.maxDistance),
            RANGE(minSettings.maxRotation, maxSettings.maxRotation),
            RANGE(minSettings.minScale, maxSettings.minScale),
            RANGE(minSettings.maxScale, maxSettings.maxScale),
            RANGE(minSettings.maxTranslation, maxSettings.maxTranslation)
        };

        distorted.push_back(std::move(distortImage(original, settings)));

        float bbase = RND*1.4f-0.7f;
        float cbase = RND*1.4f-0.7f;
        brightnessContrast(distorted.back().distorted,
            Vec3f(bbase+RND*0.1f-0.05f, bbase+RND*0.1f-0.05f, bbase+RND*0.1f-0.05f),
            Vec3f(cbase+RND*0.1f-0.05f, cbase+RND*0.1f-0.05f, cbase+RND*0.1f-0.05f));
    }
}

void TrainingImage::write(const std::string& stem) const
{
    cv::imwrite(stem + std::string("_original.exr"), static_cast<cv::Mat>(original));

    for (int i=0; i<distorted.size(); ++i) {
        std::stringstream distortedFilename, backwardMapFilename, forwardMapFilename;
        distortedFilename << stem << "_distorted" << std::setfill('0') << std::setw(4) << i << ".exr";
        backwardMapFilename << stem << "_backwardmap" << std::setfill('0') << std::setw(4) << i << ".exr";
        forwardMapFilename << stem << "_forwardmap" << std::setfill('0') << std::setw(4) << i << ".exr";

        cv::Mat backwardMapOut(distorted[i].backwardMap.rows, distorted[i].backwardMap.cols, CV_32FC3);
        cv::Mat forwardMapOut(distorted[i].forwardMap.rows, distorted[i].forwardMap.cols, CV_32FC3);
        int mix[] = {0,0, 1,1, 1,2};
        cv::mixChannels(&(distorted[i].backwardMap), 1, &backwardMapOut, 1, mix, 3);
        cv::mixChannels(&(distorted[i].forwardMap), 1, &forwardMapOut, 1, mix, 3);
        cv::imwrite(distortedFilename.str(), static_cast<cv::Mat>(distorted[i].distorted));
        cv::imwrite(backwardMapFilename.str(), backwardMapOut);
        cv::imwrite(forwardMapFilename.str(), forwardMapOut);
    }
}

bool TrainingImage::read(const std::string& stem, int nDistorted)
{
    if (!std::filesystem::exists(stem + std::string("_original.exr")))
        return false;

    original = Image<Vec3f>(cv::imread(stem + std::string("_original.exr"),
        cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH));

    distorted.clear();
    distorted.reserve(nDistorted);
    for (int i=0; i<nDistorted; ++i) {
        std::stringstream distortedFilename, backwardMapFilename, forwardMapFilename;
        distortedFilename << stem << "_distorted" << std::setfill('0') << std::setw(4) << i << ".exr";
        backwardMapFilename << stem << "_backwardmap" << std::setfill('0') << std::setw(4) << i << ".exr";
        forwardMapFilename << stem << "_forwardmap" << std::setfill('0') << std::setw(4) << i << ".exr";

        if (!std::filesystem::exists(distortedFilename.str()))
            return false;

        DistortedImage distortedImage {
            Image<Vec3f>(cv::imread(distortedFilename.str(), cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH)),
            load2ChannelImage(backwardMapFilename.str()),
            load2ChannelImage(forwardMapFilename.str())
        };
        distorted.emplace_back(std::move(distortedImage));
    }

    return true;
}

void TrainingEntry::writeToFile(const std::string& filename) const
{
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    out.write((char*) (&diff), sizeof(decltype(diff)));
    f1.writeToFile(out);
    f2.writeToFile(out);
    out.close();
}

void TrainingEntry::readFromFile(const std::string& filename)
{
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    in.read((char*) (&diff), sizeof(decltype(diff)));
    f1.readFromFile(in);
    f2.readFromFile(in);
    in.close();
}

Feature sampleMaxEnergy(
    const Image<Vec3f>& img,
    int nSamples,
    Vec2f& p,
    float firstRadius,
    std::default_random_engine& rnd)
{
    Feature f;
    for (int i=0; i<nSamples; ++i)
    {
        Vec2f p2(RND*((cv::Mat)(img)).cols, RND*((cv::Mat)(img)).rows);
        Feature f2(img, p2, firstRadius, RND*M_PI*2.0f);
        if (f2.energy > f.energy) {
            std::swap(f, f2);
            p = p2;
        }
    }
    return f;
}

TrainingEntry makeMatchingTrainingEntry(
    const TrainingImage& trainingImage,
    float diff,
    std::default_random_engine& rnd)
{
    const Image<Vec3f>* img1(nullptr);
    const Image<Vec3f>* img2(nullptr);
    Feature f1, f2;
    Vec2f p1, p2;

    // pick samples from same location (original or distorted)
    int imgSubId1 = rnd()%(trainingImage.distorted.size()+1); // 0 for original image
    int imgSubId2 = rnd()%(trainingImage.distorted.size());
    if (imgSubId1 == 0)
        img1 = &trainingImage.original;
    else
        img1 = &trainingImage.distorted[imgSubId1-1].distorted;
    img2 = &trainingImage.distorted[imgSubId2].distorted;

    // first (inner) radii for the features
    float rBase = 4.0*std::pow(Feature::frm, 16.0*RND-8.0); // base radius so the two radii are not ridiculously far from each other
    float r1 = rBase*std::pow(Feature::frm, 16.0*RND-8.0);
    float r2 = rBase*std::pow(Feature::frm, 16.0*RND-8.0);
    // feature separation distance
    float distance = diff*2.0f*(std::min(r1, r2)*Feature::fmr);

    f1 = sampleMaxEnergy(*img1, 10, p1, r1, rnd);
    if (imgSubId1 > 0) {// backward map first if first image is distorted
        p1 += trainingImage.distorted[imgSubId2].backwardMap.at<Vec2f>((int) p1(1), (int) p1(0));
        p1(0) = std::clamp(p1(0), 0.0f, (float)(((cv::Mat)(*img1)).cols-1));
        p1(1) = std::clamp(p1(1), 0.0f, (float)(((cv::Mat)(*img1)).rows-1));
    }
    p2 = p1 + trainingImage.distorted[imgSubId2].forwardMap.at<Vec2f>((int)p1(1), (int)p1(0));

    float dir = 2.0f*M_PI*RND;
    Vec2f ddir(std::cos(dir), std::sin(dir));
    p2 += ddir * distance;
    p2(0) = std::clamp(p2(0), 0.0f, (float)(((cv::Mat)(*img1)).cols-1));
    p2(1) = std::clamp(p2(1), 0.0f, (float)(((cv::Mat)(*img1)).rows-1));
    f2 = Feature(*img2, p2, r2, RND*M_PI*2.0f);

    return { std::move(f1), std::move(f2), diff };
}

void generateDataset(TrainingData& trainingData, int datasetSize)
{
    DistortSettings minSettings{
        10, 12,
        128.0, 256.0,
        M_PI*0.03125,
        0.7, 1.1,
        Vec2d(16.0, 16.0)
    };

    DistortSettings maxSettings{
        12, 24,
        512.0, 1024.0,
        M_PI*0.0625,
        0.85, 1.3,
        Vec2d(32.0, 32.0)
    };

    trainingData.clear();
    trainingData.resize(datasetSize);

    const char* trainingDataDirectory = "../training_data/";
    constexpr int nDistorted = 7;

    // read number of images
    int nImages = 0;
    for (const auto & entry : std::filesystem::directory_iterator(trainingDataDirectory))
        ++nImages;
    int featuresPerImage = datasetSize/nImages;

    // create matching training entries
    int datasetOffset = 0;
    for (const auto & entry : std::filesystem::directory_iterator(trainingDataDirectory)) {
        printf("Processing %s...\n", entry.path().string().c_str());
        TrainingImage trainingImage(readImage<Vec3f>(entry.path()), nDistorted, minSettings, maxSettings);

#if 0
        cv::imshow("original", trainingImage.original[0]);
        for (int i=0; i<nDistorted; ++i) {
            std::stringstream windowName;
            windowName << "distorted" << i;
            cv::imshow(windowName.str(), trainingImage.distorted[i].distorted[0]);
        }
        cv::waitKey(0);
#endif

        #pragma omp parallel for
        for (int i=0; i<featuresPerImage; ++i) {
            std::default_random_engine rnd(1507+715517*i);
            float diff = RND*0.5f;
            trainingData[datasetOffset+i] = makeMatchingTrainingEntry(trainingImage, diff, rnd);

#if 0
            visualizeFeature(trainingData[i].f1, "f1");
            visualizeFeature(trainingData[i].f2, "f2");
            printf("diff: %0.5f r1: %0.5f r2: %0.5f, dis: %0.5f\n",
                trainingData[i].diff,
                trainingData[i].f1.firstRadius*Feature::fmr,
                trainingData[i].f2.firstRadius*Feature::fmr,
                (trainingData[i].f1.p-trainingData[i].f2.p).norm());
            cv::waitKey(0);
#endif
        }

        datasetOffset += featuresPerImage;
    }
    // shuffle half of the entries to create mismatching entries
    std::default_random_engine rnd(1507715517);
    std::shuffle(trainingData.begin(), trainingData.end(), rnd);
    int datasetHalfSize = trainingData.size()/2;
    for (int i=0; i<datasetHalfSize; ++i) {
        int swapId = rnd()%datasetHalfSize;
        while (swapId == i)
            swapId = rnd()%datasetHalfSize;
        std::swap(trainingData[i].f2, trainingData[swapId].f2);
        trainingData[i].diff = 1.0f;
    }
    std::shuffle(trainingData.begin(), trainingData.end(), rnd);
#if 0
    for (int i=0; i<datasetSize; ++i) {
        visualizeFeature(trainingData[i].f1, "f1");
        visualizeFeature(trainingData[i].f2, "f2");
        printf("diff: %0.5f r1: %0.5f r2: %0.5f, dis: %0.5f\n",
            trainingData[i].diff,
            trainingData[i].f1.firstRadius*Feature::fmr,
            trainingData[i].f2.firstRadius*Feature::fmr,
            (trainingData[i].f1.p-trainingData[i].f2.p).norm());
        cv::waitKey(0);
    }
#endif
}

void saveDataset(const std::string& directory, const TrainingData& data)
{
    for (int i=0; i<data.size(); ++i) {
        std::stringstream filename;
        filename << directory << "/entry_" << std::setfill('0') << std::setw(5) << i << ".bin";
        data[i].writeToFile(filename.str());
    }
}

void loadDataset(const std::string& directory, TrainingData& trainingData, int datasetSize)
{
    trainingData.clear();
    trainingData.reserve(datasetSize);
    for (int i=0; i<datasetSize; ++i) {
        trainingData.emplace_back();
        std::stringstream filename;
        filename << directory << "/entry_" << std::setfill('0') << std::setw(5) << i << ".bin";
        trainingData.back().readFromFile(filename.str());
    }
}

TrainingBatch sampleBatch(const TrainingData& data, int datasetSize, int batchSize)
{
    TrainingBatch batch;
    batch.reserve(batchSize);

    static size_t p = 0;
    for (int i=0; i<batchSize; ++i) {
        batch.push_back(&data[p]);

        if (++p >= datasetSize)
            p=0;
    }

    return batch;
}

void trainFeatureDetector()
{
    constexpr int datasetSize = 32768;
    constexpr int batchSize = 256;
    constexpr int nTrainingBatches = 112;
    constexpr int nEvaluationBatches = 16;
    constexpr int batchesInEpoch = nTrainingBatches + nEvaluationBatches;
    constexpr int nEpochs = 100000;


    TrainingData trainingData;
    const std::string datasetDirectory = "../temp/dataset";
    if (!std::filesystem::exists("../temp/dataset")) {
        // generate dataset in case it does not yet exist
        printf("Generaring new dataset...\n");
        generateDataset(trainingData, datasetSize);
        printf("Saving dataset into %s...\n", datasetDirectory.c_str());
        std::filesystem::create_directories(datasetDirectory);
        saveDataset(datasetDirectory, trainingData);
    }
    else {
        loadDataset(datasetDirectory, trainingData, datasetSize);
    }

#if 0
    for (auto& entry : trainingData) {
        visualizeFeature(entry.f1, "f1", 4);
        visualizeFeature(entry.f2, "f2", 4);
        printf("diff: %0.5f r1: %0.5f r2: %0.5f, dis: %0.5f\n",
            entry.diff,
            entry.f1.firstRadius*Feature::fmr,
            entry.f2.firstRadius*Feature::fmr,
            (entry.f1.p-entry.f2.p).norm());
        cv::waitKey(0);
    }
#endif

    FeatureDetector<OptimizerAdam> detector;
    double minEvaluationLoss = std::numeric_limits<double>::max();
    int epochStartId = 0;
    for (int e=0; e<nEpochs; ++e) {
        Eigen::internal::set_is_malloc_allowed(false); // safeguard for detecting unwanted allocations during training

        auto t1 = std::chrono::high_resolution_clock::now();
        double trainingLoss = 0.0;
        for (int i=0; i<nTrainingBatches; ++i) {
            auto batch = sampleBatch(trainingData, datasetSize, batchSize);
            trainingLoss += detector.trainBatch(batch);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        trainingLoss /= nTrainingBatches;

        double evaluationLoss = 0.0;
        for (int i=0; i<nEvaluationBatches; ++i) {
            auto batch = sampleBatch(trainingData, datasetSize, batchSize);
            for (int j = 0; j < batchSize; ++j) {
                double l = detector(batch[j]->f1, batch[j]->f2) - batch[j]->diff;
                if (batch[i]->diff > 0.9f && l > 0.0f)
                    l = 0.0f;
                l *= l;
                evaluationLoss += l;
            }
        }

        evaluationLoss /= batchSize*nEvaluationBatches;

        Eigen::internal::set_is_malloc_allowed(true);

        printf("Epoch %d finished, trainingLoss: %13.10f, evaluationLoss: %13.10f, time: %ld\n",
            e, trainingLoss, evaluationLoss,
            std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());

        if (evaluationLoss < minEvaluationLoss) {
            detector.saveWeights("../feature_detector_model");
            printf("record evaluationLoss, saving\n");
            minEvaluationLoss = evaluationLoss;
        }

        epochStartId += batchesInEpoch*batchSize;
    }
}
