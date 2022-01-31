//
// Project: panoramachine
// File: FeatureDataset.cpp
//
// Copyright (c) 2022 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include <filesystem>
#include <opencv2/highgui.hpp> // TODO TEMP
#include <opencv2/imgproc.hpp> // TODO TEMP
#include "FeatureDataset.hpp"
#include "ImagePostProcessing.hpp"


#define RND ((_rnd()%1000001)*0.000001)


std::default_random_engine FeatureDataset::_rnd(1507715517);


FeatureDataset::FeatureDataset(const std::vector<Image<Vec3f>>& images) :
    _images (images)
{
}

void FeatureDataset::construct(size_t size)
{
    _features1.resize(size);
    _features2.resize(size);
    _labels.resize(size);
    _entries.reserve(size);
    for (size_t i=0; i<size; ++i) {
        _entries.push_back(Entry{&_features1[i], &_features2[i], &_labels[i]});
    }

    auto i=0lu;
    for (auto& e : _entries) {
        printf("Processing dataset entry %lu / %lu\n", i++, size);
        if (_rnd()%2)
            createRandomPositiveEntry(e, RND);
        else
            createRandomNegativeEntry(e); // TODO hard negative mining?
    }
}

void FeatureDataset::generateNewEntries(const ConstIterator& begin, const ConstIterator& end, double replaceProbability)
{
    for (auto it = begin; it != end; ++it) {
        auto& e = *it;
        if (replaceProbability > RND) {
            if (_rnd()%2)
                createRandomPositiveEntry(e, RND);
            else
                createRandomNegativeEntry(e); // TODO hard negative mining?
        }
    }
}

void FeatureDataset::shuffle(const Iterator& begin, const Iterator& end)
{
    std::shuffle(begin, end, _rnd);
}

FeatureDataset::Iterator FeatureDataset::begin()
{
    return _entries.begin();
}

FeatureDataset::Iterator FeatureDataset::end()
{
    return _entries.end();
}

FeatureDataset::ConstIterator FeatureDataset::begin() const
{
    return _entries.begin();
}

FeatureDataset::ConstIterator FeatureDataset::end() const
{
    return _entries.end();
}

void FeatureDataset::writeToDirectory(const std::string& directory)
{
    // sanity checks
    assert(_features1.size() == _features2.size());
    assert(_features2.size() == _labels.size());

    // create directory in case it doesn't exist
    std::filesystem::create_directories(directory);

    std::filesystem::path dir(directory);
    for (size_t i=0; i<_features1.size(); ++i) {
        std::stringstream f1Name, f2Name, labelName;
        f1Name << "f1_" << std::setfill('0') << std::setw(6) << i << ".bin";
        f2Name << "f2_" << std::setfill('0') << std::setw(6) << i << ".bin";
        labelName << "label_" << std::setfill('0') << std::setw(6) << i << ".bin";
        _features1[i].writeToFile(dir / f1Name.str());
        _features2[i].writeToFile(dir / f2Name.str());
        writeMatrixBinary(dir / labelName.str(), _labels[i]);
    }
}

void FeatureDataset::readFromDirectory(const std::string& directory)
{
    // find number of files
    size_t n = 0;
    for (auto& p : std::filesystem::directory_iterator(directory))
        ++n;
    n /= 3;

    _features1.clear();
    _features2.clear();
    _labels.clear();
    _entries.clear();
    _features1.resize(n);
    _features2.resize(n);
    _labels.resize(n);

    // load dataset
    std::filesystem::path dir(directory);
    for (size_t i=0; i<n; ++i) {
        std::stringstream f1Name, f2Name, labelName;
        f1Name << "f1_" << std::setfill('0') << std::setw(6) << i << ".bin";
        f2Name << "f2_" << std::setfill('0') << std::setw(6) << i << ".bin";
        labelName << "label_" << std::setfill('0') << std::setw(6) << i << ".bin";
        _features1[i].readFromFile(dir / f1Name.str());
        _features2[i].readFromFile(dir / f2Name.str());
        readMatrixBinary(dir / labelName.str(), _labels[i]);
        _entries.push_back(Entry{&_features1[i], &_features2[i], &_labels[i]});
    }
}

void FeatureDataset::createRandomPositiveEntry(const Entry& entry, double diff)
{
    size_t imgId = _rnd() % _images.size();
    auto& img = _images[imgId];
    double scale = std::pow(2.0, -2.0+RND*8.0);
    double diffDis = diff*scale*Feature::fmr;
    double diffDir = RND*2.0*M_PI;
    Vec2f diffVec(diffDis*std::cos(diffDir), diffDis*std::sin(diffDir));
    Vec2f p(diffDis + RND*(img[0].cols-2.0*diffDis), diffDis + RND*(img[0].rows-2.0*diffDis));

    FeatureDataset::DistortSettings settings1 = {
        p-0.5*diffVec,
        (float)(scale*Feature::fmr),
        0.8f+0.4f*(float)RND,
        0.15f-0.3f*(float)RND
    };
    FeatureDataset::DistortSettings settings2 = {
        p+0.5*diffVec,
        (float)(scale*Feature::fmr),
        0.8f+0.4f*(float)RND,
        0.15f-0.3f*(float)RND
    };

    Mat3f t1, t2, t;
    auto img1 = distortImage(img, settings1, &t1);
    auto img2 = distortImage(img, settings2, &t2);
    t = (t2.inverse() * t1).inverse(); // transformation from img1 to img2

    // post-processing
    float bbase1 = RND*0.5f-0.25f;
    float cbase1 = RND*0.5f-0.25f;
    brightnessContrast(img1,
        Vec3f(bbase1+RND*0.1f-0.05f, bbase1+RND*0.1f-0.05f, bbase1+RND*0.1f-0.05f),
        Vec3f(cbase1+RND*0.1f-0.05f, cbase1+RND*0.1f-0.05f, cbase1+RND*0.1f-0.05f));
    float bbase2 = RND*0.5f-0.25f;
    float cbase2 = RND*0.5f-0.25f;
    brightnessContrast(img2,
        Vec3f(bbase2+RND*0.1f-0.05f, bbase2+RND*0.1f-0.05f, bbase2+RND*0.1f-0.05f),
        Vec3f(cbase2+RND*0.1f-0.05f, cbase2+RND*0.1f-0.05f, cbase2+RND*0.1f-0.05f));

    *entry.f1 = Feature(img1, Vec2f(img1[0].cols*0.5f, img1[0].rows*0.5f), scale);
    *entry.f2 = Feature(img2, Vec2f(img2[0].cols*0.5f, img2[0].rows*0.5f), scale);

    Label& label = *entry.label;
    label << img1[0].cols*0.5f, img1[0].rows*0.5f, 1.0f;
    label = (t * label - label) / (scale*Feature::fmr); // how much the center point has moved, w.r.t. the feature scale
    label(2) = 0.0f;

#if 0
    Vec2f cp21 = (t * Vec3f(img2[0].cols*0.5f, img2[0].rows*0.5f, 1.0f)).block<2,1>(0,0);
    cv::circle(img1[0], cv::Point(img1[0].cols*0.5f, img1[0].rows*0.5f), scale*Feature::fmr, cv::Scalar(1.0f, 1.0f, 1.0f));
    cv::circle(img1[0], cv::Point(cp21(0), cp21(1)), scale*Feature::fmr, cv::Scalar(0.5f, 0.5f, 0.5f));
    cv::circle(img2[0], cv::Point(img2[0].cols*0.5f, img2[0].rows*0.5f), scale*Feature::fmr, cv::Scalar(1.0f, 1.0f, 1.0f));
    cv::imshow("img1", img1[0]);
    cv::imshow("img2", img2[0]);
    printf("%0.5f %0.5f\n", label(0), label(1));
    cv::waitKey(0);
#endif
}

void FeatureDataset::createRandomNegativeEntry(const Entry& entry)
{
    size_t imgId1 = _rnd() % _images.size();
    size_t imgId2 = _rnd() % _images.size();
    auto& img1 = _images[imgId1];
    auto& img2 = _images[imgId2];
    double scale = std::pow(2.0, -2.0+RND*8.0);
    Vec2f p1(
        scale*Feature::fmr*0.5 + RND*(img1[0].cols-scale*Feature::fmr),
        scale*Feature::fmr*0.5 + RND*(img1[0].rows-scale*Feature::fmr));
    Vec2f p2(
        scale*Feature::fmr*0.5 + RND*(img2[0].cols-scale*Feature::fmr),
        scale*Feature::fmr*0.5 + RND*(img2[0].rows-scale*Feature::fmr));
    if (imgId1 == imgId2) {
        while((p1-p2).norm() < scale*Feature::fmr*2.0) {
            p1 <<
                scale*Feature::fmr*0.5 + RND*(img1[0].cols-scale*Feature::fmr),
                scale*Feature::fmr*0.5 + RND*(img1[0].rows-scale*Feature::fmr);
            p2 <<
                scale*Feature::fmr*0.5 + RND*(img2[0].cols-scale*Feature::fmr),
                scale*Feature::fmr*0.5 + RND*(img2[0].rows-scale*Feature::fmr);
        }
    }

    FeatureDataset::DistortSettings settings1 = {
        p1,
        (float)(scale*Feature::fmr),
        0.9f+0.2f*(float)RND,
        0.1f-0.2f*(float)RND
    };
    FeatureDataset::DistortSettings settings2 = {
        p2,
        (float)(scale*Feature::fmr),
        0.9f+0.2f*(float)RND,
        0.1f-0.2f*(float)RND
    };

    auto img1d = distortImage(img1, settings1);
    auto img2d = distortImage(img2, settings2);

    // post-processing
    float bbase1 = RND*0.5f-0.25f;
    float cbase1 = RND*0.5f-0.25f;
    brightnessContrast(img1d,
        Vec3f(bbase1+RND*0.1f-0.05f, bbase1+RND*0.1f-0.05f, bbase1+RND*0.1f-0.05f),
        Vec3f(cbase1+RND*0.1f-0.05f, cbase1+RND*0.1f-0.05f, cbase1+RND*0.1f-0.05f));
    float bbase2 = RND*0.5f-0.25f;
    float cbase2 = RND*0.5f-0.25f;
    brightnessContrast(img2d,
        Vec3f(bbase2+RND*0.1f-0.05f, bbase2+RND*0.1f-0.05f, bbase2+RND*0.1f-0.05f),
        Vec3f(cbase2+RND*0.1f-0.05f, cbase2+RND*0.1f-0.05f, cbase2+RND*0.1f-0.05f));

    *entry.f1 = Feature(img1d, Vec2f(img1d[0].cols*0.5f, img1d[0].rows*0.5f), scale);
    *entry.f2 = Feature(img2d, Vec2f(img2d[0].cols*0.5f, img2d[0].rows*0.5f), scale);
    *entry.label << 1.0f, 1.0f, 0.0f;
#if 0
    cv::circle(img1d[0], cv::Point(img1d[0].cols*0.5f, img1d[0].rows*0.5f), scale*Feature::fmr, cv::Scalar(1.0f, 1.0f, 1.0f));
    cv::circle(img2d[0], cv::Point(img2d[0].cols*0.5f, img2d[0].rows*0.5f), scale*Feature::fmr, cv::Scalar(1.0f, 1.0f, 1.0f));
    cv::imshow("img1", img1d[0]);
    cv::imshow("img2", img2d[0]);
    cv::waitKey(0);
#endif
}

Image<Vec3f> FeatureDataset::distortImage(
    const Image<Vec3f>& image,
    const FeatureDataset::DistortSettings& settings,
    Mat3f* t)
{
    int size = ceil(settings.featureRadius*2.0f+1024.0f);
    cv::Mat m(size, size, CV_32FC3);

    Mat3f tSum, tt1, tr, ts, tt2;
    tt1 <<  Mat2f::Identity(),          -settings.p,
            Vec2f::Zero().transpose(),  1.0f;
    tr  <<  cosf(settings.rotation),    -sinf(settings.rotation),   0.0f,
            sinf(settings.rotation),    cosf(settings.rotation),    0.0f,
            0.0f,                       0.0f,                       1.0f;
    ts  <<  settings.scale, 0.0f,           0.0f,
            0.0f,           settings.scale, 0.0f,
            0.0f,           0.0f,           1.0f;
    tt2 <<  1.0f,   0.0f,   size*0.5f,
            0.0f,   1.0f,   size*0.5f,
            0.0f,   0.0f,   1.0f;

    tSum = (tt2*ts*tr*tt1).inverse();
    if (t != nullptr)
        *t = tSum;

    #pragma omp parallel for
    for (int j=0; j<size; ++j) {
        auto* p = m.ptr<Vec3f>(j);
        for (int i=0; i<size; ++i) {
            p[i] = image((tSum * Vec3f(i, j, 1.0f)).block<2,1>(0,0), 1.0f/settings.scale);
        }
    }

    Image<Vec3f> img(m);
    return img;
}
