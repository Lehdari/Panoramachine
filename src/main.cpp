//
// Project: image_demorphing
// File: main.cpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#include "CorrectionAlgorithms.hpp"
#include "KdTree.hpp"

#include <img_demorph_bm/ImageDemorphingBenchmark.hpp>
#include <img_demorph_bm/Utils.hpp>
#include <opencv2/highgui.hpp>
#include <random>


#define RND ((rnd() % 1000001) * 0.000001)


void testKdTree()
{
    std::default_random_engine rnd(1507715517);

    for (int k=0; k<10; ++k) {
        KdTree<Vec3f> kdTree;
        for (int i = 0; i < 10000; ++i)
            kdTree.addPoint(Vec3f(RND, RND, RND));

        kdTree.build();

        for (int j = 0; j < 25; ++j) {
            Vec3f pSearch = Vec3f(RND, RND, RND);
            auto *p1 = kdTree.getNearestNaive(pSearch);
            auto *p2 = kdTree.getNearest(pSearch);
            assert(p1 == p2);

            int nSearch = 100;
            std::vector<const KdTree<Vec3f>::Point *> pv1, pv2;
            kdTree.getKNearestNaive(pSearch, nSearch, pv1);
            kdTree.getKNearest(pSearch, nSearch, pv2);
            for (int i = 0; i < nSearch; ++i) {
                assert(pv1[i] == pv2[i]);
            }
        }
    }

    printf("K-d tree successfully tested\n");
}


int main(void)
{
    testKdTree();
#if 0
    std::string imageFileName = std::string(IMAGE_DEMORPHING_RES_DIR) + "lenna.exr";
    cv::Mat image = cv::imread(imageFileName, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

    gammaCorrect(image, 1.0f/2.2f);

    ImageDemorphingBenchmark bm(std::move(image));

    auto corr = createCorrection4(bm.getMorphedImage(), bm.getOriginalImage());

    auto imageCorrected = correctImage(bm.getMorphedImage(), corr);

    auto corr2 = corr.clone()*0.025f + 0.5f;
    cv::imshow("corr2", corr2);
    cv::imshow("imageCorrected", imageCorrected);
    cv::waitKey();

    printf("Error: %0.5f\n", bm.evaluate(corr));
#endif
    return 0;
}