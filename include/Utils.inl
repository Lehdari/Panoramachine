//
// Project: image_demorphing
// File: Utils.inl
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//


template <typename T>
inline __attribute__((always_inline)) T cubicInterpolate(const T* s[4], float x)
{
    return *s[1] + 0.5 * x *
        (*s[2] - *s[0] + x * (2.0 * *s[0] - 5.0 * *s[1] + 4.0 * *s[2] - *s[3] + x * (3.0 * (*s[1] - *s[2]) + *s[3] - *s[0])));
}

template <typename T>
inline __attribute__((always_inline)) T cubicInterpolate(const T (&s)[4], float x)
{
    return s[1] + 0.5 * x *
        (s[2] - s[0] + x * (2.0 * s[0] - 5.0 * s[1] + 4.0 * s[2] - s[3] + x * (3.0 * (s[1] - s[2]) + s[3] - s[0])));
}

template <typename T>
inline __attribute__((always_inline)) T bicubicInterpolate(const T* s[4][4], const Vec2f& p)
{
    T arr[4];
    arr[0] = cubicInterpolate((const T**)s[0], p(0));
    arr[1] = cubicInterpolate((const T**)s[1], p(0));
    arr[2] = cubicInterpolate((const T**)s[2], p(0));
    arr[3] = cubicInterpolate((const T**)s[3], p(0));
    return cubicInterpolate(arr, p(1));
}

template <typename T>
T sampleMatCubic(const cv::Mat& m, const Vec2f& p)
{
    const T* samples[4][4];

    for (int j=0; j<4; ++j) {
        for (int i=0; i<4; ++i) {
            samples[j][i] = &m.at<T>(
                std::clamp((int)p(1)+j-1, 0, m.rows-1),
                std::clamp((int)p(0)+i-1, 0, m.cols-1));
        }
    }

    return bicubicInterpolate(samples, p-p.array().floor().matrix());
}
