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
inline __attribute__((always_inline)) T cubicInterpolateDeriv(const T* s[4], float x)
{
    return 0.5*(x*(x*(3.0*(*s[3]-*s[0])+9.0*(*s[1]-*s[2]))+4.0**s[0]-10.0**s[1]+8.0**s[2]-2.0**s[3])+*s[2]-*s[0]);
}

template <typename T>
inline __attribute__((always_inline)) T cubicInterpolate(const T (&s)[4], float x)
{
    return s[1] + 0.5 * x *
        (s[2] - s[0] + x * (2.0 * s[0] - 5.0 * s[1] + 4.0 * s[2] - s[3] + x * (3.0 * (s[1] - s[2]) + s[3] - s[0])));
}

template <typename T>
inline __attribute__((always_inline)) T cubicInterpolateDeriv(const T (&s)[4], float x)
{
    return 0.5*(x*(x*(3.0*(s[3]-s[0])+9.0*(s[1]-s[2]))+4.0*s[0]-10.0*s[1]+8.0*s[2]-2.0*s[3])+s[2]-s[0]);
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
inline __attribute__((always_inline)) T bicubicInterpolateXDeriv(const T* s[4][4], const Vec2f& p)
{
    T arr[4];
    arr[0] = cubicInterpolate((const T**)s[0], p(1));
    arr[1] = cubicInterpolate((const T**)s[1], p(1));
    arr[2] = cubicInterpolate((const T**)s[2], p(1));
    arr[3] = cubicInterpolate((const T**)s[3], p(1));
    return cubicInterpolateDeriv(arr, p(0));
}

template <typename T>
inline __attribute__((always_inline)) T bicubicInterpolateYDeriv(const T* s[4][4], const Vec2f& p)
{
    T arr[4];
    arr[0] = cubicInterpolate((const T**)s[0], p(0));
    arr[1] = cubicInterpolate((const T**)s[1], p(0));
    arr[2] = cubicInterpolate((const T**)s[2], p(0));
    arr[3] = cubicInterpolate((const T**)s[3], p(0));
    return cubicInterpolateDeriv(arr, p(1));
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

template <typename T>
T sampleMatCubicXDeriv(const cv::Mat& m, const Vec2f& p)
{
    const T* samples[4][4];

    for (int j=0; j<4; ++j) {
        for (int i=0; i<4; ++i) {
            samples[i][j] = &m.at<T>(
                std::clamp((int)p(1)+j-1, 0, m.rows-1),
                std::clamp((int)p(0)+i-1, 0, m.cols-1));
        }
    }

    return bicubicInterpolateXDeriv(samples, p-p.array().floor().matrix());
}

template <typename T>
T sampleMatCubicYDeriv(const cv::Mat& m, const Vec2f& p)
{
    const T* samples[4][4];

    for (int j=0; j<4; ++j) {
        for (int i=0; i<4; ++i) {
            samples[j][i] = &m.at<T>(
                std::clamp((int)p(1)+j-1, 0, m.rows-1),
                std::clamp((int)p(0)+i-1, 0, m.cols-1));
        }
    }

    return bicubicInterpolateYDeriv(samples, p-p.array().floor().matrix());
}

template<class T_Scalar, int Rows, int Cols>
void writeMatrixBinary(std::ofstream& out, const Eigen::Matrix<T_Scalar, Rows, Cols>& matrix){
    out.write((char*) matrix.data(), Rows*Cols*sizeof(T_Scalar));
}

template<class T_Scalar, int Rows, int Cols>
void readMatrixBinary(std::ifstream& in, Eigen::Matrix<T_Scalar, Rows, Cols>& matrix){
    in.read((char*) matrix.data(), Rows*Cols*sizeof(T_Scalar));
}

template<class Matrix>
void writeMatrixBinary(const std::string& filename, const Matrix& matrix){
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    writeMatrixBinary(out, matrix);
    out.close();
}

template<class Matrix>
void readMatrixBinary(const std::string& filename, Matrix& matrix){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    readMatrixBinary(in, matrix);
    in.close();
}

template <typename T>
Image<T> readImage(const std::string& filename)
{
    cv::Mat img;
    cv::imread(filename).convertTo(img, PixelFormatMap::format<T>, 1/255.0);
    return img;
}
