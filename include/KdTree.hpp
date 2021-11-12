//
// Project: image_demorphing
// File: KdTree.hpp
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

#ifndef IMAGE_DEMORPHING_KDTREE_HPP
#define IMAGE_DEMORPHING_KDTREE_HPP


#include <Eigen/Dense>


template <typename... Args>
class KdTree {
};

template<typename T_Scalar, int K>
class KdTree<Eigen::Matrix<T_Scalar, K, 1>> {
public:
    using Point = Eigen::Matrix<T_Scalar, K, 1>;
    using Vector = std::vector<Point, Eigen::aligned_allocator<Point>>;

    void addPoint(const Point& point);
    void build();

    const Point* getNearest(const Point& point);
    void getKNearest(const Point& point, int k, std::vector<const Point*>& kNearest);

private:
    struct Node {
        int         pointId = -1;
        int         left = -1;
        int         right = -1;
        int         dSplit;
        T_Scalar    pSplit;
        bool        fullyVisited = false;
    };

    Vector                      _points;
    std::vector<unsigned char>  _visited;
    std::vector<size_t>         _visitedIndices;
    std::vector<Node>           _nodes;
    std::vector<Node*>          _visitedNodes;

    int buildTree(
        const typename Vector::iterator& begin,
        const typename Vector::iterator& end,
        int d);

    Node* searchNode(Node& node, const Point& point, T_Scalar& dis);
};


template<typename T_Scalar, int K, typename T_Handle>
class KdTree<Eigen::Matrix<T_Scalar, K, 1>, T_Handle> {
public:
    using Point = Eigen::Matrix<T_Scalar, K, 1>;
    using PointerPair = std::pair<const Point*, T_Handle*>;
    using Pair = std::pair<Point, T_Handle*>;
    using Vector = std::vector<Pair, Eigen::aligned_allocator<Pair>>;

    void addPoint(const Point& point, T_Handle* handle);
    void build();

    Pair getNearest(const Point& point);
    void getKNearest(const Point& point, int k, std::vector<PointerPair>& kNearest);

private:
    struct Node {
        int         pointId = -1;
        int         left = -1;
        int         right = -1;
        int         dSplit;
        T_Scalar    pSplit;
        bool        fullyVisited = false;
    };

    Vector                      _points;
    std::vector<unsigned char>  _visited;
    std::vector<size_t>         _visitedIndices;
    std::vector<Node>           _nodes;
    std::vector<Node*>          _visitedNodes;

    int buildTree(
        const typename Vector::iterator& begin,
        const typename Vector::iterator& end,
        int d);

    Node* searchNode(Node& node, const Point& point, T_Scalar& dis);
};


#include "KdTree.inl"


#endif //IMAGE_DEMORPHING_KDTREE_HPP
