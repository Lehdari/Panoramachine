//
// Project: image_demorphing
// File: KdTree.inl
//
// Copyright (c) 2021 Miika 'Lehdari' Lehtimäki
// You may use, distribute and modify this code under the terms
// of the licence specified in file LICENSE which is distributed
// with this source code package.
//

template<typename T_Scalar, int K>
void KdTree<Eigen::Matrix<T_Scalar, K, 1>>::addPoint(const Point& point)
{
    _points.push_back(point);
    _visited.push_back(false);
}

template<typename T_Scalar, int K>
void KdTree<Eigen::Matrix<T_Scalar, K, 1>>::build()
{
    _nodes.clear();
    buildTree(_points.begin(), _points.end(), 0);
}

template<typename T_Scalar, int K>
const Eigen::Matrix<T_Scalar, K, 1>*
KdTree<Eigen::Matrix<T_Scalar, K, 1>>::getNearestNaive(const Point& point)
{
    return getNearestVisitNaive(point);
}

template<typename T_Scalar, int K>
void KdTree<Eigen::Matrix<T_Scalar, K, 1>>::getKNearestNaive(
    const Eigen::Matrix<T_Scalar, K, 1>& point, int k,
    std::vector<const Point*>& kNearest)
{
    kNearest.clear();
    for (int i=0; i<k; ++i) {
        kNearest.push_back(getNearestVisitNaive(point, true));
    }

    for (auto& id : _visitedIndices)
        _visited[id] = false;
    _visitedIndices.clear();
}

template<typename T_Scalar, int K>
const Eigen::Matrix<T_Scalar, K, 1>*
KdTree<Eigen::Matrix<T_Scalar, K, 1>>::getNearest(const Point& point)
{
    T_Scalar dis = std::numeric_limits<T_Scalar>::max();
    return &_points[searchNode(_nodes.at(0), point, dis)->pointId];
}

template<typename T_Scalar, int K>
void KdTree<Eigen::Matrix<T_Scalar, K, 1>>::getKNearest(
    const Eigen::Matrix<T_Scalar, K, 1>& point, int k,
    std::vector<const Point*>& kNearest)
{
    kNearest.clear();
    for (int i=0; i<k; ++i) {
        T_Scalar dis = std::numeric_limits<T_Scalar>::max();
        Node* node = searchNode(_nodes.at(0), point, dis);
        kNearest.push_back(&_points[node->pointId]);
        node->fullyVisited = true;
        _visitedNodes.push_back(node);
    }

    for (auto& node : _visitedNodes)
        node->fullyVisited = false;
    _visitedNodes.clear();
}

template<typename T_Scalar, int K>
const Eigen::Matrix<T_Scalar, K, 1>*
KdTree<Eigen::Matrix<T_Scalar, K, 1>>::getNearestVisitNaive(
    const Point& point, bool markVisit)
{
    float minDis = std::numeric_limits<float>::max();
    size_t nearestPoint = 0;

    for (size_t i=0; i<_points.size(); ++i) {
        float dis = (_points[i]-point).squaredNorm();
        if (dis < minDis && !_visited[i]) {
            minDis = dis;
            nearestPoint = i;
        }
    }

    if (markVisit) {
        _visited[nearestPoint] = true;
        _visitedIndices.push_back(nearestPoint);
    }

    return &_points[nearestPoint];
}

template<typename T_Scalar, int K>
int KdTree<Eigen::Matrix<T_Scalar, K, 1>>::buildTree(
    const typename Vector::iterator& begin,
    const typename Vector::iterator& end,
    int d)
{
    int nodeId = _nodes.size();
    _nodes.emplace_back();

    size_t rangeLen = std::distance(begin, end);
    if (rangeLen > 1) {
        std::sort(begin, end, [&d](const Point& a, const Point& b) {
            return a(d) < b(d);
        });

        auto median = begin + rangeLen / 2;

        _nodes[nodeId].dSplit = d;
        _nodes[nodeId].pSplit = (*median)(d);
        _nodes[nodeId].left = buildTree(begin, median, (d + 1) % K);
        _nodes[nodeId].right = buildTree(median, end, (d + 1) % K);
    }
    else {
        _nodes[nodeId].pointId = std::distance(_points.begin(), begin);
    }

    return nodeId;
}

template<typename T_Scalar, int K>
typename KdTree<Eigen::Matrix<T_Scalar, K, 1>>::Node*
KdTree<Eigen::Matrix<T_Scalar, K, 1>>::searchNode(Node& node, const Point& point, T_Scalar& dis)
{
    if (node.fullyVisited)
        return nullptr;

    if (node.pointId == -1) {
        if (_nodes[node.left].fullyVisited && _nodes[node.right].fullyVisited) {
            node.fullyVisited = true;
            _visitedNodes.emplace_back(&node);
            return nullptr;
        }

        if (point(node.dSplit) < node.pSplit) {
            Node* nearest = searchNode(_nodes[node.left], point, dis);
            if (nearest == nullptr)
                return searchNode(_nodes[node.right], point, dis);

            if (point(node.dSplit)+dis >= node.pSplit) {
                Node* nearest2 = searchNode(_nodes[node.right], point, dis);
                if (nearest2 != nullptr)
                    return nearest2;
            }

            return nearest;
        }
        else {
            Node* nearest = searchNode(_nodes[node.right], point, dis);
            if (nearest == nullptr)
                return searchNode(_nodes[node.left], point, dis);

            if (point(node.dSplit)-dis < node.pSplit) {
                Node* nearest2 = searchNode(_nodes[node.left], point, dis);
                if (nearest2 != nullptr)
                    return nearest2;
            }

            return nearest;
        }
    }
    else {
        T_Scalar newDis = (_points[node.pointId]-point).norm();
        if (newDis > dis)
            return nullptr;

        dis = newDis;
        return &node;
    }
}
