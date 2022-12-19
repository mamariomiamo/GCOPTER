#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <queue>
#include <plan_env/grid_map.h>

//JPS reference
// https://zerowidth.com/2013/a-visual-explanation-of-jump-point-search.html
// https://zhuanlan.zhihu.com/p/25093275
// https://github.com/KumarRobotics/jps3d
// https://ieeexplore-ieee-org.libproxy1.nus.edu.sg/document/7839930
namespace GraphSearch
{
    struct GridNode;
    typedef GridNode *GridNodePtr;
    constexpr double inf = 1 >> 20;

    enum NodeState
    {
        OPENLIST = 1,
        CLOSELIST,
        UNDEFINED
    };

    enum SearchResult
    {
        SUCCESS = 1,
        SUCCESS_SUB,
        SEARCH_ERR,
        INIT_ERR,
        FAILURE,
        TIMEOUT
    };

    struct GridNode
    {
        Eigen::Vector3i index; // node index
        Eigen::Vector3i dir;   // direction of expanding used for JPS

        double g_cost{inf}; // cost from start
        double h_cost{inf}; // heuristic cost
        double f_cost{inf}; // total cost i.e. f_cost = g_cost + h_cost

        GridNodePtr parent{nullptr};

        NodeState state{UNDEFINED};
        int query_rounds{0}; // previous round of query of this node
                             // to be compared with GraphSearch::rounds_
                             // Every call of GraphSearch(start, end, ...)
    };

    struct JPS3DNeib
    {
        // for each (dx,dy,dz) these contain:
        //    ns: neighbors that are always added
        //    f1: forced neighbors to check
        //    f2: neighbors to add if f1 is forced
        int ns[27][3][26];
        int f1[27][3][12];
        int f2[27][3][12];
        // nsz contains the number of neighbors for the four different types of moves:
        // no move (norm 0):        26 neighbors always added
        //                          0 forced neighbors to check (never happens)
        //                          0 neighbors to add if forced (never happens)
        // straight (norm 1):       1 neighbor always added
        //                          8 forced neighbors to check
        //                          8 neighbors to add if forced
        // diagonal (norm sqrt(2)): 3 neighbors always added
        //                          8 forced neighbors to check
        //                          12 neighbors to add if forced
        // diagonal (norm sqrt(3)): 7 neighbors always added
        //                          6 forced neighbors to check
        //                          12 neighbors to add if forced
        static constexpr int nsz[4][2] = {{26, 0}, {1, 8}, {3, 12}, {7, 12}}; // declaration and initialization
        JPS3DNeib() = default;
    };
    // custom comparator for GridNodePtr priority_queue
    class GridNodeComparator
    {
    public:
        bool operator()(GridNodePtr node1, GridNodePtr node2)
        {
            return node1->f_cost > node2->f_cost;
        }
    };

    // GraphSearch class
    class GraphSearch
    {
    public:
        GraphSearch(int map_size, double map_resolution, bool use_jps, bool enable_virtual_ceiling);
        ~GraphSearch();

        SearchResult search(const Eigen::Vector3d &start, const Eigen::Vector3d &goal);

        void initMap(const GridMap::Ptr &map_obj);
        void setCeiling(double ceiling_height);
        // void initMap(const auto &map_obj);

        void getPath(std::vector<Eigen::Vector3d> &path_list);

        bool useJPS() { return this->use_jps_; };

    protected:
        // GridNodePtr ***GridMap_; // 3 dimensional array to store the GridMapPtr
        std::vector<GridNodePtr> GridMap_;

        GridMap::Ptr occupancy_map_;

        GridNodePtr ret_node_;

        int current_search_round_;
        double map_resolution_, map_resolution_inverse_;
        const double tie_breaker_ = 1.0 + 1.0 / 10000;
        bool use_jps_;
        bool enable_virtual_ceiling_;
        double ceiling_height_;

        Eigen::Vector3i map_size_, pool_size_, center_index_;
        Eigen::Vector3i goalIdx;
        Eigen::Vector3d start_, goal_, center_, sub_goal_;

        std::priority_queue<GridNodePtr, std::vector<GridNodePtr>, GridNodeComparator> pq; // pq stores openlist nodes, pq.top() will return the GridNodePtr with the smallest estimated total cost (highest priority)

        void initGridMap();

        virtual void getSuccessorNode(const GridNodePtr &current_node, const GridNodePtr &goal_node);

        bool ConvertToIndexAndAdjustStartEndPoints(Eigen::Vector3d start_pt, Eigen::Vector3d end_pt, Eigen::Vector3i &start_idx, Eigen::Vector3i &end_idx);

        void CheckOutOfBound(Eigen::Vector3d &position);

        inline int checkOccupancy(Eigen::Vector3d node_pos)
        {
            return occupancy_map_->getInflateOccupancy(node_pos);
        }

        inline bool isOccupied(Eigen::Vector3d node_pos)
        {
            return occupancy_map_->getInflateOccupancy(node_pos);
        }

        inline Eigen::Vector3d Index2Coord(const Eigen::Vector3i &index)
        {
            return (index - center_index_).cast<double>() * map_resolution_ + center_;
        }

        inline bool Coord2Index(const Eigen::Vector3d &pos, Eigen::Vector3i &index)
        {
            index = ((pos - center_) * map_resolution_inverse_ + Eigen::Vector3d(0.5, 0.5, 0.5)).cast<int>() + center_index_;
            // std::cout << "coord2index: " << index.transpose() << std::endl;
            // std::cout << "pos: " << pos.transpose() << std::endl;
            if (index(0) < 0)
            {
                index(0) = 1;
            }

            if (index(1) < 0)
            {
                index(1) = 1;
            }

            if (index(2) < 0)
            {
                index(2) = 1;
            }

            if (index(0) >= pool_size_.x())
            {
                index(0) = pool_size_.x() - 1;
            }

            if (index(1) >= pool_size_.y())
            {
                index(0) = pool_size_.y() - 1;
            }

            if (index(2) >= pool_size_.z())
            {
                index(0) = pool_size_.z() - 1;
            }
            return true;
        }

        inline int voxelIndex2vectIndex(const Eigen::Vector3i &index)
        {
            return index.x() + index.y() * pool_size_.x() + index.z() * pool_size_.x() * pool_size_.y();
            // return index.z() + index.y() * pool_size_.z() + index.x() * pool_size_.z() * pool_size_.y();
        }

        inline double getDiagonalHeuristic(const GridNodePtr &node1, const GridNodePtr &node2)
        {
            double dx = abs(node1->index(0) - node2->index(0));
            double dy = abs(node1->index(1) - node2->index(1));
            double dz = abs(node1->index(2) - node2->index(2));

            double h = 0.0;
            int diag = min(min(dx, dy), dz);
            dx -= diag;
            dy -= diag;
            dz -= diag;

            if (dx == 0)
            {
                h = 1.0 * sqrt(3.0) * diag + sqrt(2.0) * min(dy, dz) + 1.0 * abs(dy - dz);
            }
            if (dy == 0)
            {
                h = 1.0 * sqrt(3.0) * diag + sqrt(2.0) * min(dx, dz) + 1.0 * abs(dx - dz);
            }
            if (dz == 0)
            {
                h = 1.0 * sqrt(3.0) * diag + sqrt(2.0) * min(dx, dy) + 1.0 * abs(dx - dy);
            }
            return h;
        }

        inline double getHeuristic(const GridNodePtr &node1, const GridNodePtr &node2)
        {
            return tie_breaker_ * getDiagonalHeuristic(node1, node2);
        }

    }; // class GraphSearch

    class JPS : public GraphSearch
    {
    public:
        JPS3DNeib *jn3d;

        JPS(int map_size, double map_resolution, bool use_jps) : GraphSearch(map_size, map_resolution, use_jps, true)
        {
            jn3d = new JPS3DNeib();
        };

        ~JPS()
        {
            delete jn3d;
        };

        void getSuccessorNode(const GridNodePtr &current_node, const GridNodePtr &goal_node) override;
        void JPSGetSucc(GridNodePtr currentPtr, vector<GridNodePtr> &neighborPtrSets, vector<double> &edgeCostSets,  Eigen::Vector3i &goalIdx);
        bool hasForced(const Eigen::Vector3i &idx, const Eigen::Vector3i &dir);
        bool jump(const Eigen::Vector3i &curIdx, const Eigen::Vector3i &expDir, Eigen::Vector3i &neiIdx);
    };

}; // namespace GraphSearch
