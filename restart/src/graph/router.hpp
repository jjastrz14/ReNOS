
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: router.hpp
//  Description: Declarations for the router class, which will be used to forward packet in the network
//  Created by:  Edoardo Cabiati
//  Date:  23/09/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef ROUTER_HPP
#define ROUTER_HPP

#include "utilities.hpp"
#include <algorithm>
#include <float.h>
#include <time.h>
#include <set>


// A small set of helper function to implement the A* algorithm

using pFloatVector = std::pair<float, std::vector<int>>;

struct NodeInfo {
    float g;
    float h;
    // pointer to the parent node
    std::vector<int> parent_index;

    float f();
};

inline bool isValidNode(const std::vector<int> & index, const std::vector<int> & dimensions) {
    for (int i = 0; i < index.size(); i++) {
        if (index[i] < 0 || index[i] >= dimensions[i]) {
            return false;
        }
    }
    return true;
};

template <class packet, template <typename T> class node>
inline bool isBusyNode(const std::vector<int> & index, const Topology<packet, node> & topo) {
    return topo.getNode(index).getStatus() == StatusType::BUSY;
};

inline bool isDestination(const std::vector<int> & index, const std::vector<int> & end) {
    return index == end;
};

// Manhattan distance heuristic
inline float heuristicM(const std::vector<int> & start, const std::vector<int> & end) {
    float h = 0;
    for (int i = 0; i < start.size(); i++) {
        h += std::abs(start[i] - end[i]);
    }
    return h;
};

// Euclidean distance heuristic
inline float heuristicE(const std::vector<int> & start, const std::vector<int> & end) {
    float h = 0;
    for (int i = 0; i < start.size(); i++) {
        h += std::pow(start[i] - end[i], 2);
    }
    return std::sqrt(h);
};


template <class packet, template <typename T> class node>
class Router {
    public:
        // constructor with parameters (dimensions of the Grid)
        Router(std::vector<int> dimensions, TopologyType topo_type, std::vector<int> in_point);
        // contructor with parameters (topology)
        Router(Topology<packet, node> topo, std::vector<int> in_point);

        // destructor
        ~Router();

        // getters
        Topology<packet, node> &  getTopology();
        std::vector<int> getInPoint() const;

        // setters
        void setTopology(const Topology<packet, node> & topo);
        void setInPoint(const std::vector<int> & in_point);


        // a function to implement the A* algorithm to find the shortest path between two nodes in the presence of obstacles:
        // node busy or idle.
        // the function returs the vector containing the indexes of the nodes constituting the path
        std::vector<int> findPathAstar(const std::vector<int> & start, const std::vector<int> & end) const;

        // tracing function for the found path
        std::vector<int> tracePath(const std::vector<int> & start, const std::vector<int> & end, const std::vector<NodeInfo> & node_info) const;
        

    private:
        Topology<packet, node> _topology;
        std::vector<int> _in_point; // this member represents the position of the router in the grid
};

#endif // ROUTER_HPP