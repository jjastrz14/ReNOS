/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: router.cpp
//  Description: Definitions for the router class
//  Created by:  Edoardo Cabiati
//  Date:  23/09/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////


#include "router.hpp"

//////////////////Sub router types//////////////////////
#include "iq_router.hpp"
#include "event_router.hpp"
#include "chaos_router.hpp"
///////////////////////////////////////////////////////


float NodeInfo::f() {
    return g + h;
}

template <class packet, template <typename T> class node>
Router<packet, node>::Router(std::vector<int> dimensions, TopologyType topo_type, std::vector<int> in_point) : _topology( topo_type, dimensions), _in_point(in_point) {};

template <class packet, template <typename T> class node>
Router<packet, node>::Router(Topology<packet, node> topo, std::vector<int> in_point) : _topology(topo), _in_point(in_point) {};

template <class packet, template <typename T> class node>
Router<packet, node>::~Router() {};

template <class packet, template <typename T> class node>
Topology<packet, node> & Router<packet, node>::getTopology() {
    return _topology;
};

template <class packet, template <typename T> class node>
std::vector<int> Router<packet, node>::getInPoint() const {
    return _in_point;
};

template <class packet, template <typename T> class node>
void Router<packet,node>::setTopology(const Topology<packet,node> & topo) { _topology = topo; };

template <class packet, template <typename T> class node>
void Router<packet,node>::setInPoint(const std::vector<int> & in_point) { _in_point = in_point; };

template <class packet, template <typename T> class node>
std::vector<int> Router<packet, node>::tracePath(const std::vector<int> & start, const std::vector<int> & end, const std::vector<NodeInfo> & node_info) const{
    std::vector<int> path;
    std::vector<int> current_node = end;
    int current_index = _topology.mapIndex(current_node);

    while (current_node != start){
        path.push_back(current_index);
        current_node = node_info[current_index].parent_index;
        current_index = _topology.mapIndex(current_node);
    }

    path.push_back(_topology.mapIndex(start));
    std::reverse(path.begin(), path.end());
    return path;
};


template <class packet, template <typename T> class node>
std::vector<int> Router<packet, node>::findPathAstar(const std::vector<int> & start, const std::vector<int> & end) const {
    
    std::vector<int> dimensions = _topology.getDimensions();

    // check if the start and end points are valid
    if (!isValidNode(start, _topology.getDimensions()) || !isValidNode(end, _topology.getDimensions())) {
        throw std::invalid_argument(INVALID_NODE);
    }

    // check if one of them is a busy node
    if (isBusyNode(start, _topology) || isBusyNode(end, _topology)) {
        throw std::invalid_argument(BUSY_NODE);
    }

    // check if the start and end points are the same
    if (isDestination(start, end)) {
        throw std::invalid_argument("The start and end points are the same");
    }

    // create closed list 
    std::vector<bool> closed_list(_topology.getNodes().size(), false);
    std::vector<NodeInfo> node_info;
    //reserve space for the node_info vector
    node_info.reserve(_topology.getNodes().size());


    for (int i = 0; i < _topology.getNodes().size(); i++) {
        node_info[i].g = FLT_MAX;
        node_info[i].h = FLT_MAX;
        node_info[i].parent_index.resize(_topology.getDimensions().size());
        std::fill(node_info[i].parent_index.begin(), node_info[i].parent_index.end(), -1);
    }

    // initialize the starting node and add it to the open list (set)
    int start_index = _topology.mapIndex(start);
    node_info[start_index].g = 0;
    node_info[start_index].h = 0;
    node_info[start_index].parent_index = start;

    std::set<pFloatVector> open_list; // pair of f=g+h and multidimensional index
    open_list.insert({0, start});

    bool foundDest = false;

    while (!open_list.empty()){

        pFloatVector p = *open_list.begin();
        open_list.erase(open_list.begin());

        int current_index = _topology.mapIndex(p.second);
        closed_list[current_index] = true;

        float gNext, hNext, fNext;

        // Loop over the successors of the current node (8 successor nodes)
        for (int k = 0; k < dimensions.size(); k++){
            for(int t = -1; t < 2; t++){
                std::vector<int> succ = p.second;
                if (_topology.getType() == TopologyType::MESH){
                    succ[k] += t;
                }
                else if (_topology.getType() == TopologyType::TORUS){
                    succ[k] = (succ[k] + t + dimensions[k]) % dimensions[k];
                }
                if (isValidNode(succ, dimensions)){
                    int succ_index = _topology.mapIndex(succ);
                    if (isDestination(succ, end)){
                        node_info[succ_index].parent_index = p.second;
                        std::cout << "The destination node has been found" << std::endl;
                        // trace the path
                        foundDest = true;
                        return tracePath(start, end, node_info);
                    }
                    else if (!closed_list[succ_index] && !isBusyNode(succ, _topology)){
                        gNext = node_info[current_index].g + 1.;
                        hNext = heuristicM(succ, end); // Manhattan distance
                        fNext = gNext + hNext;

                        if(node_info[succ_index].f() == FLT_MAX || node_info[succ_index].f() > fNext){
                            open_list.insert({fNext, succ});
                            node_info[succ_index].g = gNext;
                            node_info[succ_index].h = hNext;
                            node_info[succ_index].parent_index = p.second;
                        }
                    }
                }
            }
        }
    }
    
    // throw runtime error if the destination node has not been found
    throw std::runtime_error(FAILED_SEARCH);
}

// Explicit instantiation of the class
template class Router<Packet, Node>;