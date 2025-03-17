
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: utilities.hpp
//  Description: Header for the utilities source file
//  Created by:  Edoardo Cabiati
//  Date:  18/09/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include "node.hpp"

///////////////////////////////////////- ERROR DEFINITIONS -///////////////////////////////////////

const std::string DIMENSION_ERROR = "Dimensions not corresponding";

const std::string INVALID_NODE = "Invalid node";

const std::string BUSY_NODE = "One or more nodes are busy";

const std::string FAILED_SEARCH = "Failed search";

///////////////////////////////////////- ERROR DEFINITIONS -///////////////////////////////////////

template <class packet, template <typename T> class node>
class Grid{
    public:
        // default constructor
        Grid();
        // constructor with parameters (dimensions of the Grid)
        Grid(std::vector<int> dimensions);

        // destructor
        ~Grid();

        // getters
        std::vector<int> getDimensions() const;
        node<packet> getNode(const std::vector<int> & index) const;
        std::vector<node<packet>> getNodes() const;

        //setters (wrappers for the setters of the nodes)
        void setNodeStatus(const int & index, const StatusType & type);
        void setNodeCommDelay(const int & index, const float & delay);
        void setNodeStatus(const std::vector<int> & index, const StatusType & type);
        void setNodeCommDelay(const std::vector<int> & index, const float & delay);

        // a mapping function to convert the multidimensional index to a linear index
        int mapIndex(const std::vector<int> & index) const;
        // function to get the multidimensional index from the linear one, based on the grid dimennsions
        std::vector<int> reverseIndex(const int & l_index) const;

    private:
        std::vector<int> _dimensions;
        // _nodes will be a pointer to a multidimensional array of nodes
        std::vector<node<packet>> _nodes;
        
        
};

enum class TopologyType {
    MESH,
    TORUS
};

// The following class defines the topology for the grid
template <class packet, template <typename T> class node>
class Topology{
    public:
        // user defined constructor
        Topology(const TopologyType & type ,const Grid<packet,node> & grid);
        Topology(const TopologyType & type, const std::vector<int> & dimensions);

        // copy constructor
        Topology(const Topology & other);

        //destructor
        ~Topology();

        //setters
        void reimposeOnGrid(const Grid<packet,node> & grid);
        void setType(const TopologyType & type);

        //getters
        TopologyType getType() const;
        Grid<packet,node> & getGrid();
        std::vector<std::vector<int>> getAdjMat() const;

        //wrappers for the grid getters
        std::vector<int> getDimensions() const;
        node<packet> getNode(const std::vector<int> & index) const;
        std::vector<node<packet>> getNodes() const;
        void setNodeStatus(const int & index, const StatusType & type);
        void setNodeCommDelay(const int & index, const float & delay);
        void setNodeStatus(const std::vector<int> & index, const StatusType & type);
        void setNodeCommDelay(const std::vector<int> & index, const float & delay);
        int mapIndex(const std::vector<int> & index) const;
        std::vector<int> reverseIndex(const int & l_index) const;

        // function used to compute the adjacency matrix, i.e. to reimpose the topology on the 
        // member grid
        void reimpose();
        // functions to compute the adjacent list of nodes given the position/index of the node
        std::vector<int> computeAdjacentNodes(const std::vector<int> & node_coords);
        std::vector<int> computeAdjacentNodes(const int & node_id);


    private:    
        mutable TopologyType _type;
        Grid<packet,node> _grid;
        // cons with this implementation: vector of pointers pointing to locations in memory possibly
        // very distant one with respect to the other
        std::vector<std::vector<int>> _ad_mat;
        
};


#endif // UTILITIES_HPP