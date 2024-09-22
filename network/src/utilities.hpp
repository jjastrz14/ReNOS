
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
#include "node.hpp"

///////////////////////////////////////- ERROR DEFINITIONS -///////////////////////////////////////

const std::string DIMENSION_ERROR = "Dimensions not corresponding";

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
        void setNodeStatus(const std::vector<int> & index, const StatusType & type);
        void setNodeCommDelay(const std::vector<int> & index, const float & delay);

    private:
        std::vector<int> _dimensions;
        // _nodes will be a pointer to a multidimensional array of nodes
        std::vector<node<packet>> _nodes;
        // a mapping function to convert the multidimensional index to a linear index
        int mapIndex(const std::vector<int> & index) const;
        // function to get the multidimensional index from the linear one, based on the grid dimennsions
        std::vector<int> reverseIndex(const int & l_index) const;
};






#endif // UTILITIES_HPP