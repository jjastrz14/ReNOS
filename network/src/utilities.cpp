/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: utilities.cpp
//  Description: Set of useful utilis for the definition of the network
//  Created by:  Edoardo Cabiati
//  Date:  18/09/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "utilities.hpp"

//mapIndex

template <class packet, template <typename T> class node>
int Grid<packet,node>::mapIndex(const std::vector<int> & index) const {
    int l_index(0); // linear index
    if (index.size() != _dimensions.size()){
        // raise error
        std::cout<<DIMENSION_ERROR << std::endl;
    }
    for (int i = 0; i < index.size(); i++) {
        int temp = index[i];
        for( int j =index.size()-1-i; j > 0; j--){
            temp *= _dimensions[j];
        }
        l_index +=temp;
    }
    
    return l_index;
};

//reverseIndex
template <class packet, template <class T> class node>
std::vector<int> Grid<packet,node>::reverseIndex(const int & l_index) const {
    std::vector<int> index(_dimensions.size());
    int temp_index=l_index;
    int temp_p;

    for (int i= 0;i < _dimensions.size(); i++){
        temp_p = 1;
        for(int j = i+1; j<_dimensions.size();j++){
            temp_p *=_dimensions[j];
        }
        index[i] = temp_index / temp_p;
        temp_index -= index[i]*temp_p;
    }

    return index;
}

template <class packet, template <class T> class node>
Grid<packet,node>::Grid() : _dimensions({0}), _nodes(std::vector<node<packet>>()) {};

template <class packet, template <class T> class node>
Grid<packet,node>::Grid(std::vector<int> dimensions) : _dimensions(dimensions) {
    // calculate the total number of nodes
    int nNodes = 1;
    for (int i = 0; i < dimensions.size(); i++) {
        nNodes *= dimensions[i];
    }
    // allocate the memory for the nodes
    _nodes.reserve(nNodes);

    // fill the vector with nodes
    for (int i = 0; i < nNodes; i++) {

        // determine the type (Edge, Internal) based on the index
        std::vector<int> index(this->reverseIndex(i));

        // loop over the dimensions of the index and assert if the node is on the border 
        // of the grid
        NodeType type = NodeType::INTERNAL;

        for (int j = 0; j < index.size(); j++){
            if(index[j]==0 || index[j]==_dimensions[j]){
                type = NodeType::EDGE;
            }
        }
    
        _nodes.push_back(node<packet>(i,type,1.));
        
    }

};

template <class packet, template <class T> class node>
Grid<packet,node>::~Grid(){};


template <class packet, template <class T> class node>
std::vector<int> Grid<packet,node>::getDimensions() const {
    return _dimensions;
};

template <class packet, template <class T> class node>
node<packet> Grid<packet,node>::getNode(const std::vector<int> & index) const {
    // call mapping function
    int l_index=this->mapIndex(index);
    // return reference to the node indexed
    return _nodes.at(l_index);
};

template <class packet, template <class T> class node>
std::vector<node<packet>> Grid<packet,node>::getNodes() const {
    return _nodes;
};

template <class packet, template <class T> class node>
void Grid<packet,node>::setNodeStatus(const std::vector<int> & index, const StatusType & type) {
    // call mapping function 
    int l_index=this->mapIndex(index);

    //change the status of the indexed node
    _nodes[l_index].setStatus(type);
};

template <class packet, template <class T> class node>
void Grid<packet,node>::setNodeCommDelay(const std::vector<int> & index, const float & delay) {
    // call mapping function 
    int l_index=this->mapIndex(index);

    //change the delay of the indexed node
    _nodes[l_index].setCommDelay(delay);
}


// -----------------------

// Explicit instantiation
template class Grid<Packet,Node>;



