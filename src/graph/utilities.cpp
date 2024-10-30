/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: utilities.cpp
//  Description: Set of useful utilis for the definition of the network
//  Created by:  Edoardo Cabiati
//  Date:  18/09/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "utilities.hpp"


// --------------------------------------------*     Grid   *--------------------------------------------
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//mapIndex
template <class packet, template <typename T> class node>
int Grid<packet,node>::mapIndex(const std::vector<int> & index) const {
    int l_index(0); // linear index
    if (index.size() != _dimensions.size()){
        // raise error
        throw std::runtime_error(DIMENSION_ERROR);
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
            if(index[j]==0 || index[j]==_dimensions[j]-1){
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
void Grid<packet,node>::setNodeStatus(const int & index, const StatusType & type) {
    //change the status of the indexed node
    _nodes[index].setStatus(type);
};

template <class packet, template <class T> class node>
void Grid<packet,node>::setNodeCommDelay(const int & index, const float & delay) {
    //change the delay of the indexed node
    _nodes[index].setCommDelay(delay);
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

// --------------------------------------------*  Topology  *--------------------------------------------
/////////////////////////////////////////////////////////////////////////////////////////////////////////


template <class packet, template <class T> class node>
Topology<packet,node>::Topology(const TopologyType & type,  const Grid<packet,node> & grid) : _type(type), _grid(grid) {this->reimpose();};

template <class packet, template <class T> class node>
Topology<packet,node>::Topology(const TopologyType & type, const std::vector<int> & dimensions) : _type(type), _grid(Grid<packet,node>(dimensions)) {this->reimpose();};

template <class packet, template <class T> class node>
Topology<packet,node>::Topology(const Topology & other) : _type(other._type), _grid(other._grid), _ad_mat(other._ad_mat) {};

template <class packet, template <class T> class node>
Topology<packet,node>::~Topology(){};

template <class packet, template <class T> class node>
void Topology<packet,node>::reimposeOnGrid(const Grid<packet,node> & grid){
    _grid = grid;
    this->reimpose();
};

template <class packet, template <class T> class node>
void Topology<packet,node>::setType(const TopologyType & type){
    _type = type;
    this->reimpose();
};

template <class packet, template <class T> class node>
TopologyType Topology<packet,node>::getType() const{
    return _type;
};

template <class packet, template <class T> class node>
Grid<packet,node> & Topology<packet,node>::getGrid() {
    return _grid;
};

template <class packet, template <class T> class node>
std::vector<std::vector<int>> Topology<packet,node>::getAdjMat() const{
    return _ad_mat;
};

//////////////////////////////////////////// WRAPPERS FOR GRID METHODS //////////////////////////////////////

template <class packet, template <class T> class node>
std::vector<int> Topology<packet,node>::getDimensions() const{
    return _grid.getDimensions();
};

template <class packet, template <class T> class node>
node<packet> Topology<packet,node>::getNode(const std::vector<int> & index) const{
    return _grid.getNode(index);
};

template <class packet, template <class T> class node>
std::vector<node<packet>> Topology<packet,node>::getNodes() const{
    return _grid.getNodes();
};

template <class packet, template <class T> class node>
void Topology<packet,node>::setNodeStatus(const int & index, const StatusType & type){
    _grid.setNodeStatus(index,type);
};

template <class packet, template <class T> class node>
void Topology<packet,node>::setNodeCommDelay(const int & index, const float & delay){
    _grid.setNodeCommDelay(index,delay);
};

template <class packet, template <class T> class node>
void Topology<packet,node>::setNodeStatus(const std::vector<int> & index, const StatusType & type){
    _grid.setNodeStatus(index,type);
};

template <class packet, template <class T> class node>
void Topology<packet,node>::setNodeCommDelay(const std::vector<int> & index, const float & delay){
    _grid.setNodeCommDelay(index,delay);
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////

template <class packet, template <class T> class node>
int Topology<packet,node>::mapIndex(const std::vector<int> & index) const{
    // call the mapIndex function of the grid
    return _grid.mapIndex(index);
};

template <class packet, template <class T> class node>
std::vector<int> Topology<packet,node>::reverseIndex(const int & l_index) const{
    // call the reverseIndex function of the grid
    return _grid.reverseIndex(l_index);
};

template <class packet, template <class T> class node>
void Topology<packet,node>::reimpose(){
    // Access the dimension of the grid
    std::vector<int> dimensions(_grid.getDimensions());

    // resize _ad_mat based on the dimensions
    int tot_nodes = 1;
    for (auto i : dimensions){
        tot_nodes *= i;
    }
    // resize the _ad_mat vector to host the total number of nodes
    _ad_mat.resize(tot_nodes);

    for(int i = 0; i<_ad_mat.size(); i ++){
        _ad_mat[i]=this->computeAdjacentNodes(i);
    }
};

template <class packet, template <class T> class node>
std::vector<int> Topology<packet,node>::computeAdjacentNodes(const int & node_id){
    // Based on node_id, derive the position in the grid
    std::vector<int> node_coordinates = _grid.reverseIndex(node_id);
    return this->computeAdjacentNodes(node_coordinates);

}

template <class packet, template <class T> class node>
std::vector<int> Topology<packet,node>::computeAdjacentNodes(const std::vector<int> & node_coords){

    // No need to pass also the node type, it can be inferred from the dimensions and type of the grid
    NodeType node_type = NodeType::INTERNAL;
    int edge_order(0);
    std::vector<int> grid_dimensions;
    grid_dimensions = _grid.getDimensions();

    for (int i = 0; i < node_coords.size(); i++){
        // If the node is on the grid border, mark as edge
        if(node_coords[i]==0 || node_coords[i]==grid_dimensions[i]-1){
            node_type = NodeType::EDGE;
            edge_order +=1;
        }
    }

    // Allocate space for the adjacency list
    std::vector<int> adjacent_list;

    // Check the topology type: if MESH, all but the EDGE nodes will have 4 (in 2D) adjacent nodes (EDGEs will have 3)
    if (_type == TopologyType::MESH){
        
        // if node type is internal, we get the id of the neighboring nodes in each dimension:
        if (node_type == NodeType::INTERNAL){

            adjacent_list.reserve(2*grid_dimensions.size());
            for(int i = 0; i < node_coords.size(); i++){
                std::vector<int> temp_coords(node_coords);
                temp_coords[i]=node_coords[i]-1;
                int id_low = _grid.mapIndex(temp_coords);
                temp_coords[i]=node_coords[i]+1;
                int id_high = _grid.mapIndex(temp_coords);
                // append the neighbors to the list
                adjacent_list.push_back(id_low);
                adjacent_list.push_back(id_high);
            }
        }
        // if the node is to the border, the number of adjacent nodes changes depending on the "edge order"
        else if(node_type == NodeType::EDGE){

            adjacent_list.reserve(2*(grid_dimensions.size()-edge_order)+edge_order);
            for(int i = 0; i < node_coords.size(); i++){
                std::vector<int> temp_coords(node_coords);
                // lower edge case
                if(node_coords[i]==0){
                    temp_coords[i]=node_coords[i]+1;
                    // append just this node for this dimension
                    int id_high(_grid.mapIndex(temp_coords));
                    adjacent_list.push_back(id_high);
                }
                else if (node_coords[i]==grid_dimensions[i]-1){
                    temp_coords[i]=node_coords[i]-1;
                    int id_low(_grid.mapIndex(temp_coords));
                    adjacent_list.push_back(id_low);
                }
                else{
                    // the node is not an EDGE in this dimension
                    temp_coords[i]=node_coords[i]-1;
                    int id_low = _grid.mapIndex(temp_coords);
                    temp_coords[i]=node_coords[i]+1;
                    int id_high = _grid.mapIndex(temp_coords);
                    // append the neighbors to the list
                    adjacent_list.push_back(id_low);
                    adjacent_list.push_back(id_high);
                } 
            }
        }     
    }
    else if (_type == TopologyType::TORUS){ // "else if" in case of later extension of types for the topology class
        // if the network has torus dimension, each node has the same number of neighbors (2*grid_dimensions.size())
        adjacent_list.reserve(2*grid_dimensions.size());
        // if the node is internal, we can just compute the neighbors the same way we did before
        if (node_type == NodeType::INTERNAL){
            for(int i = 0; i < node_coords.size(); i++){
                std::vector<int> temp_coords(node_coords);
                temp_coords[i]=node_coords[i]-1;
                int id_low = _grid.mapIndex(temp_coords);
                temp_coords[i]=node_coords[i]+1;
                int id_high = _grid.mapIndex(temp_coords);
                // append the neighbors to the list
                adjacent_list.push_back(id_low);
                adjacent_list.push_back(id_high);
            }
        }
        else if(node_type == NodeType::EDGE){
            for(int i = 0; i < node_coords.size(); i++){
                std::vector<int> temp_coords(node_coords);
                if(node_coords[i]==0){
                    temp_coords[i]=node_coords[i]+1;
                    int id_high(_grid.mapIndex(temp_coords));
                    temp_coords[i]=grid_dimensions[i]-1;
                    int id_low(_grid.mapIndex(temp_coords));

                    adjacent_list.push_back(id_high);
                    adjacent_list.push_back(id_low);
                }
                else if (node_coords[i]==grid_dimensions[i]-1){
                    temp_coords[i]=node_coords[i]-1;
                    int id_low(_grid.mapIndex(temp_coords));
                    temp_coords[i]=0;
                    int id_high(_grid.mapIndex(temp_coords));

                    adjacent_list.push_back(id_low);
                    adjacent_list.push_back(id_high);
                }
                else{
                    // the node is not an EDGE in this dimension
                    temp_coords[i]=node_coords[i]-1;
                    int id_low = _grid.mapIndex(temp_coords);
                    temp_coords[i]=node_coords[i]+1;
                    int id_high = _grid.mapIndex(temp_coords);

                    adjacent_list.push_back(id_low);
                    adjacent_list.push_back(id_high);
                } 
            }
        }
    }
    return adjacent_list;
}


// -----------------------

// Explicit instantiation
template class Grid<Packet,Node>;
template class Topology<Packet,Node>;



