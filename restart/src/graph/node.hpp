/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: node.hpp
//  Description: Header file for the node class
//  Created by:  Edoardo Cabiati
//  Date:  18/09/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef NODE_HPP
#define NODE_HPP

#include <iostream>
#include <vector>
#include <stdexcept>
#include "packet.hpp"
#include "params.hpp"

enum class NodeType {
    INTERNAL,
    EDGE
};

enum class StatusType {
    BUSY,
    IDLE
};

// struct NodeParams {
//     int id;
//     NodeType type;
//     StatusType status;
//     float commDelay;
// };

template <typename T>
class Node {
    public:

        // default constructor
        Node();
        // constructor with parameters
        Node(int id, NodeType type, float commDelay);

        //copy constructor
        Node(const Node & other);

        //copy assign
        Node & operator=(const Node& other);

        // destructor
        ~Node();

        // setters and getters
        int getId() const;
        NodeType getType() const;
        StatusType getStatus() const;
        float getCommDelay() const;
        std::queue<T> getBuffer() const;

        void setStatus(const StatusType & status);
        void setCommDelay(const float & commDelay);
        void addInBuffer(const T & packet);
        T removeFromBuffer();

    private:
        const int _id;
        const NodeType _type;
        std::queue<T> _buffer;
        StatusType _status;
        float _commDelay;
};
        
#endif // NODE_HPP