
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: node.cpp
//  Description: Definitions for the node class
//  Created by:  Edoardo Cabiati
//  Date:  18/09/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "node.hpp"

template <typename T>
Node<T>::Node() : _id(0), _type(NodeType::INTERNAL), _commDelay(COMM_DELAY), _status(StatusType::IDLE), _buffer(std::queue<T>()) {};

template <typename T>
Node<T>::Node(int id, NodeType type, float commDelay) : _id(id), _type(type), _commDelay(commDelay), _status(StatusType::IDLE), _buffer(std::queue<T>()) {};

template <typename T>
Node<T>::~Node() {};

template <typename T>
int Node<T>::getId() const {
    return _id;
}

template <typename T>
NodeType Node<T>::getType() const {
    return _type;
}

template <typename T>
StatusType Node<T>::getStatus() const {
    return _status;
}

template <typename T>
float Node<T>::getCommDelay() const {
    return _commDelay;
}

template <typename T>
std::queue<T> Node<T>::getBuffer() const {
    return _buffer;
}

template <typename T>
void Node<T>::setStatus(const StatusType & status) {
    _status = status;
}

template <typename T>
void Node<T>::setCommDelay(const float & commDelay) {
    _commDelay = commDelay;
}

template <typename T>
void Node<T>::addInBuffer(const T & packet) {
    // Append the packet to the buffer (FIFO)
    _buffer.push(packet);

}

template <typename T>
T Node<T>::removeFromBuffer(){
    // Pop the last element from the buffer
    T top = _buffer.front();
    _buffer.pop();
    return top;
}


template class Node<Packet>;
