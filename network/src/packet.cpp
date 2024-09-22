
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: packet.cpp
//  Description: Definitions for the packet class
//  Created by:  Edoardo Cabiati
//  Date:  20/09/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "packet.hpp"

Packet::Packet() : _id(0), _size(0) {};

Packet::Packet(int id, int size) : _id(id), _size(size) {};

Packet::~Packet() {};

int Packet::getId() const {
    return _id;
};

int Packet::getSize() const {
    return _size;
};

void Packet::setId(const int & id) {
    _id = id;
};

void Packet::setSize(const int & size) {
    _size = size;
};



