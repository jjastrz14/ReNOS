
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: packet.hpp
//  Description: Definitions for the packet class
//  Created by:  Edoardo Cabiati
//  Date:  20/09/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef PACKET_HPP
#define PACKET_HPP


#include <iostream>
#include <vector>


class Packet {
    public:
        // default constructor
        Packet();
        // constructor 
        Packet(int id, int size);
        // destructor
        ~Packet();

        // getters
        int getId() const;
        int getSize() const;

        //setters
        void setId(const int & id);
        void setSize(const int & size);

    private:    
        mutable int _id;
        mutable int _size;    

};

#endif // PACKET_HPP