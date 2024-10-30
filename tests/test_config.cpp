// -*- test for config.cpp -*-

#include <iostream>
#include <catch2/catch_test_macros.hpp>
#include "config.hpp"

TEST_CASE("Test Config") {
    Configuration * c = new Configuration();
    // call the parseJSONFile method
    c->parseJSONFile("../network/tests/config.json");
    // check if the values are correctly assigned
    REQUIRE(c->getStrField("topology") == "mesh2D");
    REQUIRE(c->getIntField("nodes") == 144 );
    REQUIRE(c->getIntField("buffer_size") == 1000 );
    REQUIRE(c->getIntField("packet_size") == 1000 );
    REQUIRE(c->getIntArray("private_buffer_size").size() == 4);
    REQUIRE(c->getIntArray("private_buffer_size")[0] == 9);
    REQUIRE(c->getIntArray("private_buffer_size")[1] == 4);


    int index = 0;
    auto firstPacket = c->getPacket(index);
    REQUIRE(firstPacket.id == 0);
    REQUIRE(firstPacket.src == 0);
    REQUIRE(firstPacket.dst == 1);
    REQUIRE(firstPacket.size == 1000);
    REQUIRE(firstPacket.dep == -1);
    REQUIRE(firstPacket.cl == 0);
    REQUIRE(firstPacket.type == 2);
    REQUIRE(firstPacket.pt_required == 10);
}
