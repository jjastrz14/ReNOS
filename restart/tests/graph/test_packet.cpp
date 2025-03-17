#include <catch2/catch_test_macros.hpp>
#include "packet.hpp"


TEST_CASE("Test Packet") {
    SECTION("Constructor 1") {
        Packet packet;
        REQUIRE(packet.getId() == 0);
        REQUIRE(packet.getSize() == 0);
    }

    SECTION("Constructor 2") {
        Packet packet(1, 2);
        REQUIRE(packet.getId() == 1);
        REQUIRE(packet.getSize() == 2);
    }

    SECTION("Setters and Getters") {
        Packet packet;
        packet.setId(1);
        packet.setSize(2);
        
        REQUIRE(packet.getId() == 1);
        REQUIRE(packet.getSize() == 2);
    }
}
