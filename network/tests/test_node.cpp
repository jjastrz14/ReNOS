#include "node.hpp"
#include "packet.hpp"
#include <cassert>
#include <iostream>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test Node") {

    SECTION("Constructor 1") {
        Node<Packet> node;
        REQUIRE(node.getId() == 0);
        REQUIRE(node.getType() == NodeType::INTERNAL);
        REQUIRE(node.getCommDelay() == 1.0);
        REQUIRE(node.getStatus() == StatusType::IDLE);
    }

    SECTION("Constructor 2") {
        Node<Packet> node(1, NodeType::EDGE, 1.0);
        REQUIRE(node.getId() == 1);
        REQUIRE(node.getType() == NodeType::EDGE);
        REQUIRE(node.getCommDelay() == 1.0);
        REQUIRE(node.getStatus() == StatusType::IDLE);
    }

    SECTION("Setters and Getters") {
        Node<Packet> node;
        node.setStatus(StatusType::BUSY);
        node.setCommDelay(2.0);
        REQUIRE(node.getStatus() == StatusType::BUSY);
        REQUIRE(node.getCommDelay() == 2.0);
    }
}