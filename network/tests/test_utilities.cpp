#include <iostream>
#include <vector>
#include "utilities.hpp"
#include "packet.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test Grid") {
    SECTION("Constructor 1") {
        Grid<Packet,Node> grid;
        REQUIRE(grid.getDimensions().size() == 1);
        REQUIRE(grid.getDimensions()[0] == 0);
        REQUIRE(grid.getNodes().size() == 0);
    }

    SECTION("Constructor 2") {
        std::vector<int> dimensions = {2, 3};
        Grid<Packet,Node> grid(dimensions);
        REQUIRE(grid.getDimensions().size() == 2);
        REQUIRE(grid.getDimensions()[0] == 2);
        REQUIRE(grid.getDimensions()[1] == 3);
        REQUIRE(grid.getNodes().size() == 6);
    }

    SECTION("getDimensions") {
        std::vector<int> dimensions = {2, 3};
        Grid<Packet, Node> grid(dimensions);
        REQUIRE(grid.getDimensions().size() == 2);
        REQUIRE(grid.getDimensions()[0] == 2);
        REQUIRE(grid.getDimensions()[1] == 3);
    }

    SECTION("getNode") {
        std::vector<int> dimensions = {2, 3};
        Grid<Packet, Node> grid(dimensions);
        REQUIRE(grid.getNode({0, 0}).getId() == 0);
        REQUIRE(grid.getNode({0, 1}).getType() == NodeType::EDGE);
        REQUIRE(grid.getNode({1, 2}).getId() == 5);
    }

    SECTION("setNodeStatus") {
        std::vector<int> dimensions = {2, 3};
        Grid<Packet, Node> grid(dimensions);
        grid.setNodeStatus({0, 0}, StatusType::BUSY);
        REQUIRE(grid.getNode({0, 0}).getStatus() == StatusType::BUSY);
    }


    SECTION("setNodeCommDelay") {
        std::vector<int> dimensions = {2, 3};
        Grid<Packet,Node> grid(dimensions);
        grid.setNodeCommDelay({0, 0}, 1.0);
        REQUIRE(grid.getNode({0, 0}).getCommDelay() == 1.0);
    }


}

