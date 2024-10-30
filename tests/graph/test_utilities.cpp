#include "utilities.hpp"
#include <catch2/catch_test_macros.hpp>
#include <set>

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
        std::vector<int> dimensions = {3, 3};
        Grid<Packet, Node> grid(dimensions);
        REQUIRE(grid.getNode({0, 0}).getId() == 0);
        REQUIRE(grid.getNode({0, 1}).getType() == NodeType::EDGE);
        REQUIRE(grid.getNode({1, 2}).getId() == 5);
        REQUIRE(grid.getNode({1, 2}).getType() == NodeType::EDGE);
        REQUIRE(grid.getNode({1, 1}).getType() == NodeType::INTERNAL);
        REQUIRE(grid.getNode({2, 2}).getType() == NodeType::EDGE);

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

TEST_CASE("Test Topology"){
    SECTION("Constructor and basic getters") {
        std::vector<int> dimensions = {4, 4};
        Grid<Packet,Node> grid(dimensions);
        TopologyType type(TopologyType::MESH);
        Topology<Packet,Node> topo(type,grid);
        REQUIRE(topo.getType()==TopologyType::MESH);
        REQUIRE(topo.getGrid().getDimensions().size() == 2);
    }

    SECTION("get Adjacency Matrix") {
        std::vector<int> dimensions = {4, 4};
        Grid<Packet,Node> grid(dimensions);
        TopologyType type(TopologyType::MESH);
        Topology<Packet,Node> topo(type,grid);
        std::vector<std::vector<int>> adjMat = topo.getAdjMat();
        REQUIRE(adjMat.size() == 16);
        REQUIRE(adjMat[0].size() == 2);
        REQUIRE(std::set<int>(adjMat[0].begin(), adjMat[0].end()) == std::set<int>({1, 4}));
        REQUIRE(std::set<int>(adjMat[5].begin(), adjMat[5].end()) == std::set<int>({1, 4, 6, 9}));
    }

    SECTION("reimposeOnGrid") {
        std::vector<int> dimensions = {4, 4};
        Grid<Packet,Node> grid(dimensions);
        TopologyType type(TopologyType::MESH);
        Topology<Packet,Node> topo(type,grid);
        std::vector<int> dimensions2 = {3, 3};
        Grid<Packet,Node> grid2(dimensions2);
        topo.reimposeOnGrid(grid2);
        REQUIRE(topo.getGrid().getDimensions().size() == 2);
        REQUIRE(topo.getGrid().getDimensions()[0] == 3);
        REQUIRE(topo.getGrid().getDimensions()[1] == 3);
        REQUIRE(topo.getAdjMat().size() == 9);
    }

    SECTION("setType") {
        std::vector<int> dimensions = {4, 4};
        Grid<Packet,Node> grid(dimensions);
        TopologyType type(TopologyType::MESH);
        Topology<Packet,Node> topo(type,grid);
        topo.setType(TopologyType::TORUS);
        REQUIRE(topo.getType() == TopologyType::TORUS);
    }

    SECTION("computeAdjacientNodes 1"){
        std::vector<int> dimensions = {4, 4};
        Grid<Packet,Node> grid(dimensions);
        TopologyType type(TopologyType::MESH);
        Topology<Packet,Node> topo(type,grid);
        std::vector<int> ad_nodes = topo.computeAdjacentNodes(std::vector({1,0}));
        REQUIRE(std::set<int>(ad_nodes.begin(), ad_nodes.end()) == std::set<int>({0,5,8}));
    }

    

}

