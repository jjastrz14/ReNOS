#include "utilities.hpp"
#include "router.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test NodeInfo") {
    NodeInfo nodeInfo;
    nodeInfo.g = 1.0;
    nodeInfo.h = 2.0;
    std::vector<int> parent_index = {1, 2};
    nodeInfo.parent_index = parent_index;
    REQUIRE(nodeInfo.f() == 3.0);
}


TEST_CASE("Test Router") {
    SECTION("Constructor 1"){
        std::vector<int> dimensions = {2, 3};
        std::vector<int> in_point = {0, 0};
        Router<Packet, Node> router(dimensions, TopologyType::TORUS, in_point);
        REQUIRE(router.getTopology().getDimensions().size() == 2);
        REQUIRE(router.getTopology().getDimensions()[0] == 2);
        REQUIRE(router.getTopology().getDimensions()[1] == 3);
        REQUIRE(router.getInPoint().size() == 2);
        REQUIRE(router.getInPoint()[0] == 0);
        REQUIRE(router.getInPoint()[1] == 0);
    }
    SECTION("Constructor 2") {
        std::vector<int> dimensions = {2, 3};
        Topology<Packet, Node> topo(TopologyType::MESH, dimensions);
        std::vector<int> in_point = {0, 0};
        Router<Packet, Node> router(topo, in_point);
        REQUIRE(router.getTopology().getDimensions().size() == 2);
        REQUIRE(router.getTopology().getDimensions()[0] == 2);
        REQUIRE(router.getTopology().getDimensions()[1] == 3);
        REQUIRE(router.getInPoint().size() == 2);
        REQUIRE(router.getInPoint()[0] == 0);
        REQUIRE(router.getInPoint()[1] == 0);
    }

    SECTION("getTopology") {
        std::vector<int> dimensions = {2, 3};
        std::vector<int> in_point = {0, 0};
        Router<Packet, Node> router(dimensions, TopologyType::TORUS, in_point);
        REQUIRE(router.getTopology().getDimensions().size() == 2);
        REQUIRE(router.getTopology().getDimensions()[0] == 2);
        REQUIRE(router.getTopology().getDimensions()[1] == 3);
    }

    SECTION("getInPoint") {
        std::vector<int> dimensions = {2, 3};
        std::vector<int> in_point = {0, 0};
        Router<Packet, Node> router(dimensions, TopologyType::TORUS, in_point);
        REQUIRE(router.getInPoint().size() == 2);
        REQUIRE(router.getInPoint()[0] == 0);
        REQUIRE(router.getInPoint()[1] == 0);
    }

    SECTION("setTopology") {
        std::vector<int> dimensions = {2, 3};
        std::vector<int> in_point = {0, 0};
        Router<Packet, Node> router(dimensions, TopologyType::TORUS, in_point);
        std::vector<int> dimensions2 = {3, 4};
        Topology<Packet, Node> topo(TopologyType::MESH, dimensions2);
        router.setTopology(topo);
        REQUIRE(router.getTopology().getDimensions().size() == 2);
        REQUIRE(router.getTopology().getDimensions()[0] == 3);
        REQUIRE(router.getTopology().getDimensions()[1] == 4);
    }

    SECTION("setInPoint") {
        std::vector<int> dimensions = {2, 3};
        std::vector<int> in_point = {0, 0};
        Router<Packet, Node> router(dimensions, TopologyType::TORUS, in_point);
        std::vector<int> in_point2 = {1, 1};
        router.setInPoint(in_point2);
        REQUIRE(router.getInPoint().size() == 2);
        REQUIRE(router.getInPoint()[0] == 1);
        REQUIRE(router.getInPoint()[1] == 1);
    }

    SECTION("tracePath") {
        std::vector<int> dimensions = {2, 3};
        std::vector<int> in_point = {0, 0};
        Router<Packet, Node> router(dimensions, TopologyType::TORUS, in_point);
        std::vector<NodeInfo> node_info = {NodeInfo(), NodeInfo(), NodeInfo(), NodeInfo(), NodeInfo(), NodeInfo()};
        node_info[0].parent_index = {0, 1};
        node_info[1].parent_index = {1, 1};
        node_info[2].parent_index = {1, 2};
        node_info[3].parent_index = {0, 2};
        node_info[4].parent_index = {0, 1};
        node_info[5].parent_index = {0, 0};
        std::vector<int> start = {0, 0};
        std::vector<int> end = {1, 2};
        std::vector<int> path = router.tracePath(start, end, node_info);
        REQUIRE(path.size() == 2);
        REQUIRE(path[0] == 0);
        REQUIRE(path[1] == 5);

        
    }

    SECTION("findPathAstar") {
        std::vector<int> dimensions = {2, 3};
        std::vector<int> in_point = {0, 0};
        Router<Packet, Node> router(dimensions, TopologyType::TORUS, in_point);
        std::vector<int> start = {0, 0};
        std::vector<int> end = {0, 2};
        std::vector<int> path = router.findPathAstar(start, end);
        REQUIRE(path.size() == 2);
        REQUIRE(path[0] == 0);
        REQUIRE(path[1] == 2);

        std::vector<int> dimensions2 = {3,4};
        std::vector<int> in_point2 = {1, 1};
        Router<Packet, Node> router2(dimensions2, TopologyType::MESH, in_point2);
        std::vector<int> start2 = {0, 0};
        std::vector<int> end2 = {2, 3};
        // set some of the nodes to busy
        router2.getTopology().setNodeStatus({0, 0}, StatusType::BUSY);
        REQUIRE(router2.getTopology().getNode({0, 0}).getStatus() == StatusType::BUSY);
        REQUIRE(isBusyNode({0, 0}, router2.getTopology()) == true);
        router2.getTopology().setNodeStatus({1, 0}, StatusType::BUSY);
        REQUIRE_THROWS(router2.findPathAstar(start2, end2));

        start2 = {0, 0};
        end2 = {2, 1};
        router2.getTopology().setNodeStatus({0, 0}, StatusType::IDLE);
        router2.getTopology().setNodeStatus({1, 1}, StatusType::BUSY);
        router2.getTopology().setNodeStatus({1, 0}, StatusType::BUSY);
        router2.getTopology().setNodeStatus({0, 1}, StatusType::BUSY); 
        REQUIRE_THROWS(router2.findPathAstar(start2, end2));
        
    }

}