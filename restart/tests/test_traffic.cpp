// -*- test for traffic.cpp (UserDefinedTrafficPattern) -*-

#include <iostream>
#include <catch2/catch_test_macros.hpp>
#include "traffic.hpp"
#include "config.hpp"

// TEST_CASE("Test UserDefinedTrafficPattern") {
//     //create a new configuration
//     Configuration * c = new Configuration();
//     // call the parseJSONFile method
//     c->parseJSONFile("../network/tests/config.json");
//     auto packets = c->getPackets();
//     // create a new UserDefinedTrafficPattern
//     UserDefinedTrafficPattern * u = new UserDefinedTrafficPattern(4, &packets);
//     REQUIRE(u->check_user_defined() == true);
//     REQUIRE(u->dest(0) == 1);
//     REQUIRE(u->dest(1) == -1);
//     REQUIRE(u->id() == 0);
//     REQUIRE(u->type() == 2);
//     REQUIRE(u->size() == 1000);
//     u->next();
//     REQUIRE(u->_reached_end == false);
//     REQUIRE(u->id() == 1);
//     REQUIRE(u->type() == 1);
//     REQUIRE(u->size() == 500);
//     REQUIRE(u->dest(2) == 3);
//     u->next();
//     REQUIRE(u->_reached_end == true);
// }



