// -*- test for channel.cpp -*-

#include <iostream>
#include <catch2/catch_test_macros.hpp>
#include "channel.hpp"

// TEST_CASE("Test FlitChannel") {
//     FlitChannel * fc = new FlitChannel(nullptr, "FlitChannel", 1);
//     fc->setLatency(2);
//     REQUIRE(fc->getLatency() == 2);
//     auto f = new Flit(0, commType::ANY, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, true, 0);
//     fc->send(f);
//     fc->readInputs();
//     std::cout << GetSimTime() << std::endl;
//     Clock::tick();
//     std::cout << GetSimTime() << std::endl;
//     fc->writeOutputs();
//     REQUIRE(fc->receive()->id == 0);
//     delete fc;
// }

// TEST_CASE("Test CreditChannel") {
//     CreditChannel * cc = new CreditChannel(nullptr, "CreditChannel");
//     cc->setLatency(1);
//     REQUIRE(cc->getLatency() == 1);
//     cc->send(new Credit());
//     cc->readInputs();
//     cc->writeOutputs();
//     REQUIRE(cc->receive()->id == -1);
//     delete cc;
// }