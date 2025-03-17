// -*- test for outset.cpp -*-

#include <iostream>
#include <catch2/catch_test_macros.hpp>
#include "outset.hpp"

TEST_CASE("Test OutSet") {
    OutSet * o = new OutSet();
    REQUIRE(o->getOutSet().size() == 0);
    o->addRange(0, 0, 3, 0);
    o->addRange(1, 0, 3, 1);
    o->addRange(0, 4, 7, 4);
    o->addRange(2, 0, 7, 2);
    o->add(4, 0, 3);
    REQUIRE(o->empty() == false);
    REQUIRE(o->outputEmpty(0) == false);
    REQUIRE(o->outputEmpty(1) == false);
    REQUIRE(o->outputEmpty(3) == true);
    REQUIRE(o->numOutputVCs(0) == 8);
    REQUIRE(o->numOutputVCs(1) == 4);
    REQUIRE(o->getVC(0, 0, nullptr) == 4);
    REQUIRE(o->getVC(0, 6, nullptr) == 2);
    int * port = new int;
    int * vc = new int;
    REQUIRE(o->getPortVC(port,vc) == false);

    OutSet * o2 = new OutSet();
    o2->add(0, 0, 0);
    REQUIRE(o2->getPortVC(port,vc) == true);
}