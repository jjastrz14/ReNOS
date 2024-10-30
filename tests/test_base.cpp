// -*- test for base.hpp -*-

#include <iostream>
#include <sstream>
#include <catch2/catch_test_macros.hpp>
#include "base.hpp"

TEST_CASE("Test base.hpp") {
    Clock c;
    c.reset();
    REQUIRE(c.time() == 0);
    c.tick();
    REQUIRE(c.time() == 1);
    c.tick();
    REQUIRE(c.time() == 2);
    c.reset();
    REQUIRE(c.time() == 0);
}

TEST_CASE("Test Module") {
    Module * m = new Module(nullptr, "m");
    REQUIRE(m->getName() == "m");
    REQUIRE(m->getFullName() == "m");
    delete m;

    Module * m1 = new Module(nullptr, "m1");
    Module * m2 = new Module(m1, "m2");

    std::stringstream ss;
    m1->printHierarchy(0, ss);
    REQUIRE(ss.str() == "m1\n  m2\n");
    delete m1;
    // erase the content of the stringstream
    ss.str(std::string());

    Module * m3 = new Module(nullptr, "m3");
    REQUIRE_THROWS(m3->error("error", 0, ss));
    REQUIRE(ss.str() == "ERROR: m3: error\n");
    ss.str(std::string());
    REQUIRE_NOTHROW(m3->printDebug("debug", ss));
    REQUIRE(ss.str() == "DEBUG: m3: debug\n");

    delete m3;
}


    