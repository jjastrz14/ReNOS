// -*- test for injection.cpp -*-

#include <stdio.h>
#include <set>
#include <catch2/catch_test_macros.hpp>
#include "injection.cpp"


// TEST_CASE("Test DependentInjectionProcess"){

//     //create a new configuration
//     Configuration * c = new Configuration();
//     Clock clock;
//     // call the parseJSONFile method
//     c->parseJSONFile("../network/tests/config.json");
//     auto packets = c->getPackets();
//     // create a new UserDefinedTrafficPattern
//     UserDefinedTrafficPattern * u = new UserDefinedTrafficPattern(4, &packets);
//     // create a set for landed packets
//     std::set<std::pair<int,int>> landed_packets;
//     DependentInjectionProcess * injection = new DependentInjectionProcess(4,&clock,u, &landed_packets);

//     clock.tick();
//     REQUIRE(injection->test(1)== false );
//     clock.tick();
//     REQUIRE(injection->test(3)== false );
//     clock.tick();
//     REQUIRE(injection->test(0)== false );

//     clock.tick();
//     injection->decurPTime(0);
//     for(int i = 0; i<8 ; i++){
//         clock.tick();
//         injection->test(0);
//     }

//     clock.tick();
//     REQUIRE(injection->test(0)== false); 


//     for(int i = 0; i <10 ; i++){
//         // the idea of decurPTime is that is should de
//         // decreasing the process time counter while at the same time
//         clock.tick();
//         injection->decurPTime(0);
//     }
//     clock.tick();
//     REQUIRE(injection->test(0)== true); 
//     clock.tick();
//     REQUIRE(injection->test(0)== false);
//     clock.tick();
//     REQUIRE(injection->test(2) == false);

//     // crease the counter by 3 more, then try test again
//     for(int i = 0; i <3 ; i++){
//         clock.tick();
//         injection->decurPTime(2);
//     }
    
//     clock.tick();
//     REQUIRE(injection->test(2)== false);
//     // append the first packet to the landed packets
//     landed_packets.insert(std::make_pair(0, clock.time()));
//     clock.tick();
//     REQUIRE(injection->test(2)== false);
//     for (int i = 0; i < 7; i++){
//         clock.tick();
//         injection->decurPTime(2);
//     }
//     clock.tick();
//     REQUIRE(injection->test(2)== true);

// }