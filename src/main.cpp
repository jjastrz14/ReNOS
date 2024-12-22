// $Id$

/*
 Copyright (c) 2007-2015, Trustees of The Leland Stanford Junior University
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 Redistributions of source code must retain the above copyright notice, this 
 list of conditions and the following disclaimer.
 Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*main.cpp
 *
 *The starting point of the network simulator
 *-Include all network header files
 *-initilize the network
 *-initialize the traffic manager and set it to run
 *
 *
 */
#include <sys/time.h>

#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>


// bindings for pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include "globals.hpp"
#include "routefunc.hpp"
#include "traffic.hpp"
#include "booksim_config.hpp"
#include "trafficmanager.hpp"
#include "random_utils.hpp"
#include "network.hpp"
#include "injection.hpp"
#include "power_module.hpp"



///////////////////////////////////////////////////////////////////////////////
// Definitions for the functions defined in globals.hpp
//////////////////////




int GetSimTime(const SimulationContext* context) {
  return context->trafficManager->getTime();
}

class Stats;
Stats * GetStats(const std::string & name, const SimulationContext* context) {
  Stats* test =  context->trafficManager->getStats(name);
  if(test == 0){
    cout<<"warning statistics "<<name<<" not found"<<endl;
  }
  return test;
}


 
/////////////////////////////////////////////////////////////////////////////

int Simulate( BookSimConfig const & config, SimulationContext & context, tRoutingParameters & par ) {
  
  vector<Network *> net;

  int subnets = config.getIntField("subnets");
  /*To include a new network, must register the network here
   *add an else if statement with the name of the network
   */
  net.resize(subnets);
  for (int i = 0; i < subnets; ++i) {
    ostringstream name;
    name << "network_" << i;
    net[i] = Network::New( config, context, par, name.str());
  }

  /*tcc and characterize are legacy
   *not sure how to use them 
   */

  
  TrafficManager * trafficManager = TrafficManager::New( config, net, context, par );


  // Assign the traffic manager to the routing parameters class for timing purposes
  context.setTrafficManager(trafficManager);

  /*Start the simulation run
   */ 

  double total_time; /* Amount of time we've run */
  struct timeval start_time, end_time; /* Time before/after user code */
  total_time = 0.0;
  gettimeofday(&start_time, NULL);

  int result;
  if(config.getIntField("user_defined_traffic") > 0){
    result = trafficManager->RunUserDefined();

  }else{
    result = trafficManager->Run();
  }


  gettimeofday(&end_time, NULL);
  total_time = ((double)(end_time.tv_sec) + (double)(end_time.tv_usec)/1000000.0)
            - ((double)(start_time.tv_sec) + (double)(start_time.tv_usec)/1000000.0);

  *(context.gDumpFile) << "Total simulation time: " << total_time << " seconds" << endl;

  for (int i=0; i<subnets; ++i) {

    ///Power analysis
    if(config.getIntField("sim_power") > 0){
      Power_Module pnet(net[i], config);
      pnet.run();
    }

    delete net[i];
  }

  delete trafficManager;
  trafficManager = NULL;

  return result;
}

/////////////////////////////////////////////////////////////////////////////
// Create a wrapper for the simulator, in order to expose it to Python

namespace py = pybind11;

int SimulateWrapper(const std::string &config_file, const std::string &output_file) {

    // Generate a simulation context
    SimulationContext context;

    // define the dump file (name passed as argument)
    if (output_file == "") {
        context.gDumpFile = &context.nullStream;;
    } else if (output_file == "-") {
        context.gDumpFile = &std::cout;
    } else {
        context.gDumpFile = new std::ofstream(output_file.c_str());
    }

    // release GIL
    py::gil_scoped_release release;

    BookSimConfig config;
    char *argv[] = {const_cast<char *>("nocsim"), const_cast<char *>(config_file.c_str())};
    int argc = 2; // name of the program + name of the config file

    if (!ParseArgs(&config, context, argc, argv)) {
        throw std::runtime_error("Failed to parse configuration file");
    }

    tRoutingParameters p=initializeRoutingMap(config);

    context.gPrintActivity = (config.getIntField("print_activity") > 0);
    context.gTrace = (config.getIntField("viewer_trace") > 0);

    std::string watch_out_file = config.getStrField("watch_out");
    if (watch_out_file == "") {
        context.gWatchOut = &context.nullStream;
    } else if (watch_out_file == "-") {
        context.gWatchOut = &std::cout;
    } else {
        context.gWatchOut = new std::ofstream(watch_out_file.c_str());
    }


    // For now, just return the latency of the simulation
    int result = Simulate(config, context, p);

    // Re-acquire GIL
    py::gil_scoped_acquire acquire;

    return result;
}

PYBIND11_MODULE(nocsim, m) {
    m.doc() = "Network-on-Chip simulator";
    m.def("simulate", &SimulateWrapper, "Run the simulation with the given configuration file");
}


/////////////////////////////////////////////////////////////////////////////

int main( int argc, char **argv )
{

  SimulationContext context;

  context.gDumpFile = &cout;
  //context.gDumpFile = &context.nullStream;

  BookSimConfig config;

  if ( !ParseArgs( &config, context, argc, argv ) ) {
    cerr << "Usage: " << argv[0] << " configfile... [param=value...]" << endl;
    return 0;
 } 

  
  /*initialize routing, traffic, injection functions
   */
  tRoutingParameters p=initializeRoutingMap(config);

  context.gPrintActivity = (config.getIntField("print_activity") > 0);
  context.gTrace = (config.getIntField("viewer_trace") > 0);
  
  string watch_out_file = config.getStrField( "watch_out" );
  if(watch_out_file == "") {
    context.gWatchOut = NULL;
  } else if(watch_out_file == "-") {
    context.gWatchOut = &cout;
  } else {
    context.gWatchOut = new ofstream(watch_out_file.c_str());
  }

  /*configure and run the simulator
   */
  cout << "Simulation context: " << &context << endl;
  int result = Simulate(config, context, p);
  return result >0 ? 0 : -1;
}
