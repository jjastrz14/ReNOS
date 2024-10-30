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

#ifndef _INJECTION_HPP_
#define _INJECTION_HPP_

#include "config.hpp"
#include "base.hpp"
#include "traffic.hpp"

using namespace std;

class InjectionProcess {
protected:
  int _nodes;
  double _rate;
  InjectionProcess(int nodes, double rate);
public:
  virtual ~InjectionProcess() {}
  virtual bool test(int source) = 0;
  virtual void reset();
  bool reached_end;
  static InjectionProcess * New(string const & inject, int nodes, double load,
				Configuration const * const config = NULL);
  static InjectionProcess * NewUserDefined(string const & inject, int nodes, Clock * clock, TrafficPattern * traffic,  set<pair<int,int>> * landed_packets,
        Configuration const * const config = NULL);

};

class BernoulliInjectionProcess : public InjectionProcess {
public:
  BernoulliInjectionProcess(int nodes, double rate);
  virtual bool test(int source);
};

class OnOffInjectionProcess : public InjectionProcess {
private:
  double _alpha;
  double _beta;
  double _r1;
  vector<int> _initial;
  vector<int> _state;
public:
  OnOffInjectionProcess(int nodes, double rate, double alpha, double beta, 
			double r1, vector<int> initial);
  virtual void reset();
  virtual bool test(int source);
};

// ------ additions to support user defined injection processes ------

class DependentInjectionProcess : public InjectionProcess {
  private:
    // pointer to the sets of landed packets, i.e. the packets that have reached their destination, and the time 
    // at which they have been received
    set<pair<int,int>> * _landed_packets;
    // vector of timers (and decurring flags) to keep track of the time elapsed since all the dependencies have been satisfied
    // to simulate the processing time needed to generate all the data that will be injected in the next packet
    vector<int> _processing_time;
    vector<bool> _decurring;
    // the vector hosting the packets that are waiting for their dependencies to be processed
    vector<deque<const Packet *>> _waiting_packets;
    // a pointer to the clock of traffic manager
    Clock * _clock;
    // finally, a pointer to the traffic object, holding the packets to be injected
    TrafficPattern * _traffic;
    
    // a method to check if the dependencies have been satisfied for the packet
    int _dependenciesSatisfied (const Packet * p) const;
    // a method to build the waiting queues
    void _buildWaitingQueues();
    // a method to set the clock to a specific value
    void _setProcessingTime(int node, int value);

  public:
    DependentInjectionProcess(int nodes,Clock * clock, TrafficPattern * traffic , set<pair<int,int>> * landed_packets);
    virtual void reset();
    // method to model elapsing processing time to call when test() is not current control flow
    void decurPTime(int source);
    // finally, a method to test if the packet can be injected
    virtual bool test(int source);
    
};

#endif 
