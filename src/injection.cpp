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

#include <iostream>
#include <vector>
#include <cassert>
#include <limits>
#include "random_utils.hpp"
#include "injection.hpp"

using namespace std;

InjectionProcess::InjectionProcess(int nodes, double rate)
  : _nodes(nodes), _rate(rate), reached_end(false)
{
  if(nodes <= 0) {
    cout << "Error: Number of nodes must be greater than zero." << endl;
    exit(-1);
  }
  if((rate < 0.0) || (rate > 1.0)) {
    cout << "Error: Injection process must have load between 0.0 and 1.0."
	 << endl;
    exit(-1);
  }
}

void InjectionProcess::reset()
{

}

InjectionProcess * InjectionProcess::New(string const & inject, int nodes, 
					 double load,
					 Configuration const * const config)
{
  string process_name;
  string param_str;
  size_t left = inject.find_first_of('(');
  if(left == string::npos) {
    process_name = inject;
  } else {
    process_name = inject.substr(0, left);
    size_t right = inject.find_last_of(')');
    if(right == string::npos) {
      param_str = inject.substr(left+1);
    } else {
      param_str = inject.substr(left+1, right-left-1);
    }
  }
  vector<string> params = tokenize_str(param_str);

  InjectionProcess * result = NULL;
  if(process_name == "bernoulli") {
    result = new BernoulliInjectionProcess(nodes, load);
  } else if(process_name == "on_off") {
    bool missing_params = false;
    double alpha = numeric_limits<double>::quiet_NaN();
    if(params.size() < 1) {
      if(config) {
	alpha = config->getFloatField("burst_alpha");
      } else {
	missing_params = true;
      }
    } else {
      alpha = atof(params[0].c_str());
    }
    double beta = numeric_limits<double>::quiet_NaN();
    if(params.size() < 2) {
      if(config) {
	beta = config->getFloatField("burst_beta");
      } else {
	missing_params = true;
      }
    } else {
      beta = atof(params[1].c_str());
    }
    double r1 = numeric_limits<double>::quiet_NaN();
    if(params.size() < 3) {
      r1 = config ? config->getFloatField("burst_r1") : -1.0;
    } else {
      r1 = atof(params[2].c_str());
    }
    if(missing_params) {
      cout << "Missing parameters for injection process: " << inject << endl;
      exit(-1);
    }
    if((alpha < 0.0 && beta < 0.0) || 
       (alpha < 0.0 && r1 < 0.0) || 
       (beta < 0.0 && r1 < 0.0) || 
       (alpha >= 0.0 && beta >= 0.0 && r1 >= 0.0)) {
      cout << "Invalid parameters for injection process: " << inject << endl;
      exit(-1);
    }
    vector<int> initial(nodes);
    if(params.size() > 3) {
      initial = tokenize_int(params[2]);
      initial.resize(nodes, initial.back());
    } else {
      for(int n = 0; n < nodes; ++n) {
	initial[n] = randomInt(1);
      }
    }
    result = new OnOffInjectionProcess(nodes, load, alpha, beta, r1, initial);
  } else {
    cout << "Invalid injection process: " << inject << endl;
    exit(-1);
  }
  return result;
}

InjectionProcess * InjectionProcess::NewUserDefined(string const & inject, int nodes, Clock * clock, TrafficPattern * traffic,  set<pair<int,int>> * landed_packets,
        Configuration const * const config)
{
  string process_name;
  string param_str;
  size_t left = inject.find_first_of('(');
  if(left == string::npos) {
    process_name = inject;
  } else {
    process_name = inject.substr(0, left);
    size_t right = inject.find_last_of(')');
    if(right == string::npos) {
      param_str = inject.substr(left+1);
    } else {
      param_str = inject.substr(left+1, right-left-1);
    }
  }
  vector<string> params = tokenize_str(param_str);

  InjectionProcess * result = NULL;
  if(process_name == "dependent") {
    result = new DependentInjectionProcess(nodes, clock, traffic, landed_packets);
  } else {
    cout << "Invalid injection process: " << inject << endl;
    exit(-1);
  }
  return result;
}

//=============================================================

BernoulliInjectionProcess::BernoulliInjectionProcess(int nodes, double rate)
  : InjectionProcess(nodes, rate)
{

}

bool BernoulliInjectionProcess::test(int source)
{
  assert((source >= 0) && (source < _nodes));
  return (randomFloat() < _rate);
}

//=============================================================

OnOffInjectionProcess::OnOffInjectionProcess(int nodes, double rate, 
					     double alpha, double beta, 
					     double r1, vector<int> initial)
  : InjectionProcess(nodes, rate), 
    _alpha(alpha), _beta(beta), _r1(r1), _initial(initial)
{
  assert(alpha <= 1.0);
  assert(beta <= 1.0);
  assert(r1 <= 1.0);
  if(alpha < 0.0) {
    assert(beta >= 0.0);
    assert(r1 >= 0.0);
    _alpha = beta * rate / (r1 - rate);
  } else if(beta < 0.0) {
    assert(alpha >= 0.0);
    assert(r1 >= 0.0);
    _beta = alpha * (r1 - rate) / rate;
  } else {
    assert(r1 < 0.0);
    _r1 = rate * (alpha + beta) / alpha;
  }
  reset();
}

void OnOffInjectionProcess::reset()
{
  _state = _initial;
}

bool OnOffInjectionProcess::test(int source)
{
  assert((source >= 0) && (source < _nodes));

  // advance state
  _state[source] = 
    _state[source] ? (randomFloat() >= _beta) : (randomFloat() < _alpha);

  // generate packet
  return _state[source] && (randomFloat() < _r1);
}



// ============================================================================================================

DependentInjectionProcess::DependentInjectionProcess(int nodes, Clock * clock, TrafficPattern * traffic , set<pair<int,int>> * landed_packets)
  : InjectionProcess(nodes, 0.0), _traffic(traffic), _landed_packets(landed_packets), _clock(clock)
{
  assert(_traffic);
  assert(_landed_packets);
  assert(_clock);
  _processing_time.resize(nodes, 0);
  _decurring.resize(nodes, false);
  _waiting_packets.resize(nodes);
  _buildWaitingQueues();
  //reset();
}

void DependentInjectionProcess::reset()
{
  _processing_time.clear();
  _decurring.clear();
}

void DependentInjectionProcess::_setProcessingTime(int node, int value)
{

  assert((node >= 0) && (node < _nodes));
  _processing_time[node] = value;
  // check in the _landed_packets for the packet with the id equal to the dependency
  _decurring[node] = true;
}

int DependentInjectionProcess::_dependenciesSatisfied(const Packet * p) const
{
  int last_dependecy = p->dep;
  if (p->dep == -1)
    return 0;
  else{
    // check in landed packets if the (last) dependency for the packet has been satisfied
    auto it = std::find_if(_landed_packets->begin(), _landed_packets->end(), [last_dependecy](const pair<int,int> & p){
      return p.first == last_dependecy;
    });
    // if the search find a dependency in the set, it means it has already landed, so 
    // return the time at which it has landed
    return (it != _landed_packets->end()) ? it->second : -1;
  }
  
}

// the method is to be called within the constructor
void DependentInjectionProcess::_buildWaitingQueues(){

  // starting from the first packet in the list, loop over the packets
  // and build the waiting queues for each source node
  _traffic->reset();
  _waiting_packets.clear();

  // STUPID COMPILING OPTION, MAY DECIDE TO UPGRADE LATER
  while(_traffic->reached_end == false){
    const Packet * p = _traffic->getNext();
    assert(p);
    assert((p->size > 0));
    assert((p->src >= 0) && (p->src < _nodes));
    int source = p->src;
    _waiting_packets[source].push_back(p);
    _traffic->updateNext();
  }

  _traffic->reset();
}

void DependentInjectionProcess::decurPTime(int source)
{
  //
}

bool DependentInjectionProcess::test(int source)
{
  bool valid = false;
  const Packet * p = nullptr;

  // check in the _waiting_packets if there are any packets for the source node
  // currently considered
  if(!_waiting_packets[source].empty()){
    p = _waiting_packets[source].front();
  }
  else{
    return valid;
  }

  assert(p);
  assert((source >= 0) && (source < _nodes));
  assert((p->size > 0));

  // check if the dependencies have been satisfied for the packet
  int dep_time = _dependenciesSatisfied(&(*p));
  if(dep_time>=0 && source == p->src){
    // check if the processing time has elapsed and the time for that source node is decurring
    // -> the node with dependencies has finisched processing it so we can 
    //    send the dependent packet ( NEED CHECK THAT THE DESTINATION NODE
    //    OF THE DEPENDECY IS THE SOURCE NODE OF THE CURRENT PACKET)
    if(_processing_time[source] == 0 && _decurring[source]){
      // this is possible only if there are waiting packets still to inject
      assert(_waiting_packets[source].front() != nullptr);

      _decurring[source] = false; // reset the decurring flag for further packets
      _traffic->updateCurPacket(p); // set the current packet to the one that has been waiting
      _waiting_packets[source].pop_front(); // remove the packet from the waiting queue
      
      valid = true;
    }
    // if the processing time has not elapsed, decrement the time
    // -> THE SOURCE NODE NEEDS TO WAIT FOR THE PROCESSING
    //    OF THE DEPENDENCY TO BE FINISHED BEFORE SENDING THE PACKET
    else if(_processing_time[source] > 0){
      --_processing_time[source];
    }
    // both the processing time is 0 and the decurring flag is false,
    // -> THE SOURCE NODE IS IDLE, SO IT CAN TAKE THE NEXT PACKET
    else if(_processing_time[source]==0 && !_decurring[source]){
      std::cout << "======================================================================================================" << std::endl;
      std::cout << "Setting processing time for packet " << p->id << " at node " << source << " and time " << _clock->time() << std::endl;
      
      if (p->dep == -1){
        // no need to wait for dependencies, injection can start directly
        _traffic->updateCurPacket(p);
        _waiting_packets[source].pop_front(); // remove the packet from the waiting queue
        valid = true;
      }
      else{
        // find the processing time of the dependence
        auto bpacket = _traffic->packetByID(p->dep);
        //set the processing time also taking into account the amount of time already passed
        int new_time = bpacket->pt_required - (_clock->time()-dep_time);
        new_time = new_time < 0 ? 0 : new_time;
        if(new_time == 0){
          // no need to wait for dependencies, injection can start directly
          _traffic->updateCurPacket(p);
          _waiting_packets[source].pop_front(); // remove the packet from the waiting queue
          valid = true;
        }else{
          // set the processing time for the source node
          _setProcessingTime(source, new_time);

          std::cout << "PROCESSING TIME SET TO " << new_time << std::endl;
        }
      }
      std::cout << "======================================================================================================" << std::endl;
    }
    else{
      exit(-1);
    }
  }

  
  // check that there are still waiting packets to inject
  // if there are none, _reached_end is set to true
  int count = 0;
  for(int i = 0; i < _nodes; i++){
    if(_waiting_packets[i].empty()){
      count++;
    }
    if(count == _nodes){
      reached_end = true;
    }   
  }

  return valid;
}

// ============================================================================================================