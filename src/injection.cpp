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

InjectionProcess * InjectionProcess::NewUserDefined(string const & inject, int nodes, Clock * clock, TrafficPattern * traffic,  vector<set<tuple<int,int,int>>> * landed_packets,
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

DependentInjectionProcess::DependentInjectionProcess(int nodes, Clock * clock, TrafficPattern * traffic , vector<set<tuple<int,int,int>>> * landed_packets)
  : InjectionProcess(nodes, 0.0), _traffic(traffic), _processed(landed_packets), _clock(clock)
{
  assert(_traffic);
  assert(_processed);
  assert(_clock);
  _executed.resize(nodes);
  _timer.resize(nodes, 0);
  _decur.resize(nodes, false);
  _waiting_packets.resize(nodes);
  for (int i = 0; i < nodes; ++i){
    _waiting_packets[i].resize(0);
  }
  _pending_packets.resize(nodes);
  _waiting_workloads.resize(nodes);
  for (int i = 0; i < nodes; ++i){
    _waiting_workloads[i].resize(0);
  }
  _pending_workloads.resize(nodes);
  _buildStaticWaitingQueues();
  //reset();
}

void DependentInjectionProcess::reset()
{
  _timer.clear();
  _decur.clear();
}

void DependentInjectionProcess::_setProcessingTime(int node, int value)
{

  assert((node >= 0) && (node < _nodes));
  _timer[node] = value;
  // check in the _landed_packets for the packet with the id equal to the dependency
  _decur[node] = false;
}


// the method is to be called within the constructor
void DependentInjectionProcess::_buildStaticWaitingQueues(){

  // starting from the first packet in the list, loop over the packets
  // and build the waiting queues for each source node
  _traffic->reset();
  _waiting_packets.clear();
  _waiting_workloads.clear();
  _pending_packets.clear();

  // STUPID COMPILING OPTION, MAY DECIDE TO UPGRADE LATER
  if (_traffic->getNextPacket() == nullptr){
    _traffic->reached_end_packets = true;
  }
  if (_traffic->getNextWorkload() == nullptr){
    _traffic->reached_end_workloads = true;

  }

  while(_traffic->reached_end_packets == false){
    const Packet * p = _traffic->getNextPacket();
    assert(p);
    assert((p->size > 0));
    assert((p->src >= 0) && (p->src < _nodes));
    int source = p->src;
    _waiting_packets[source].push_back(p);
    _traffic->updateNextPacket();
  }

  while(_traffic->reached_end_workloads == false){
    const ComputingWorkload * w = _traffic->getNextWorkload();
    assert(w);
    assert((w->ct_required > 0));
    assert((w->node >= 0) && (w->node < _nodes));
    int source = w->node;
    _waiting_workloads[source].push_back(w);
    _traffic->updateNextWorkload();
  }

  _traffic->reset();
}


void DependentInjectionProcess::addToWaitingQueue(int source, Packet * p)
{
  // starting from the first packet in the list, iterate over the packets
  // and append the new packet before the first one with 0 priority
  p->priority = 1;

  auto it = _waiting_packets[source].begin();
  while (it != _waiting_packets[source].end() && (*it)->priority != 0) {
    //check that the packet is not a dependency for the packets already in the queue
    ++it;
  }

  if (it != _waiting_packets[source].end() && (*it)->priority != 0){
    it++;
  }

  _waiting_packets[source].insert(it, p); // INSERT AFTER

}

bool DependentInjectionProcess::test(int source)
{
  bool valid = false;
  const ComputingWorkload * w = nullptr;
  const Packet * p = nullptr;

  if(!(_waiting_packets[source].empty())){
    p = _waiting_packets[source].front();
    assert(p);
    assert((p->size > 0));
    assert((p->src == source));
  }
  if (!(_waiting_workloads[source].empty())){
    w =_waiting_workloads[source].front();
    assert(w);
    assert((w->ct_required > 0));
    assert(w->node == source);
  }

  
  if (p == nullptr && w == nullptr && _pending_packets[source] == nullptr && _pending_workloads[source] == nullptr){
    return valid;
  }

  assert((source >= 0) && (source < _nodes));

  int dep_time_w = -1;
  int dep_time_p = -1;
  if (!(w == nullptr)){
    dep_time_w = _dependenciesSatisfied(&(*w), source);
  }
  if (!(p == nullptr)){
    dep_time_p = _dependenciesSatisfied(&(*p), source);
  }

  // if there are pending workloads, the timer should be decremented
  if (_timer[source] > 0){
    assert(!(_pending_workloads[source] == nullptr));
    --_timer[source];
    return valid;
  } else if (_timer[source] == 0 && _decur[source] == true && !(_pending_workloads[source] == nullptr)){
    // the workload has been processed
    _executed[source].insert(make_tuple(_pending_workloads[source]->id, _pending_workloads[source]->type, _clock->time()));
    _pending_workloads[source] = nullptr;
    _decur[source] = false;
    std::cout << "Workload at node " << source << " has finished processing at time " << _clock->time() << std::endl;
  }
  // last case: node is idle
  else if (!(_pending_packets[source] == nullptr)){
    assert(_pending_workloads[source] == nullptr);
    std::cout << "HERE is the problem" << std::endl;
    _traffic->cur_packet = _pending_packets[source];
    _pending_packets[source] = nullptr;
    valid = true;
    _decur[source] = true;
    std::cout << "Packet with ID:" << _traffic->cur_packet->id <<" and type " << _traffic->cur_packet->type << " at node " << source << " has been injected at time " << _clock->time() << std::endl;
  }
  
  // the new workload can be executed only if its dependecies (packets and workloads) have been satisfied
  if(dep_time_w>=0 && source == w->node && !(w==nullptr)){
    if(_timer[source] == 0){
      // the node is idle and can process the workload
      assert(_pending_workloads[source] == nullptr);
      if (_decur[source] == false) assert(_pending_packets[source] == nullptr);
      _pending_workloads[source] = w;
      _waiting_workloads[source].pop_front(); // remove the workload from the waiting queue
      _timer[source] =(_decur[source] == false ) ? w->ct_required - 1 : w->ct_required ; // update the timer for the required time
      _decur[source] = true;
      std::cout << "Workload at node " << source << " has started processing at time " << _clock->time() << std::endl;
    }
  }else if (dep_time_p>=0 && source == p->src && !(p == nullptr)){
    assert(_pending_workloads[source] == nullptr);
    assert(_pending_packets[source] == nullptr);
    // 1. a packet request has been serviced in the current cycle
    std::cout << "HERE" << std::endl;
    if(_timer[source] == 0 && _decur[source]){
      _pending_packets[source] = p; // the new pending packet is the one currently considerd
      _waiting_packets[source].pop_front(); // remove the packet from the waiting queue

    }
    // 2. the node is idel and can process the packet request
    else if(_timer[source]==0 && !_decur[source]){

      // the packet has already been cleared for the dependencies, 
      // we can inject directly --> bypass _pending_packets
      _traffic->cur_packet = p;
      _waiting_packets[source].pop_front(); // remove the packet from the waiting queue
      //no need to set p to pending packet, as it will be injected direcly
      valid = true;
      std::cout << "Packet with ID:" << _traffic->cur_packet->id <<" and type " << _traffic->cur_packet->type << " at node " << source << " has been injected at time " << _clock->time() << std::endl;
    }
    else{
      exit(-1);
    }
  }

  // check that there are still waiting packets to inject
  // if there are none, _reached_end is set to true
  int count = 0;
  for(int i = 0; i < _nodes; i++){
    if(_waiting_packets[i].empty() && _waiting_workloads[i].empty() && _pending_packets[i] == nullptr && _pending_workloads[i] == nullptr){
      count++;
    }
    if(count == _nodes  && _timer[source] == 0 && _decur[source] == false){
      reached_end = true;
    }   
  }

  std::cout << "Source: " << source << " Timer: " << _timer[source] << " Decur: " << _decur[source] << " Valid: " << valid << " P. Packet: " << (_pending_packets[source] != nullptr)  << " W. Packets: " << (_waiting_packets[source].front() != nullptr)<< std::endl;
  return valid;
}

// ============================================================================================================