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


commType intToCommType(int i)
{
  switch(i) {
    case 1: return READ_REQ;
    case 2: return WRITE_REQ;
    case 3: return READ_ACK; 
    case 4: return WRITE_ACK; 
    case 5: return READ;
    case 6: return WRITE;
    case 0: return ANY; 
    default: throw std::invalid_argument("Invalid integer value for commType");
  }
};

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

InjectionProcess * InjectionProcess::NewUserDefined(string const & inject, int nodes, int local_memory_size, int reconfig_cycles, Clock * clock, TrafficPattern * traffic, vector<set<tuple<int,int,int>>> * landed_packets, EventLogger * logger,
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
    int resort = config ? config->getIntField("resort_waiting_queues") : 0;
    // std::cout << "Resort waiting queues: " << resort << std::endl;
    // std::cout << "Nodes: " << nodes << std::endl;
    // std::cout << "Clock: " << clock << std::endl;
    // std::cout << "Traffic: " << traffic << std::endl;
    // std::cout << "Landed packets: " << landed_packets << std::endl;
    result = new DependentInjectionProcess(nodes, local_memory_size, reconfig_cycles, clock, traffic,landed_packets, resort, logger);
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

DependentInjectionProcess::DependentInjectionProcess(int nodes, int local_mem_size, int reconfig_cycles ,Clock * clock, TrafficPattern * traffic, vector<set<tuple<int,int,int>>> * landed_packets, int resort, EventLogger * logger) 
  : InjectionProcess(nodes, 0.0), _traffic(traffic), _memory_set(nodes,local_mem_size), _processed(landed_packets), _clock(clock), _logger(logger)
{
  assert(_traffic);
  assert(_processed);
  assert(_clock);
  _enable_reconf = local_mem_size > 0 ? true : false;
  _nvm = new NVMPar(reconfig_cycles);
  _executed.resize(nodes);
  _timer.resize(nodes, 0);
  _decur.resize(nodes, false);
  _reconf_active.resize(nodes, false);
  _waiting_packets.resize(nodes);
  _waiting_workloads.resize(nodes);
  _pending_packets.resize(nodes);
  _requiring_ouput_deallocation.resize(nodes);
  _reconf_deadlock_timer.resize(nodes, 0);

  for (int i = 0; i < nodes; ++i){
    _waiting_packets[i].resize(0);
    _waiting_workloads[i].resize(0);
    _pending_packets[i].resize(0);
  }
  _pending_workloads.resize(nodes);

  _resorting = resort;
  _buildStaticWaitingQueues();
    
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

int DependentInjectionProcess::_reconfigure(int source){
  
  // starting from the first valid workload, we allocate the memory of the next ones in the queue
  // until the memory is full
  const ComputingWorkload * w;
  w = _memory_set.get_current_workload(source);
  if (w == nullptr){
    assert (_waiting_workloads[source].size() == 0);
  }
  int total_size = 0;
  int avail_mem_for_reconf = _memory_set.getMemoryUnit(source).getAvailableForReconf();
  
  // find the element if waiting_queue tha is referenced by w
  auto it = std::find_if(_waiting_workloads[source].begin(), _waiting_workloads[source].end(), [w](const ComputingWorkload * p){
    return p->id == w->id;
  });
  while (it != _waiting_workloads[source].end()){
    w = *it;
    if (total_size + w->size <= avail_mem_for_reconf){
      _memory_set.allocate(source, w);
      total_size += w->size;
      it = std::next(it);
    }
    else{
      break;
    }
  }
  _memory_set.set_current_workload(source, w);
  // we return the total size of the allocated workloads
  return total_size;

}


bool DependentInjectionProcess::_checkReconfNeed(int source){
  // a reconfiguration should take place when:
  // 1. there are still workloads in the queue that need to be loaded in memory
  // 2. there is just one workload with allocated space in memory
  // 3. the aforementioned workload has been processed

  bool valid = false;
  // when all of the above conditions are met, we can toggle the reconfiguration flag
  if (_waiting_workloads[source].size() > 0 && 
      _memory_set.get_current_workload(source) != *_waiting_workloads[source].end() &&
      _memory_set.getMemoryUnit(source).getNumCurAllocatedWorkloads() == 1){
        auto it = std::find_if(_executed[source].begin(), _executed[source].end(), [this,source](const tuple<int,int,int> & p){
          return std::get<0>(p) == this->_memory_set.getMemoryUnit(source).getCurAllocatedWorkloads()[0]->id;
        }); 

        if (it != _executed[source].end()){
          valid = true;
        }
      };
  return valid;
}


// the method is to be called within the constructor
void DependentInjectionProcess::_buildStaticWaitingQueues(){

  // starting from the first packet in the list, loop over the packets
  // and build the waiting queues for each source node
  _traffic->reset();

  // ==================  RECONFIGURATION ==================
  // buffers to compute the scheduled reconfigurations
  std::vector<int> running_filled_iospace(_nodes, 0); // for input/output
  std::vector<int> running_filled_wspace(_nodes, 0); // for weights
  bool use_freed_up_space = false; // a flag to enable the use of the freed up space
  // ==================  RECONFIGURATION ==================
  

  if (_logger) {
    _logger->initialize_event_info(_traffic->getPacketsSize() + _traffic->getWorkloadsSize());
  }

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
    // make space for the packet
    _waiting_packets[source].emplace_back(p);
    _traffic->updateNextPacket();

    // create a EventInfo object for each packet
    if (_logger) {
      commType ctype = intToCommType(p->type);
      TrafficEventInfo * tei = new TrafficEventInfo(p->id, ctype, p->src, p->dst, p->size);
      _logger->add_tevent_info(tei);
    }
      
  }

  while(_traffic->reached_end_workloads == false){
    const ComputingWorkload * w = _traffic->getNextWorkload();
    assert(w);
    assert((w->ct_required > 0));
    assert((w->node >= 0) && (w->node < _nodes));
    int source = w->node;
    _waiting_workloads[source].emplace_back(w);
    _traffic->updateNextWorkload();

    // create a EventInfo object for each workload
    if (_logger) {
      ComputationEventInfo * cei = new ComputationEventInfo(w->id, w->node, w->ct_required);
      _logger->add_tevent_info(cei);
    }
  }

  _traffic->reset();
  _memory_set.init(_waiting_workloads);

  if (_enable_reconf){

    // for each node
    for (int i = 0; i < _nodes; ++i){
      // for each workload in waiting, we create a new
      // entry in the _requiring_output_deallocation data structure
      for (auto & w : _waiting_workloads[i]){
        _requiring_ouput_deallocation[i].insert(std::make_pair(w->id, std::set<int>()));
      }

      
      // for each packet in waiting, if that packet depends on a workload
      // we add the packet to the set of packets that will require the deallocation
      // of the output space of the workload
      for (auto & p : _waiting_packets[i]){
        for (auto & dep : p->dep){
          if (dep != -1){
              _requiring_ouput_deallocation[i].at(dep).insert(p->id);
            }
            
        }
      }
    }
    
    // load the first workloads in memory
    for (int i = 0; i < _nodes; ++i){
      _reconfigure(i); // for the first recofiguration, we don't care about the 
      // time the reconfiguration takes: we are assuming the workload are preloaded
      if(_memory_set.getMemoryUnit(i).getNumCurAllocatedWorkloads() > 0){
        for (auto & w : _memory_set.getMemoryUnit(i).getCurAllocatedWorkloads()){
        }
      }
    }

  }
}


bool DependentInjectionProcess::_managePacketInjection(const Packet * p){
  bool valid = false;
  if (p->src == p->dst){
    // in this case, there is no actual need to inject the packet
    // on the NoC, but we have to mark it as processed

    int type = p->type;
    if (type == 1 || type == 2){
      // if the packet is a write request or a read request, we change the type to the corresponding bulk message
      type = type == 1 ? 5 : 6;
    }        
    _processed->at(p->src).insert(make_tuple(p->id, type, _clock->time()));
    _waiting_packets[p->src].pop_front(); // remove the packet from the waiting queue
    // std::cout << " ---- HERE Packet with ID:" << p->id <<" and type " << p->type << " at node " << source << " has been processed at time " << _clock->time() << std::endl;

  }
  else{
    _traffic->cur_packet = p;
    _waiting_packets[p->src].pop_front(); // remove the packet from the waiting queue
    //no need to set p to pending packet, as it will be injected direcly
    valid = true;

    // std::cout << "Packet with ID:" << _traffic->cur_packet->id <<" and type " << _traffic->cur_packet->type << " at node " << source << " has been injected at time " << _clock->time() << std::endl;
  }
  return valid;

}


bool DependentInjectionProcess::_manageReconfPacketInjection(const Packet * p){
  // this method gets called each time a packet can be sent by the NPU.
  // it will then manage the seding of packets based on the space in the destination
  // memory.

  bool valid = false;
  std::tuple<const ComputingWorkload *, const Packet *> * elem = nullptr;

  // first, we check for pending packets that needs to be injected: we loop over them
  // and check if the memory has been allocated for the corresponding workload
  // if so, we can inject the packet: this takes precedence over the packets in the waiting queue
  for (auto it = _pending_packets[p->src].begin(); it != _pending_packets[p->src].end(); ++it){
    if (_memory_set.already_allocated(std::get<0>(*it), std::get<1>(*it), _waiting_workloads[std::get<1>(*it)->dst]))
    {
      // the packet can be sent directly to the destination
      // without need to wait
      // fist check if src and dst are equal
      elem = &(*it);
      const Packet *packet = std::get<1>(*it);
      if (packet->src == packet->dst){
        // in this case, there is no actual need to inject the packet
        // on the NoC, but we have to mark it as processed
        
        int type = p->type;
        if (type == 1 || type == 2){
          // if the packet is a write request or a read request, we change the type to the corresponding bulk message
          type = type == 1 ? 5 : 6;
        }       
        _processed->at(packet->src).insert(make_tuple(packet->id, packet->type, _clock->time()));


        if (type == 6){
          int prev_mem = _memory_set.getAvailable(packet->src);
          _memory_set.deallocate(packet->src, std::get<0>(*it), true);
          int new_mem = _memory_set.getAvailable(packet->src);
          // check _requiring_output_deallocation for the packet
          auto it = _requiring_ouput_deallocation[packet->src].at(packet->dep[0]).find(packet->id);
          assert(it != _requiring_ouput_deallocation[packet->src].at(packet->dep[0]).end());
          // remove the packet from the set
          _requiring_ouput_deallocation[packet->src].at(packet->dep[0]).erase(it);
          // and check if the set for the dependency is empty: in this case, we can deallocate the memory
          // for the output
          if (_requiring_ouput_deallocation[packet->src].at(packet->dep[0]).size() == 0){
            int prev_mem = _memory_set.getAvailable(packet->src);
            auto w = _traffic->workloadByID(packet->dep[0]);
            _memory_set.deallocate_output(packet->src, w);
            int new_mem = _memory_set.getAvailable(packet->src);
          }

        }
        
      }
      else{
        _traffic->cur_packet = packet;
        valid = true;
        // no need to set p to pending packet, as it will be injected direcly.
        // in this case, the result of the workload computation needs to be stored in the memory as long as
        // the tail of the packet leaves the node
        // the space worth for the output of the workload will be deallocated when the packet received is 
        // the write reply to the packet sent
        if (packet->type == 6){
          int prev_mem = _memory_set.getAvailable(packet->src);
          _memory_set.deallocate(packet->src, std::get<0>(*it), true);
          int new_mem = _memory_set.getAvailable(packet->src);
        }
      }

      break;
    }
  }
  if (elem != nullptr){
    // remove the packet from the pending packets queue and return
    _pending_packets[p->src].erase(std::find(_pending_packets[p->src].begin(), _pending_packets[p->src].end(), *elem));
    return valid;
  }

  // if there are no eligible pending packets, we can check the waiting queue:
  // check if the the packet can be sent directly to the destination
  int src = p->src;
  int dst = p->dst;

  if (p->dep[0] == -1){
    return _managePacketInjection(p);
  }

  const ComputingWorkload * associated_workload = nullptr;
  associated_workload = _traffic->workloadByID(p->dep[0]);

  if (_memory_set.already_allocated(associated_workload, p, _waiting_workloads[dst])){
    // the packet can be sent directly to the destination
    // without need to wait
    // fist check if src and dst are equal
    if (p->src == p->dst){
      // in this case, there is no actual need to inject the packet
      // on the NoC, but we have to mark it as processed

      int type = p->type;
      if (type == 1 || type == 2){
        // if the packet is a write request or a read request, we change the type to the corresponding bulk message
        type = type == 1 ? 5 : 6;
      }        
      _processed->at(src).insert(make_tuple(p->id, type, _clock->time()));
      _waiting_packets[src].pop_front(); // remove the packet from the waiting queue

      // in this case, the output of the previous layer is alredy presen on the same node where the input for the next layer
      // has already been allocated, so we can dirctly deallocate the memory worth for the entire workload
      if (type == 6){
        int prev_mem = _memory_set.getAvailable(src);
        _memory_set.deallocate(src, associated_workload, true);
        int new_mem = _memory_set.getAvailable(src);
        // check _requiring_output_deallocation for the packet
        auto it = _requiring_ouput_deallocation[src].at(p->dep[0]).find(p->id);
        assert(it != _requiring_ouput_deallocation[src].at(p->dep[0]).end());
        // remove the packet from the set
        _requiring_ouput_deallocation[src].at(p->dep[0]).erase(it);
        // and check if the set for the dependency is empty: in this case, we can deallocate the memory
        // for the output
        if (_requiring_ouput_deallocation[src].at(p->dep[0]).size() == 0){
          int prev_mem = _memory_set.getAvailable(src);
          _memory_set.deallocate_output(src, associated_workload);
          int new_mem = _memory_set.getAvailable(src);
          
        }
      }

    }
    else{
      _traffic->cur_packet = p;
      _waiting_packets[src].pop_front(); // remove the packet from the waiting queue
      //no need to set p to pending packet, as it will be injected direcly
      valid = true;
      // in this case, the result of the workload computation needs to be stored in the memory as long as
      // the tail of the packet leaves the node
      // the space worth for the output of the workload will be deallocated when the packet received is 
      // the write reply to the packet sent
      if (p->type == 6){
        int prev_mem = _memory_set.getAvailable(src);
        _memory_set.deallocate(src, associated_workload, true);
        int new_mem = _memory_set.getAvailable(src);
      }
    }
  }
  else{
    // in case the space has not been allocated yet, we have preserve the result in 
    // the memory until the packer can be injected and append the packet to the pending packets
    _pending_packets[src].emplace_back(make_tuple(associated_workload, p));

  };

  return valid;

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

  if (_logger) {
    commType ctype = intToCommType(p->type);
    TrafficEventInfo * tei = new TrafficEventInfo(p->id, ctype, p->src, p->dst, p->size);
    _logger->add_tevent_info(tei);
  }

}

bool DependentInjectionProcess::test(int source)
{
  bool valid = false;
  const ComputingWorkload * w = nullptr;
  const Packet * p = nullptr;

  if (_resorting){
    if (_resortWaitingQueues(_waiting_packets, source)){
    // std::cout << "Resorted the waiting packets" << std::endl;
    }
    if (_resortWaitingQueues(_waiting_workloads, source)){
      // std::cout << "Resorted the waiting workloads" << std::endl;
    }
  }
  
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

  
  if (p == nullptr && w == nullptr && _pending_workloads[source] == nullptr){
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
    assert(!(_pending_workloads[source] == nullptr) || _reconf_active[source]);
    --_timer[source];
    return valid;
  }
  else if (_timer[source] == 0 && _decur[source] == true && _reconf_active[source]){
    _reconf_active[source] = false;
    _decur[source] = false;
    if (_logger){
        _logger->register_event(EventType::END_RECONFIGURATION, _clock->time(), source);
    }
  } 
  else if ((_timer[source] == 0 && _decur[source] == true && !(_pending_workloads[source] == nullptr)) || _reconf_deadlock_timer[source] > 0){
    // the workload has finished computation
    if (!(_pending_workloads[source] == nullptr)){;
      _decur[source] = false;
      // logger
      if (_logger) {
        _logger->register_event(EventType::END_COMPUTATION, _clock->time(), _pending_workloads[source]->id);
      }
      _executed[source].insert(make_tuple(_pending_workloads[source]->id, _pending_workloads[source]->type, _clock->time()));
    }
    
    // check for reconfiguration need
    if (_enable_reconf){
      bool ready_for_reconf = _checkReconfNeed(source);
      if (ready_for_reconf){
        // if the reconfiguration is needed, we can allocate the memory for the next workloads
        int total_size = _reconfigure(source);
        if (total_size == 0){
          // increment the deadlock timer
          _reconf_deadlock_timer[source]++; // to retrigger the reconfiguration
        }
        else{
          int time_needed = _nvm->cycles_reconf(total_size);
          _decur[source] = true;
          _reconf_active[source] = true;
          _reconf_deadlock_timer[source] = 0;
          _timer[source] = time_needed;
          if (_logger) {
            _logger->register_event(EventType::START_RECONFIGURATION, _clock->time(), source);
          }
        }
        
      }
    }
    _pending_workloads[source] = nullptr;
  }

  
  // the new workload/packet can be executed only if its dependecies (packets and workloads) have been satisfied
  if (dep_time_p>=0 && source == p->src && !(p == nullptr)){
    
    assert(_pending_workloads[source] == nullptr);
    // the node is ideal and can process the packet request
    if(_timer[source]==0){
      assert(!_decur[source]);
      // the packet has already been cleared for the dependencies, we can inject
      valid = _enable_reconf ? _manageReconfPacketInjection(p) : _managePacketInjection(p);
    }
    else{
      exit(-1);
    }
  }else if(dep_time_w>=0 && source == w->node && !(w==nullptr) && ((_enable_reconf)?_memory_set.is_allocated(source, w):1)){
    if(_timer[source] == 0){
      // the node is idle and can process the workload
      assert(_pending_workloads[source] == nullptr);

      _pending_workloads[source] = w;
      _waiting_workloads[source].pop_front(); // remove the workload from the waiting queue
      _timer[source] = w->ct_required ; // update the timer for the required time
      _decur[source] = true;
      if (_logger) {
        _logger->register_event(EventType::START_COMPUTATION, _clock->time(), w->id);
      }
    }
  }

  // check that there are still waiting packets to inject
  // if there are none, _reached_end is set to true
  int count = 0;
  for(int i = 0; i < _nodes; i++){
    if(_waiting_packets[i].empty() && _waiting_workloads[i].empty() && _pending_workloads[i] == nullptr){ // && _pending_packets[i] == nullptr
      count++;
    }
    if(count == _nodes  && _timer[source] == 0 && _decur[source] == false){
      reached_end = true;
    }   
  }
  return valid;
}

// ============================================================================================================