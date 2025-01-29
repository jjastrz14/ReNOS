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

InjectionProcess * InjectionProcess::NewUserDefined(string const & inject, int nodes, Clock * clock, TrafficPattern * traffic, const NVMPar * nvm_par, vector<set<tuple<int,int,int>>> * landed_packets, EventLogger * logger,
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
    result = new DependentInjectionProcess(nodes, clock, traffic, nvm_par ,landed_packets, resort, logger);
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

DependentInjectionProcess::DependentInjectionProcess(int nodes, Clock * clock, TrafficPattern * traffic, const NVMPar * nvm_par, vector<set<tuple<int,int,int>>> * landed_packets, int resort, EventLogger * logger) 
  : InjectionProcess(nodes, 0.0), _traffic(traffic), _nvm_par(nvm_par), _processed(landed_packets), _clock(clock), _logger(logger)
{
  assert(_traffic);
  assert(_processed);
  assert(_clock);
  _executed.resize(nodes);
  _timer.resize(nodes, 0);
  _decur.resize(nodes, false);
  _reconf_active.resize(nodes, false);
  _scheduled_reconf.resize(nodes);
  _waiting_packets.resize(nodes);
  _waiting_workloads.resize(nodes);

  for (int i = 0; i < nodes; ++i){
    _scheduled_reconf[i].resize(0);
    _waiting_packets[i].resize(0);
    _waiting_workloads[i].resize(0);
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

    // ==================  RECONFIGURATION ==================
    if (_nvm_par){
      // if the reconfiguration is enabled, for each workload appended to the waiting queues,
      // we check that the space on that node does not exceed the maximum local space
      int io_space = w->size - w->wsize; // input/output space needed by the workload
      int w_space = w->wsize; // weight space needed by the workload
      bool end = _traffic->reached_end_workloads;
      

      if (io_space > running_filled_iospace[source] + (use_freed_up_space? running_filled_wspace[source] : 0)){
        // in this case, we take into account the possibility of freeing up space occupied by the weights after
        // the corresponding task is performed:
        // This means that, when considreting the space needed for input/output by each of the workloads coming after
        // the first one, we must also consider the available space that will be freed up by the weights progressively
        // after each task 
        running_filled_iospace[source] = use_freed_up_space? io_space - running_filled_wspace[source] : io_space;
      }
      running_filled_wspace[source] += w_space;

      if (running_filled_iospace[source] + running_filled_wspace[source] > _nvm_par->get_max_pe_memory()){

        ReconfigBit rb;
        // if the space needed by the weights exceeds the maximum local space, schedule a reconfiguration
        // for after the workload before the current one
        const ComputingWorkload * cut = _waiting_workloads[source].end()[-2];
        rb.id = cut->id;
        rb.wsize = 0;
        _scheduled_reconf[source].push_back(rb);
        if (_scheduled_reconf[source].size() > 1){
          // fill the wsize field of the previous scheduled ReconfigBit
          // with the estimation of the space that will be occupied by the weights
          // of the next batch of workloads
          auto prev = _scheduled_reconf[source].end() - 2;
          (*prev).wsize = running_filled_wspace[source] - w_space;
        }
        running_filled_iospace[source] = io_space;
        running_filled_wspace[source] = w_space; 
      }
      if (_traffic->reached_end_workloads && _scheduled_reconf[source].size() != 0){
        // if the end of the workloads has been reached, fill the wsize field of the last scheduled ReconfigBit
        // with the estimation of the space that will be occupied by the weights
        // of the next batch of workloads
        auto last = _scheduled_reconf[source].end() - 1;
        std::cout << "Last scheduled reconfiguration for node " << source << " is for workload " << (*last).id << ", wsize" << (*last).wsize <<std::endl;
        (*last).wsize = running_filled_wspace[source];
      }
    }
    // ==================  RECONFIGURATION ==================

    // create a EventInfo object for each workload
    if (_logger) {
      ComputationEventInfo * cei = new ComputationEventInfo(w->id, w->node, w->ct_required);
      _logger->add_tevent_info(cei);
    }
  }

  // print the scheduled reconfigurations
  for (int i = 0; i < _nodes; ++i){
    if (_scheduled_reconf[i].size() > 0){
      std::cout << "Scheduled reconfigurations for node " << i << std::endl;
      for (auto & rb : _scheduled_reconf[i]){
        std::cout << "Workload id: " << rb.id << " wsize: " << rb.wsize << std::endl;
      }
    }
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
  } else if (_timer[source] == 0 && _decur[source] == true && !(_pending_workloads[source] == nullptr)){
    // the workload has been processed
    _decur[source] = false;
    // std::cout << "Workload at node " << source << " has finished processing at time " << _clock->time() << std::endl;
    // logger
    if (_logger) {
      _logger->register_event(EventType::END_COMPUTATION, _clock->time(), _pending_workloads[source]->id);
    }
    _executed[source].insert(make_tuple(_pending_workloads[source]->id, _pending_workloads[source]->type, _clock->time()));

    // ==================  RECONFIGURATION ==================
    if (_nvm_par) {
      
      // check if the workload is scheduled for reconfiguration
      auto match = std::find_if(_scheduled_reconf[source].begin(), _scheduled_reconf[source].end(), [this, source](const ReconfigBit & rb) {
        return rb.id == _pending_workloads[source]->id;
      });
      if (match != _scheduled_reconf[source].end()){
        // the workload is scheduled for reconfiguration
        _scheduled_reconf[source].erase(match);
        _reconf_active[source] = true;
        // compute the reconfiguration time needed based on the dimensions of the weights to be
        // pulled from the NVM
        int reconf_time = _nvm_par->cycles_reconf(_pending_workloads[source]->wsize);
        _timer[source] = reconf_time;
        _decur[source] = true;
        // std::cout << "Reconfiguration at node " << source << " has started at time " << _clock->time() << std::endl;
        // std::cout << "Reconfiguration time: " << reconf_time << std::endl;
      }
    }

    _pending_workloads[source] = nullptr;
    // ==================  RECONFIGURATION ==================
    
  } else if (_timer[source] == 0 && _decur[source] == true && _reconf_active[source]){
    // the reconfiguration has been completed
    _reconf_active[source] = false;
    // std::cout << "Reconfiguration at node " << source << " has finished at time " << _clock->time() << std::endl;
  }
  
  // the new workload/packet can be executed only if its dependecies (packets and workloads) have been satisfied
  if (dep_time_p>=0 && source == p->src && !(p == nullptr)){
    assert(_pending_workloads[source] == nullptr);
    // the node is ideal and can process the packet request
    if(_timer[source]==0){
      
      assert(!_decur[source]);
      // the packet has already been cleared for the dependencies, 
      // we can inject directly

      // fist check if src and dst are equal
      if (p->src == p->dst){
        // in this case, there is no actual need to inject the packet
        // on the NoC, but we have to mark it as processed

        int type = p->type;
        if (type == 1 || type == 2){
          // if the packet is a write request or a read request, we change the type to the corresponding bulk message
          type = type == 1 ? 5 : 6;
        }        
        _processed->at(source).insert(make_tuple(p->id, type, _clock->time()));
        _waiting_packets[source].pop_front(); // remove the packet from the waiting queue
        // std::cout << " ---- HERE Packet with ID:" << p->id <<" and type " << p->type << " at node " << source << " has been processed at time " << _clock->time() << std::endl;

      }
      else{
        _traffic->cur_packet = p;
        _waiting_packets[source].pop_front(); // remove the packet from the waiting queue
        //no need to set p to pending packet, as it will be injected direcly
        valid = true;
        // std::cout << "Packet with ID:" << _traffic->cur_packet->id <<" and type " << _traffic->cur_packet->type << " at node " << source << " has been injected at time " << _clock->time() << std::endl;
      }
      
    }
    else{
      exit(-1);
    }
  }else if(dep_time_w>=0 && source == w->node && !(w==nullptr)){
    if(_timer[source] == 0){
      // the node is idle and can process the workload
      assert(_pending_workloads[source] == nullptr);
      _pending_workloads[source] = w;
      _waiting_workloads[source].pop_front(); // remove the workload from the waiting queue
      _timer[source] = w->ct_required +1 ; // update the timer for the required time
      _decur[source] = true;
      if (_logger) {
        _logger->register_event(EventType::START_COMPUTATION, _clock->time(), w->id);
      }
      // std::cout << "Workload at node " << source << " has started processing at time " << _clock->time() << std::endl;
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

  // std::cout << "Source: " << source << " Timer: " << _timer[source] << " Decur: " << _decur[source] << " Valid: " << valid << " P. Packet: " << (_pending_packets[source] != nullptr)  << " W. Packets: " << (_waiting_packets[source].front() != nullptr)<< std::endl;
  return valid;
}

// ============================================================================================================