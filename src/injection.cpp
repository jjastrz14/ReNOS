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

bool operator==(const std::pair<int, const ComputingWorkload *> &lhs, const std::pair<int, const ComputingWorkload *> &rhs) {
    return lhs.first == rhs.first && lhs.second == rhs.second;
}

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

InjectionProcess * InjectionProcess::NewUserDefined(string const & inject, int nodes, int local_memory_size, double reconfig_cycles,  int reconf_batch_size, int flit_size, Clock * clock, TrafficPattern * traffic, vector<set<tuple<int,int,int>>> * landed_packets, const SimulationContext * context,
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
    result = new DependentInjectionProcess(nodes, local_memory_size, reconfig_cycles, reconf_batch_size, flit_size, clock, traffic,landed_packets, context, resort);
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

DependentInjectionProcess::DependentInjectionProcess(int nodes, int local_mem_size, double reconfig_cycles, int reconf_batch_size, int flit_size ,Clock * clock, TrafficPattern * traffic, vector<set<tuple<int,int,int>>> * landed_packets, const SimulationContext * context,int resort) 
  : InjectionProcess(nodes, 0.0), _traffic(traffic), _memory_set(nodes,local_mem_size), _processed(landed_packets), _clock(clock), _logger(context->logger), _context(context)
{
  assert(_traffic);
  assert(_processed);
  assert(_clock);
  assert(context);
  _enable_reconf = local_mem_size > 0 ? true : false;
  _reconf_batch_size = reconf_batch_size;
  _nvm = new NVMPar(reconfig_cycles);

  _executed.resize(nodes);
  _timer.resize(nodes, 0);
  _decur.resize(nodes, false);
  _reconf_active.resize(nodes, false);
  _reconf_ready.resize(nodes, false);
  _waiting_packets.resize(nodes);
  _waiting_workloads.resize(nodes);
  _pending_packets.resize(nodes);
  _requiring_ouput_deallocation.resize(nodes);
  _output_left_to_deallocate.resize(nodes);
  _reconf_deadlock_timer.resize(nodes, 0);

  for (int i = 0; i < nodes; ++i){
    _waiting_packets[i].resize(0);
    _waiting_workloads[i].resize(0);
    _pending_packets[i].resize(0);
  }
  _pending_workloads.resize(nodes);

  _resorting = resort;
  _buildStaticWaitingQueues();

  // redefine flit size macro
  #undef FLIT_SIZE
  #define FLIT_SIZE flit_size
    
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
  
  // find the element if waiting_queue that is referenced by w
  auto it = _waiting_workloads[source].begin();
  if (it != _waiting_workloads[source].end()) assert ((*it)->id == w->id);
  while (it != _waiting_workloads[source].end()){
    w = *it;
    assert (w->size <= _memory_set.getMemoryUnit(source).getTotalAvailableForReconf());
    if (total_size + w->size <= avail_mem_for_reconf){
      
      // when a workload is allocated, we have a look in the _output_left_to_deallocate
      // we check if there are any entries related to the workload that is going to be reconfigured
      // if there are, we deallocate the output space of the workload associated to the entry
      
      // 1. if the workload has a key in _output_left_to_deallocate[source], we deallocate the outputs
      // for all te workloads in the set corresponding to the key
      auto wit = _output_left_to_deallocate[source].find(w->id);
      if (wit != _output_left_to_deallocate[source].end()){
        for (auto & ow : wit->second){
          _memory_set.deallocate_output(source, ow);
          *(_context->gDumpFile) << " DEALLOCATING OUTPUT FOR WORKLOAD " << ow->id << " to host WORKLOAD " <<  w->id << " at node " << source << " at time " << _clock->time() << std::endl;
        }
        _output_left_to_deallocate[source].erase(wit);
      }

      _memory_set.allocate(source, w);
      *(_context->gDumpFile) << "ALLOCATING WORKLOAD " << w->id << " of size " << w->size << " at node " << source << " at time " << _clock->time() << std::endl;


      total_size += w->size;
      it = std::next(it);
    }
    else{
      break;
    }
  } 
  // if we have reached the end of the queue, we set the _no_more_to_reconfigure flag to true
  if (it == _waiting_workloads[source].end()){
    _memory_set.getMemoryUnit(source).setNoMoreToReconfigure();
  }

  _memory_set.set_current_workload(source, w);
  // we return the total size of the allocated workloads
  return total_size;
}

int DependentInjectionProcess::_computeReconfMem(int source){

  int total_size = 0;
  int avail_mem_for_reconf = _memory_set.getMemoryUnit(source).getAvailableForReconf();

  // find the element if waiting_queue that is referenced by w
  auto it = _waiting_workloads[source].begin();
  if (it != _waiting_workloads[source].end()) assert ((*it)->id == _memory_set.get_current_workload(source)->id);
  while (it != _waiting_workloads[source].end()){
    const ComputingWorkload * w = *it;
    assert (w->size <= avail_mem_for_reconf);
    if (total_size + w->size <= avail_mem_for_reconf){
      total_size += w->size;
      it = std::next(it);
    }
    else{
      break;
    }
  }

  return total_size;
}


bool DependentInjectionProcess::_checkReconfNeed(int source, bool bypass_output_check){
  // a reconfiguration should take place when:
  // 1. there are still workloads in the queue that need to be loaded in memory
  // 2. there are no more workloads allocated
  // 3. all reply messages created from sending the results of the last workload have been received
  //    (that means that all the ouput placeholders have been removed)
  

  bool valid = false;
  int allocated_workloads = _memory_set.getMemoryUnit(source).getNumCurAllocatedWorkloads();
  int allocated_outputs = _memory_set.getMemoryUnit(source).getNumCurAllocatedOutputs();
  // when all of the above conditions are met, we can toggle the reconfiguration flag
  if ((bypass_output_check ? true : (allocated_outputs < 1))  &&
      allocated_workloads < 1 &&
      _waiting_workloads[source].size() > 0){    
          valid = true;
      };
  return valid;
}


void DependentInjectionProcess::stageReconfiguration(int source, bool bypass_output_check){
  // right after the output deallocation, we check for reconfiguration need:
  // if the conditions are met, we can start the reconfiguration
  if (_checkReconfNeed(source, bypass_output_check) && !_reconf_active[source]){
    // if the reconfiguration is needed, we can allocate the memory for the next workloads
    int total_size = _computeReconfMem(source);
    if (total_size == 0){
      // in this case, something went wrong: there should be enough memory to allocate at least one workload
      // if this is not the case, we have to either resize the memory, or the percentage of memory allocated for reconfiguration
      // or the size of the workloads
      *(_context->gDumpFile) << "ERROR: Not enough memory to allocate the next workload at node " << source << std::endl;
      exit(-1);
      // // increment the deadlock timer
      // _reconf_deadlock_timer[source]++; // to retrigger the reconfiguration
      // *(_context->gDumpFile) << "Reconfiguration of node " << source << " has been delayed" << std::endl;
      // if (_reconf_deadlock_timer[source] % 10 == 0){
      //   *(_context->gDumpFile) << "POSSIBLE DEADLOCK DETECTED" << std::endl;
      // }
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
      *(_context->gDumpFile) << "Reconfiguration of node " << source << " has started at time " << _clock->time() << std::endl;
    }
  }
}

void DependentInjectionProcess::stageBatchReconfiguration(int source, bool bypass_output_check){
  // this function is to be used if the _reconf_batch_size is greater than 0:
  // in this case, we force the reconfiguration to happen in batches of _reconf_batch_size: this means that 
  // the reconfiguration will be hold off and triggered in parallel on the first _reconf_batch_size nodes
  // that are ready for reconfiguration.
  // OSS: since there may happen border cases where the total number of reconfigurations is not a multiple of _reconf_batch_size
  // we will also have to check if any other reconfigurations will take place, i.e. that all the other non-ready nodes
  // have their memory _no_more_to_reconfigure flag set to true.
  // if this is the case, we can trigger the reconfiguration on the remaining nodes.
  // first, we check if the reconfiguration is needed for the source node
  if (_checkReconfNeed(source, bypass_output_check)){
    // if the reconfiguration is needed, we set to true the _reconf_ready flag
    _reconf_ready[source] = true;
  }

  // we also check that the reconfiguration is needed for other _reconf_batch_size - 1 nodes
  // if so, we can stage the reconfiguration for the first _reconf_batch_size nodes
  int count = 0;
  for (int i = 0; i < _nodes; ++i){
    if (_reconf_ready[i]){
      count++;
    }
  }
  if (count >= _reconf_batch_size){
    // we can stage the reconfiguration for the first _reconf_batch_size nodes
    for (int i = 0; i < _nodes; ++i){
      if (_reconf_ready[i] && count > 0){
        stageReconfiguration(i, bypass_output_check);
        _reconf_ready[i] = false;
        count--;
      }
    }
  }
  else{
    // if the reconfiguration is not needed for the first _reconf_batch_size nodes, we check if the remaining nodes
    // have already reconfigured all of their workloads: if all of the non-ready nodes have their memory set to _no_more_to_reconfigure
    // we can eventually trigger the reconfiguration on the remaining nodes
    bool all_ready = true;
    for (int i = 0; i < _nodes; ++i){
      if (_reconf_ready[i]){
        continue;  // if the node is ready, we skip it
      }
      else{
        // otherwise, we :
        // 1. check if the node has no more workloads to reconfigure
        // 2. check if the nodes have allocated workloads wich are dependent
        // on the output of workloads that are not yet allocated (to avoid deadlocks)
        // if either the node has still workloads to reconfigure but the dependencies are satisfied, we wait and set all_ready to false
        // if the node has no more workloads to reconfigure, we can trigger the reconfiguration, so we don't do anything


        // to check for 2, this means that all the workloads on the other nodes have been deallocated (have
        // all already written its results to the receiving nodes, but the allocated workloads on the source
        // node still don't have their dependencies satisfied)
        bool no_more_to_reconfigure = _memory_set.getMemoryUnit(i).noMoreToReconfigure();
        bool dependencies_satisfied = true;
        for (auto & w : _memory_set.getMemoryUnit(i).getCurAllocatedWorkloads()){
          if (_dependenciesWSatisfied(w,i)<0){
            dependencies_satisfied = false;
            break;
          }
        }
        if (!no_more_to_reconfigure && dependencies_satisfied ) // if the node has still workloads to reconfigure and the dependencies are satisfied
        {
          all_ready = false;
          break;
        }
      }
    }
    if (all_ready){
      for (int i = 0; i < _nodes; ++i){
        if (_reconf_ready[i]){
          stageReconfiguration(i, bypass_output_check);
        }
      }
    }
  }

}


bool DependentInjectionProcess::_manageReconfPacketInjection(const Packet * p, int p_dep_time, int source){
  // this method gets called each time a packet can be sent by the NPU.
  // it will then manage the seding of packets based on the space in the destination
  // memory.

  bool valid = false;
  const Packet * elem = nullptr;


  if (p && p_dep_time >=0){
  
    int src = p->src;
    int dst = p->dst;

    if (p->dep[0] == -1){
      return _managePacketInjection(p);
    }

    if (_memory_set.is_ready( p, _waiting_workloads[dst])){ // ALWAYS TRUE, deprecated
      
      if (src == dst){
        // in this case, there is no actual need to inject the packet
        // on the NoC, but we have to mark it as processed

        int type = p->type;
        if (type == 1 || type == 2){
          // if the packet is a write request or a read request, we change the type to the corresponding bulk message
          type = type == 1 ? 5 : 6;
        }        
        _processed->at(src).insert(make_tuple(p->id, type, _clock->time()));
        _waiting_packets[src].pop_front(); // remove the packet from the waiting queue
      *(_context->gDumpFile) << " ---- HERE Packet with ID:" << p->id <<" and type " << p->type << " at node " << p->src << " has been processed at time " << _clock->time() << std::endl;

        if (type == 6){
          // check _requiring_output_deallocation for the packet
          auto it = _requiring_ouput_deallocation[src].at(p->dep[0]).find(p->id);
          assert(it != _requiring_ouput_deallocation[src].at(p->dep[0]).end());
          
          // finalize the communication for reconfiguration
          finalizeWrite(src, p->dep[0], p->id);
          finalizeCommunication(src, p->dep[0], p->id);

        }
      }
      else{
        _traffic->cur_packet = p;
        _waiting_packets[src].pop_front(); // remove the packet from the waiting queue
        //no need to set p to pending packet, as it will be injected direcly
        valid = true;
        *(_context->gDumpFile) << "Packet with ID:" << _traffic->cur_packet->id <<" and type " << _traffic->cur_packet->type << " at node " << p->src << " has been injected at time " << _clock->time() << std::endl;
      }
    }

  }
  return valid;

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
    *(_context->gDumpFile) << " ---- HERE Packet with ID:" << p->id <<" and type " << p->type << " at node " << p->src << " has been processed at time " << _clock->time() << std::endl;

  }
  else{
    _traffic->cur_packet = p;
    _waiting_packets[p->src].pop_front(); // remove the packet from the waiting queue
    //no need to set p to pending packet, as it will be injected direcly
    valid = true;

    *(_context->gDumpFile) << "Packet with ID:" << _traffic->cur_packet->id <<" and type " << _traffic->cur_packet->type << " at node " << p->src << " has been injected at time " << _clock->time() << std::endl;
  }
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
    (*_context->gDumpFile) << "Packet " << p->id << " with size " << p->size << " inserted in queue at node " << source << std::endl;

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
    (*_context->gDumpFile) << "Workload " << w->id << " with size " << w->size << " inserted in queue at node " << source << std::endl;
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
    }

    // print the memory status
    *(_context->gDumpFile) << "MEMORY STATUS at INITIALIZATION" << std::endl;
    for (int i = 0; i < _nodes; ++i){
      *(_context->gDumpFile) <<"======================" << std::endl;
      *(_context->gDumpFile) << "Node " << i << std::endl;
      for (auto & w : _memory_set.getMemoryUnit(i).getCurAllocatedWorkloads()){
        *(_context->gDumpFile) << "Workload " << w->id << " with size " << w->size << std::endl;
      }
      *(_context->gDumpFile) <<"======================" << std::endl;
    }
    // exit(-1);
  }
}

void DependentInjectionProcess::_setProcessingTime(int node, int value)
{

  assert((node >= 0) && (node < _nodes));
  _timer[node] = value;
  // check in the _landed_packets for the packet with the id equal to the dependency
  _decur[node] = false;
}


void DependentInjectionProcess::reset()
{
  _timer.clear();
  _decur.clear();
}


void DependentInjectionProcess::addToWaitingQueue(int source, Packet * p)
{
  // starting from the first packet in the list, iterate over the packets
  // and append the new packet before the first one with 0 priority
  p->priority = 1;

  auto it = _waiting_packets[source].begin();
  while (it != _waiting_packets[source].end()){
    // first check if the packet dependecy as been cleared
    if (_dependenciesPSatisfied(*it, source) == -1 && _dependenciesPSatisfied(p, source) >= 0){
      // if the dependency for the packet in the waiting queue has not been satisfied,
      // we can insert the packet before it
      break; 
    }
    if ((*it)->priority == 0){
      break;
    }
    it = std::next(it);
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

  // check that there are still waiting packets to inject
  // if there are none, _reached_end is set to true
  int count = 0;
  for(int i = 0; i < _nodes; i++){
    if(_waiting_packets[i].empty() && _waiting_workloads[i].empty() && _pending_workloads[i] == nullptr && _pending_packets[i].size() == 0){
      count++;
    }
    if(count == _nodes  && _timer[source] == 0 && _decur[source] == false){
      reached_end = true;
    }   
  }



  bool valid = false;
  const ComputingWorkload * w = nullptr;
  const Packet * p = nullptr;


  // if there are pending workloads, the timer should be decremented
  if (_timer[source] > 0){
    assert(!(_pending_workloads[source] == nullptr) || _reconf_active[source]);
    --_timer[source];
    return valid;
  }
  else if ( _reconf_deadlock_timer[source] > 0){
    // in this case, the reconfiguration has been stage and delayed:
    // we must stage the reconfiguration again
    (_reconf_batch_size>0)? stageBatchReconfiguration(source): stageReconfiguration(source);
  }
  else if (_timer[source] == 0 && _decur[source] == true && _reconf_active[source]){
    _reconf_active[source] = false;
    _decur[source] = false;
    // actually reconfigure
    _reconfigure(source);
    *(_context->gDumpFile) << "Reconfiguration of node " << source << " has been completed at time " << _clock->time() << std::endl;
    if (_logger){
        _logger->register_event(EventType::END_RECONFIGURATION, _clock->time(), source);
    }
  } 
  else if ((_timer[source] == 0 && _decur[source] == true && !(_pending_workloads[source] == nullptr))){
    // the workload has finished computation
    if (!(_pending_workloads[source] == nullptr)){;
      _decur[source] = false;
      // logger
      if (_logger) {
        _logger->register_event(EventType::END_COMPUTATION, _clock->time(), _pending_workloads[source]->id);
      }
      *(_context->gDumpFile) << "Workload with ID:" << _pending_workloads[source]->id << " at node " << source << " has been processed at time " << _clock->time() << std::endl;
      _executed[source].insert(make_tuple(_pending_workloads[source]->id, _pending_workloads[source]->type, _clock->time()));
    }

    // enable resorting of the waiting queues for the packets
    if (_resorting || _enable_reconf){
      if (_resortWaitingPQueues(_waiting_packets, source)){
        *(_context->gDumpFile) << "Resorted the waiting packets at node " << source << std::endl;
      }
    }

    if (_enable_reconf){
      // delete from memory the workload that has been processed
      if (_pending_workloads[source] != nullptr){
        assert (_reconf_deadlock_timer[source] == 0);
        int prev_mem = _memory_set.getAvailable(source);
        _memory_set.deallocate(source, _pending_workloads[source], true);
        int new_mem = _memory_set.getAvailable(source);
        *(_context->gDumpFile) << "DEALLOCATED WORKLOAD - id : " <<  _pending_workloads[source]->id <<",  size : " << new_mem - prev_mem << " from node: " << source << " at time: " << _clock->time() << "(prev mem : " << prev_mem << ", new mem: " << new_mem << ")" << std::endl;
      }

      // if no packet is dependent on the workload, check directly for reconfiguration, bypassing the output memory check
      if (_requiring_ouput_deallocation[source].at(_pending_workloads[source]->id).size() == 0){
        stageReconfiguration(source, true); // boder case (should only happen for example runs, so in this case we don't include the batch extension)
      }

    }
    _pending_workloads[source] = nullptr;
  }


  // get the next packet and workloads in the queue
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

  // std::cout << "packet: " << p << " workload: " << w << " pending workload: " << _pending_workloads[source] << " pending packets: " << _pending_packets[source].size() << std::endl;

  
  if (p == nullptr && w == nullptr && _pending_workloads[source] == nullptr && _pending_packets[source].size() == 0){
    return valid;
  }

  assert((source >= 0) && (source < _nodes));

  int dep_time_w = -1;
  int dep_time_p = -1;
  if (!(w == nullptr)){
    dep_time_w = _dependenciesWSatisfied(&(*w), source);
  }
  if (!(p == nullptr)){
    dep_time_p = _dependenciesPSatisfied(&(*p), source); 
  }


  // the new workload/packet can be executed only if its dependecies (packets and workloads) have been satisfied
  if (dep_time_p>=0 && source == p->src && !(p == nullptr)){
    // the node is idle and can process the packet request
    assert(!_decur[source]);
    // the packet has already been cleared for the dependencies, we can inject
    valid = _enable_reconf ? _manageReconfPacketInjection(p, dep_time_p, source) : _managePacketInjection(p);
    if (valid == true){
      return valid;
    }
    
  }
  if(dep_time_w>=0 && source == w->node && !(w==nullptr) && ((_enable_reconf)?_memory_set.is_workload_allocated(source, w):1)){
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
      *(_context->gDumpFile) << "Workload with ID:" << w->id << " at node " << source << " has started processing at time " << _clock->time() << std::endl;
    }
  }

  return valid;
}

// ============================================================================================================