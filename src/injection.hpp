
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
#include "logger.hpp"

using namespace std;

commType intToCommType(int i);

class InjectionProcess {
protected:
  int _nodes;
  double _rate;
  InjectionProcess(int nodes, double rate);
public:
  virtual ~InjectionProcess() {}
  virtual bool test(int source) = 0;
  virtual bool isIdle(int source) { return false; }
  virtual void addToWaitingQueue(int source, Packet * p){};
  virtual int getAvailableMemory(int source){return 0;};
  virtual void stageReconfiguration(int source, bool bypass_empty_memory_check = false) {};
  virtual void removeFromRequiringOutputDeallocation(int source, int id, int packet_id){}
  virtual bool isRequiringOutputDeallocationEmpty(int source, int id){return false;}
  virtual void deallocateMemory(int source, const ComputingWorkload * w){};
  virtual void reset();
  bool reached_end;
  static InjectionProcess * New(string const & inject, int nodes, double load,
				Configuration const * const config = NULL);
  static InjectionProcess * NewUserDefined(string const & inject, int nodes, int local_memory_size, int reconfig_cycles, Clock * clock, TrafficPattern * traffic, vector<set<tuple<int,int,int>>> * landed_packets, const SimulationContext * context,
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

  struct ReconfigBit {
    /* 
    The struct will be used to schedule an online reconfiguration (if reconfiguraitons are enabled).
    The information needed for the reconfigurations are:
    - the id of the task after which to perform the reconfiguration
    - the size of the weights to be pulled from the NVM using TSVs: this information
      is then used to estimate the number of reconfiguration cycles needed to actually perform 
      the reconfiguration during the simulation
      */
    int id; // the id of the task after which to perform reconfiguration
    int wsize; // the size of the weights to be pulled from the NVM
  };

  class MemoryUnit {
    // The following class will be used to model the memory unit of the PE.
    // For initialization, the class will be given an integer, representing the size (in bytes/flit) of the local 
    // memory unit available to the single NPU. It provides a very high level abstraction of the memory unit.
    public:
      MemoryUnit(int size, float threshold = 0.9) : _size(size), _available(size), _threshold(threshold) {}
      int getSize() const { return _size; }
      int getAvailable() const { return _available; }
      int getTotalAvailableForReconf() const { return _size*_threshold; }
      int getAvailableForReconf() const { return _threshold * _size - (_size-_available); }
      int getNumCurAllocatedWorkloads() const { return _cur_allocated_workload.size(); }
      std::deque<const ComputingWorkload *> & getCurAllocatedWorkloads() { return _cur_allocated_workload; }

      void reset() { 
        _available = _size; 
        for (auto & w : _cur_allocated_workload){
          delete w;
        }
        _cur_allocated_workload.clear();
      }

      bool check_mem_avail(int size){
        /*
        the method will be used to check if the memory is available
        if the memory is available, the method will return true
        and the available memory will be decreased by the size of the memory to be allocated.
        If the memory is not available, the method will return false
        */
        if (size <= _available){
          return true;
        }
        return false;
      }

      bool is_allocated(const ComputingWorkload * w){
        /*
        the method is used to check if the workload is already allocated in the memory
        */
        auto it = std::find(_cur_allocated_workload.begin(), _cur_allocated_workload.end(), w);
        return it != _cur_allocated_workload.end();
      }

      bool is_in_next_reconfig_batch(const ComputingWorkload * w, const std::deque<const ComputingWorkload *> & waiting_workloads){
        /*
        the method is used to check if the workload is in the next reconfiguration batch:
        */
        int total_size = 0;
        int avail_mem_for_reconf = getAvailableForReconf();
        auto it = waiting_workloads.begin();
        while (it != waiting_workloads.end()){
          const ComputingWorkload * n = *it;
          if (total_size + n->size <= avail_mem_for_reconf){
            total_size += n->size;
            it = std::next(it);
            if (n->id == w->id){
              return true;
            }
          }
          else{
            break;
          }
        }
        return false;
      }

      bool is_ready(const Packet * p, const std::deque<const ComputingWorkload *> & waiting_workloads){
        /*
        the method is used to check if the destionation already has loaded the corresponding
        partition that will use the result of w, passed by p, as its input. In this case, the packet
        can be sent direcly to the destination. Otherwise, the result of the computation will have to 
        be stored in the memory of the destination 
        */

      
        // CASE 1. check if the workload is already allocated
        bool ready = false;
        for (auto & ow : _cur_allocated_workload){
          // check if the workload is dependent on the current workload via the packet
          auto it = std::find(ow->dep.begin(), ow->dep.end(), p->id);
          if (it != ow->dep.end()){
            ready = true;
            return ready;
          }
        }

        
        for (auto & w : waiting_workloads){
          auto it = std::find(w->dep.begin(), w->dep.end(), p->id);

          // CASE 2. if the packet is addressed to a partition that is scheduled to be reconfigured
          // right after on the same node, we can mark the packet as ready to be sent (DEADLOCK AVOIDANCE: reconfiguration
          // is performed when the memory is empty, this means that also the output of previous workloads that
          // is waiting to be sent needs to be purged, by actually sending the packet. If the packet and next workload
          // are addressed to the same node, this created a deadlock situation)
          if (it != w->dep.end() && w->node == p->dst){
            if (is_in_next_reconfig_batch(w, waiting_workloads)){
              ready = true;
              return ready;
            }
            else{
              std::cout << "DEADLOCK AVOIDANCE: before leaving, the packet needs the receiving workload to be reconfigured.\n This can not happen because this is not scheduled in the next reconfiguration batch." << std::endl;
              exit(-1);
            }
          }

          // CASE 3. we must also check that a depending workload is actually going to be scheduled
          // on the PE at a certain point
          if (it != w->dep.end()){
            ready = true;
          }
        }
        return !ready;
      }


      bool is_empty(){
        return _available == _size;
      }

      bool allocate_(int size){
        /*
        the method will be used to allocate the memory not direcly assocated
        with a workload but needed to store the results of the computations
        coming from others NPUs. If the memory is available, the method will return true
        and the available memory will be decreased by the size of the memory to be allocated.
        If the memory is not available, the method will return false
        */
        assert(size <= _size && size <= _available);
        if (size <= _available){
          _available -= size;
          return true;
        }
        return false;

      }

      bool deallocate_(int size){
        /*
        the method will be used to deallocate the memory not direcly assocated
        with a workload but needed to store the results of the computations
        coming from others NPUs. If the memory is found in the list of allocated memory,
        the memory will be removed from the list and the available memory will be increased by the size of the memory. In this case,
        the method will return true. If the memory is not found in the list of allocated memory, the method will return false
        */
        assert(size <= _size && _available + size <= _size);
        if (_available + size <= _size){
          _available += size;
          return true;
        }
        return false;
      }

      bool allocate(const ComputingWorkload * w, bool optimize = false){
        /*
        the method will be used to allocate the memory for the workload
        if the memory is available, the workload will be added to the list of allocated workloads
        and the available memory will be decreased by the size of the workload. In this case,
        the method will return true. If the memory is not available, the method will return false

        Additionally, if the optimize flag is set to true, the method will try to optimize memory allocation
        and this will be taken into account in the available memory calculation: specifically, if optimize is set to 
        true, the method will snoop at the input_range of the workload and compare it with the input_range 
        of the already allocated workloads. If some input overlapping is detected
        the method will subtract the overlapping size from the size of memory that needs to be allocated
         for the workload and then proceed with the allocation.
        */


        int size_to_allocate = w->size;
        int scale = FLIT_SIZE;

        if (optimize){
          // check for overlapping in the input space
          int in_overlap = 0;
          for (auto & ow : _cur_allocated_workload){
            if ( w->layer == ow->layer){
              int in_overlap_with_workload = 0;
              in_overlap_with_workload = in_overlap_in(w, ow);
              assert(in_overlap_with_workload > 0);
              in_overlap += in_overlap_with_workload;
            }
          }
          // subtract the overlapping space from the size of the workload
          size_to_allocate -= in_overlap;
        }

        if (size_to_allocate <= _available){
          _cur_allocated_workload.push_back(w);
          return allocate_(size_to_allocate);
        }
        return false;
      };

      bool deallocate_output(const ComputingWorkload * w){
        /*
        The method will be used to deallocate the memory for the output of the workload
        if the workload is found in the list of allocated workloads, the workload will be removed
        from the list and the available memory will be increased by the size of the workload. In this case,
        the method will return true. If the workload is not found in the list of allocated workloads, the method will return false.
        */

        int output_size = 1;
        for (int i = 0; i < w->output_range.lower_sp.size(); ++i){
          output_size *= (w->output_range.upper_sp[i] - w->output_range.lower_sp[i]);
        }
        if (w->output_range.channels.size() > 0)
          output_size *= (w->output_range.channels[1] - w->output_range.channels[0]);
        int scale = FLIT_SIZE;
        output_size /= scale;
        output_size = output_size > 0 ? output_size : 1;

        return deallocate_(output_size);
      };

      bool deallocate(const ComputingWorkload * w, bool keep_output, bool optimize = false){
        /*
        The method will be used to deallocate the memory for the workload
        if the workload is found in the list of allocated workloads, the workload will be removed
        from the list and the available memory will be increased by the size of the workload. In this case,
        the method will return true. If the workload is not found in the list of allocated workloads, the method will return false.
        */

       // first, we check if any other waiting workload is 

        int output_size = 1;
        for (int i = 0; i < w->output_range.lower_sp.size(); ++i){
          output_size *= (w->output_range.upper_sp[i] - w->output_range.lower_sp[i]);
        }
        if (w->output_range.channels.size() > 0)
          output_size *= (w->output_range.channels[1] - w->output_range.channels[0]);
        int scale = FLIT_SIZE;
        output_size /= scale;
        output_size = output_size > 0 ? output_size : 1;


        int size_to_deallocate = w->size - output_size;
        
        auto it = std::find(_cur_allocated_workload.begin(), _cur_allocated_workload.end(), w);
        if (it != _cur_allocated_workload.end()){
        // Additionally, if the optimize flag is set to true, the method will optimize memory deallocation
        // i.e. it simply avoid to deallocate the memory that is shared with other workloads. This is done by checking
        // the input_range and output_range of the workload and compare it with the input_range and output_range
        // of the already allocated workloads. If some input overlapping is detected, or if any output can be direcly added to some other output,
        // the method will subtract the overlapping size from the size of memory that needs to be deallocated for the workload and then proceed with the deallocation.
          if (optimize){
            // check for overlapping in the input space
            int in_overlap = 0;
            for (auto & ow : _cur_allocated_workload){
              if ( w->layer == ow->layer){
                int in_overlap_with_workload = 0;
                in_overlap_with_workload = in_overlap_in(w, ow);
                assert(in_overlap_with_workload > 0);
                in_overlap += in_overlap_with_workload;
              }
            }
            size_to_deallocate -= in_overlap;
          }

          _cur_allocated_workload.erase(it);
          bool output_deallocation = false;
          if (!keep_output){
            output_deallocation = deallocate_(output_size);
          } else {
            output_deallocation = true;
          }
          return deallocate_(size_to_deallocate) && output_deallocation;
        }
        return false;
      };


      static int in_overlap_in(const ComputingWorkload * w1,const ComputingWorkload * w2){
        /*
        The method will be used to check if two workloads (belonging to the same level) have overlapping in the input space.
        The method will return the size of the overlapping space, if any, otherwise it will return 0.
        */

        int overlap = 0;
        int scale = FLIT_SIZE;
        // check for overlapping in the input space
        for (int i = 0; i < w1->input_range.lower_sp.size(); ++i){
          if (w2->input_range.lower_sp[i] <= w1->input_range.upper_sp[i] && w2->input_range.upper_sp[i] >= w1->input_range.lower_sp[i]){
            if (overlap == 0){
              int contribution = min(w2->input_range.upper_sp[i], w1->input_range.upper_sp[i]) - max(w2->input_range.lower_sp[i], w1->input_range.lower_sp[i]);
              overlap += contribution;
            } else {
              int contribution = min(w2->input_range.upper_sp[i], w1->input_range.upper_sp[i]) - max(w2->input_range.lower_sp[i], w1->input_range.lower_sp[i]);
              overlap *= contribution;
            }
          }
        }
        // multiply by the number of channels
        if (w1->input_range.channels.size() > 0 && w2->input_range.channels.size() > 0){
          overlap *= (w1->input_range.channels[1]- w1->input_range.channels[0]);
        }
        overlap /= scale;
        return overlap;
      };

      static int out_overlap_in(const ComputingWorkload * w1,const ComputingWorkload * w2){
        /*
        The method will be used to check if two workloads (belonging to adjacent levels) have overlapping between the output of the first
        and the input of the second. The method will return the size of the overlapping space, if any, otherwise it will return 0.
        */

        int overlap = 0;
        int scale = FLIT_SIZE;

        for (int i = 0; i < w1->output_range.lower_sp.size(); ++i){
          if (w2->input_range.lower_sp[i] <= w1->output_range.upper_sp[i] && w2->input_range.upper_sp[i] >= w1->output_range.lower_sp[i]){
            if (overlap == 0){
              int contribution = min(w2->input_range.upper_sp[i], w1->output_range.upper_sp[i]) - max(w2->input_range.lower_sp[i], w1->output_range.lower_sp[i]);
              overlap += contribution;
            } else {
              int contribution = min(w2->input_range.upper_sp[i], w1->output_range.upper_sp[i]) - max(w2->input_range.lower_sp[i], w1->output_range.lower_sp[i]);
              overlap *= contribution;
            }
          }
        }
        // multiply by the number of channels
        if (w1->output_range.channels.size() > 0 && w2->input_range.channels.size() > 0){
          overlap *= min(w1->output_range.channels[1], w2->input_range.channels[1]) - max(w1->output_range.channels[0], w2->input_range.channels[0]);
        }

        overlap /= scale;
        return overlap;
        
      };
      

    private:
      int _size;
      int _available; // total memory available
      float _threshold; // to avoid deadlocks
      // int _available_for_reconf; // the memory available for reconfigurations
      std::deque<const ComputingWorkload *> _cur_allocated_workload;
  };



  class MemorySet {
    /*
    The class will be used to manage the set of local memories for the PEs, acting as a central manager for 
    the memory units. This is required for correct the allocation of memory across different nodes as result messages
    are exchanged: if a node has completed the processing of a workload and has to send the results to another node, we will 
    first need to check if the memory of the destination processor is enough to store the results. If it is not, sending of
    results packets will have to be postponed untill the memory of the destination processor is freed up. Untill then, the space
    allotted to the finished partition in the source processor will have to be preserved.
    */

   public: 
    MemorySet(int nodes, int size) : _nodes(nodes), _size(size) {
      _pointed_workload_in_queue.resize(_nodes, nullptr);
      for (int i = 0; i < _nodes; ++i){
        _memory_units.push_back(MemoryUnit(size));
      }
    }

    void init(vector<deque<const ComputingWorkload * >> & waiting_workloads){
      for (int i = 0; i < _nodes; ++i){
        _pointed_workload_in_queue[i] = waiting_workloads[i].size() > 0 ? waiting_workloads[i][0] : nullptr;
      }
    }

    DependentInjectionProcess::MemoryUnit & getMemoryUnit(int node){
      return _memory_units[node];
    }

    bool is_allocated(int node, const ComputingWorkload * w){
      return _memory_units[node].is_allocated(w);
    }

    bool is_ready(const Packet * p, const std::deque<const ComputingWorkload *> & waiting_workloads){
      return _memory_units[p->dst].is_ready( p, waiting_workloads);
    }

    const ComputingWorkload *  get_current_workload(int node){
      return _pointed_workload_in_queue[node];
    }

    void set_current_workload(int node, const ComputingWorkload * w){
      _pointed_workload_in_queue[node] = w;
    }

    int getSize() const { return _size; }

    int getAvailable(int source){
      return _memory_units[source].getAvailable();
    }

    bool all_empty(){
      for (int i = 0; i < _nodes; ++i){
        if (!_memory_units[i].is_empty()){
          return false;
        }
      }
      return true;
    }

    bool is_empty(int node){
      return _memory_units[node].is_empty();
    }

    void all_reset(){
      for (int i = 0; i < _nodes; ++i){
        _memory_units[i].reset();
      }
    }

    void reset(int node){
      _memory_units[node].reset();
    }

    bool check_mem_avail(int node, int size){
      return _memory_units[node].check_mem_avail(size);
    }

    bool allocate_(int node, int size){
      return _memory_units[node].allocate_(size);
    }

    bool deallocate_(int node, int size){
      return _memory_units[node].deallocate_(size);
    }

    bool allocate(int node, const ComputingWorkload * w, bool optimize = false){
      return _memory_units[node].allocate(w, optimize);
    }

    bool deallocate_output(int node, const ComputingWorkload * w){
      return _memory_units[node].deallocate_output(w);
    }

    bool deallocate(int node, const ComputingWorkload * w, bool keep_output , bool optimize = false){
      return _memory_units[node].deallocate(w, keep_output, optimize);
    }


    private:
      int _nodes;
      int _size;
      std::vector<const ComputingWorkload *> _pointed_workload_in_queue; // a pointer to the next workload in the queue to be included in the next reconfiguration
      std::vector<MemoryUnit> _memory_units;

  };



  private:
    int _resorting;
    vector<int> _timer; // timer only used for workloads (eventually reconfiguration), packets processing time is managed in trafficmanager
    vector<bool> _decur;

    // ==================  RECONFIGURATION ==================
    bool _enable_reconf;
    std::vector<int> _reconf_deadlock_timer;
    std::vector<bool> _reconf_active;
    std::vector<std::map<int,set<int>>> _requiring_ouput_deallocation; // a datastructure to manage the deallocation of the output space for the workloads
    DependentInjectionProcess::MemorySet _memory_set;
    const NVMPar * _nvm;
    // ==================  RECONFIGURATION ==================

    vector<set<tuple<int, int, int>>> * _processed;
    vector<set<tuple<int, int, int>>> _executed;
    vector<deque<const Packet *>> _waiting_packets;
    vector<deque< const Packet *>> _pending_packets; // used as buffer to store packets whose dependecies have been cleared, but which cannot
    // be injected directly because of memory constraints on receiving node 
    vector<deque<const ComputingWorkload * >> _waiting_workloads;
    vector<const ComputingWorkload *> _pending_workloads; // used as buffer for workloads being processed

    const SimulationContext * _context;
    EventLogger * _logger;
    Clock * _clock;
    //a pointer to the traffic object, holding the packets to be injected
    TrafficPattern * _traffic;
    
    // a method to check if the dependencies have been satisfied for the packet/workload
    template <typename T>
    int _dependenciesSatisfied (const T * u, int source, bool extensive_search= false) const{
      // get the vector of dependencies and convert it to a set
      assert(u);
      assert((u->dep.size() > 0));
      set<int> dep(u->dep.begin(), u->dep.end());
      std::vector<int> dep_time;

      if (dep.size() == 1){
        // check wether the only element is -1
        if (find(dep.begin(), dep.end(), -1) != dep.end()){
          return 0;
        }
      }
      if (!extensive_search) // consider just the specified source
      {
        // check if the dependencies have been satisfied for the packet
        for (auto it = dep.begin(); it != dep.end(); ++it){
          // check in the processed packets for the coinsidered source if there are requests whose
          // id matches the dependency: THE PACKET ID MUST MATCH WITH THE DEPENDENCY, AS WELL AS BE A WRITE PACKET
          auto match = std::find_if(_processed->at(source).begin(), _processed->at(source).end(), [it](const tuple<int,int,int> & p){
            return std::get<0>(p) == *it && std::get<1>(p) == commType::WRITE;
          });
          if (match != _processed->at(source).end()){
            dep_time.push_back(std::get<2>(*match));
          }
        }

        // do the same for the executed workloads
        for (auto it = dep.begin(); it != dep.end(); ++it){
          // check in landed packets if the (last) dependency for the packet has been satisfied
          auto match = std::find_if(_executed[source].begin(), _executed[source].end(), [it](const tuple<int,int,int> & p){
            return std::get<0>(p) == *it;
          });
          if (match != _executed[source].end()){
            dep_time.push_back(std::get<2>(*match));
          }
        }
      }
      else{ // consider also all the other sources

        // check if the dependencies have been satisfied for the packet
        for (auto it = dep.begin(); it != dep.end(); ++it){
          // check in the processed packets for the coinsidered source if there are requests whose
          // id matches the dependency: THE PACKET ID MUST MATCH WITH THE DEPENDENCY, AS WELL AS BE A WRITE PACKET
          for (int i = 0; i < _nodes; ++i){
            auto match = std::find_if(_processed->at(i).begin(), _processed->at(i).end(), [it](const tuple<int,int,int> & p){
              return std::get<0>(p) == *it && std::get<1>(p) == commType::WRITE;
            });
            if (match != _processed->at(i).end()){
              dep_time.push_back(std::get<2>(*match));
            }
          }
        }

        // do the same for the executed workloads
        for (auto it = dep.begin(); it != dep.end(); ++it){
          // check in landed packets if the (last) dependency for the packet has been satisfied
          for (int i = 0; i < _nodes; ++i){
            auto match = std::find_if(_executed[i].begin(), _executed[i].end(), [it](const tuple<int,int,int> & p){
              return std::get<0>(p) == *it;
            });
            if (match != _executed[i].end()){
              dep_time.push_back(std::get<2>(*match));
            }
          }
        }
      }
      
      // if the set is empty, return the maximum time of the dependencies
      if (dep.size() == dep_time.size()){
        // std::cout << "\n"<<std::endl;
        // std::cout << "Dependencies satisfied for node: " << source << std::endl;
        for(auto & t : dep_time){
          // std::cout << t << " ";
        }
        // std::cout << "\n"<<std::endl;
        return *max_element(dep_time.begin(), dep_time.end());
      }
      else{
        return -1;
      }
    }

    // a new method to resort the waiting queues: if the function gets called, 
    // it will loop over the waiting queues and check if the dependencies have been satisfied.
    // If so, the packet/workload will be moved to the front of the queue. If no packets/workloads
    // dependencies have been satisfied, the function will scan over all the elements in the queue and 
    // finally return false. Otherwise, it stops at the first element that has dependencies satisfied
    // and moves it to the front of the queue, returning true
    template <typename T>
    bool _resortWaitingQueues(T& container, int source) {
        bool resorted = false;
        // scan over the waiting packets
        for (auto it = container[source].begin(); it != container[source].end(); ++it) {
            // check if the dependencies have been satisfied
            if (_dependenciesSatisfied(&(*(*it)), source) >= 0) {
                // if the item is already at the front of the queue, do nothing
                if (it == container[source].begin()) {
                    return resorted;
                }
                // else, move the item to the front of the queue and return true
                it = container[source].erase(it);
                container[source].push_front(*it);
                resorted = true;
                return resorted;
            }
        }
        return resorted;
    }

    //================ RECONFIGURATION =================
    // a method to manage the reconfiguration of the PEs
    int _reconfigure(int source);
    // a method to check if the a reconfiguration should be performed
    bool _checkReconfNeed(int source, bool bypass_empty_memory_check = false);
    // a method to manage end of computation and integration with reconfiguration
    bool _manageReconfPacketInjection(const Packet * p, int source);
    //================ RECONFIGURATION =================

    bool _managePacketInjection(const Packet * p);
    // a method to build the waiting queues
    void _buildStaticWaitingQueues();
    // a method to set the clock to a specific value
    void _setProcessingTime(int node, int value);

  public:
    DependentInjectionProcess(int nodes, int local_memory_size , int reconfig_cycles , Clock * clock, TrafficPattern * traffic ,  vector<set<tuple<int,int,int>>> * landed_packets, const SimulationContext * context , int resort = 0);
    virtual void reset();
    // a method used to append additional packets to the waiting queues
    virtual void addToWaitingQueue(int source, Packet * p);
    // a method to remove a packet id from _requiring_ouput_deallocation
    void removeFromRequiringOutputDeallocation(int source, int id, int packet_id){
      _requiring_ouput_deallocation[source].at(id).erase(packet_id);
    }
    // a method to check if the set of packets corresponding to a workload in _requiring_ouput_deallocation is empty
    bool isRequiringOutputDeallocationEmpty(int source, int id){
      return _requiring_ouput_deallocation[source].at(id).empty();
    }
    // a method to test if the landed packets can be processed
    virtual bool isIdle(int source){return _timer[source] == 0;};
    // a method to get the total available memory
    virtual int getAvailableMemory(int source){return _memory_set.getAvailable(source);};
    // a method to deallocate memory
    virtual void deallocateMemory(int source, const ComputingWorkload * w){_memory_set.deallocate_output(source, w);};
    // a method to stage reconfiguration
    virtual void stageReconfiguration(int source, bool bypass_empty_memory_check = false);
    // finally, a method to test if the packet can be injected
    virtual bool test(int source);
    
};

#endif 
