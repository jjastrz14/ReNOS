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
  virtual void reset();
  bool reached_end;
  static InjectionProcess * New(string const & inject, int nodes, double load,
				Configuration const * const config = NULL);
  static InjectionProcess * NewUserDefined(string const & inject, int nodes, Clock * clock, TrafficPattern * traffic, const NVMPar * nvm_par, vector<set<tuple<int,int,int>>> * landed_packets, EventLogger * logger,
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
    int id; // the id of the task after which to perform reconfiguration
    int wsize; // the size of the weights to be pulled from the NVM
  };

  private:
    int _resorting;
    vector<int> _timer; // timer only used for workloads (eventually reconfiguration), packets processing time is managed in trafficmanager
    vector<bool> _decur;

    // ==================  RECONFIGURATION ==================
    vector<bool> _reconf_active; // a vector of booleans to check if the PE is being reconfigured: this is used only if the reconfiguration is enabled
    vector<std::deque<ReconfigBit>> _scheduled_reconf; // a vector of queues to store the ids of the workload after which a reconfiguration will have to be performed
                                               // OSS: a reconfiguration is simply simulated by incrementing the timer of the PE by the time required to
                                               // perform the reconf. i.e. bring down the new weights from the top NVM to the PE. The unit of time ([flit size / cycle])
                                                // is given by the NVMPar object
    // ==================  RECONFIGURATION ==================

    vector<set<tuple<int, int, int>>> * _processed;
    vector<set<tuple<int, int, int>>> _executed;
    vector<deque<const Packet *>> _waiting_packets;
    //vector<const Packet *> _pending_packets; // used as buffer for packets being processed
    vector<deque<const ComputingWorkload * >> _waiting_workloads;
    vector<const ComputingWorkload *> _pending_workloads; // used as buffer for workloads being processed

    EventLogger * _logger;
    Clock * _clock;
    //a pointer to the traffic object, holding the packets to be injected
    TrafficPattern * _traffic;
    const NVMPar * _nvm_par;
    
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

    // a method to build the waiting queues
    void _buildStaticWaitingQueues();
    // a method to set the clock to a specific value
    void _setProcessingTime(int node, int value);

  public:
    DependentInjectionProcess(int nodes,Clock * clock, TrafficPattern * traffic ,const NVMPar * nvm_par ,  vector<set<tuple<int,int,int>>> * landed_packets, int resort = 0,EventLogger * logger = nullptr);
    virtual void reset();
    // a method used to append additional packets to the waiting queues
    virtual void addToWaitingQueue(int source, Packet * p);
    // a method to test if the landed packets can be processed
    virtual bool isIdle(int source){return _timer[source] == 0;};
    // finally, a method to test if the packet can be injected
    virtual bool test(int source);
    
};

#endif 
