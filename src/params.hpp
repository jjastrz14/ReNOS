
///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: params.hpp
//  Description: File used to group all the most relevant parameters used in the simulation
//                  
//               Great inspiration taken from the Booksim2 NoC simulator (https://github.com/booksim/booksim2)
//               Copyright (c) 2007-2015, Trustees of The Leland Stanford Junior University
//               All rights reserved.
//
//               Redistribution and use in source and binary forms, with or without
//               modification, are permitted provided that the following conditions are met:
//
//               Redistributions of source code must retain the above copyright notice, this 
//               list of conditions and the following disclaimer.
//               Redistributions in binary form must reproduce the above copyright notice, this
//               list of conditions and the following disclaimer in the documentation and/or
//               other materials provided with the distribution.
//
//               THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//               ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//               WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
//               DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
//               ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//               (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//               LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
//               ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//               (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//               SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//  Created by:  Edoardo Cabiati
//  Date:  20/09/2024
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <vector>
#include <map>
#include <cassert>
#include <iostream>

#define COMM_DELAY 1
#define FLIT_SIZE 64 // in bytes
#define MAX_LOCAL_MEMORY 56 // in flit size
#define MAX_NVM_MEMORY 1024 // in flit size

// the following file is used to host the parameters:
// - reconfiguration rates for a certain technological node
// - the MAC/FLOP rate for a certain type of workload
// to be used for the UserDefinedTrafficPattern class

enum class WorkloadType{
    CONV, // convolutional wokload
    FC, // fully connected workload
};

class NVMPar {
    public:
        // constructor
        NVMPar( double reconf_rate /* [flit size / s] */, double clock_freq /* [Hz] */) : _reconf_rate(reconf_rate), _clock_freq(clock_freq) { assert(_reconf_rate > 0. || _reconf_cycles > 0.);} 
        NVMPar( double reconf_cycles /* [flit size / cycle] */) : _reconf_cycles(reconf_cycles) {}

        int cycles_reconf(int size /* [ flit size ] */, double noc_freq = -1.) const {

            

            if (_reconf_rate > 0. && noc_freq != -1.) {
                return std::ceil(double(size)/ byte_per_cycles(noc_freq));
            }

            return std::ceil(double(size) *  _reconf_cycles);
        }
        
        // a method to get the byte per cycles
        double byte_per_cycles(double noc_freq = -1. /* [Hz] */) const {
            assert(_reconf_rate > 0);
            if (noc_freq == -1) {
                return _reconf_rate / _clock_freq;
            }
            double scale = _clock_freq / noc_freq;
            return _reconf_rate * scale;
        }

        
    private:
        // the reconfiguration rate for a certain technological node [bytes/s]
        double _reconf_rate;
        double _reconf_cycles;
        // the clock frequency used for the NVM [Hz]
        double _clock_freq;
};
        

class NPUPar {
    public:
        // constructor
        NPUPar(double clock_freq /* [Hz] */, std::map<WorkloadType, double> mac_flop_rate /* [flop/s] */) : _clock_freq(clock_freq), _mac_flop_rate(mac_flop_rate) {}

        int cycles_workload(int size, WorkloadType type) const {
            assert(_mac_flop_rate.find(type) != _mac_flop_rate.end());
            return std::ceil(double(size) / mac_flop_per_cycles(type));
        }

        // a method to get the MAC/FLOP rate for a certain type of workload
        double mac_flop_per_cycles(WorkloadType type, double noc_freq = -1 /* [Hz] */) const {
            if (noc_freq == -1.) {
                return _mac_flop_rate.at(type) / _clock_freq;
            }
            double scale = _clock_freq / noc_freq;
            return _mac_flop_rate.at(type) * scale;
        }

    private:
        // the MAC/FLOP rate for a certain type of workload
        std::map<WorkloadType, double> _mac_flop_rate;
        // clock frequency used for the NPU
        double _clock_freq;
};
#endif