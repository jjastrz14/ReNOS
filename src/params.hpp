
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

#define COMM_DELAY 1

// the following file is used to host the parameters:
// - reconfiguration rates for a certain technological node
// - the MAC/FLOP rate for a certain type of workload
// to be used for the UserDefinedTrafficPattern class

enum class WorkloadType{
    conv, // convolutional wokload
    fc, // fully connected workload
};


class NVMParams {
    public:
        // constructor
        NVMParams(double reconf_rate, double clock_freq) : reconf_rate(reconf_rate), clock_freq(clock_freq) {}
        
        // a method to get the byte per cycles
        double get_byte_per_cycles() const {
            return reconf_rate / clock_freq;
        }
    private:
        // the reconfiguration rate for a certain technological node
        double reconf_rate;
        // the clock frequency used for the NVM
        double clock_freq;
};
        

class NPUParams {
    private:
        // the MAC/FLOP rate for a certain type of workload
        std::map<WorkloadType, double> mac_flop_rate;
        // clock frequency used for the NPU
        double clock_freq;
};
#endif