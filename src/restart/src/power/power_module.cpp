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

#include "power_module.hpp"
#include "booksim_config.hpp"
#include "buffer_monitor.hpp"
#include "switch_monitor.hpp"
#include "iq_router.hpp"
#include "logger.hpp"

Power_Module::Power_Module(Network * n , const Configuration &config)
  : Module( 0, "power_module" ){

  
  string pfile = config.getStrField("tech_file");
  PowerConfig pconfig;
  pconfig.parseJSONFile(pfile);

  net = n;
  output_file_name = config.getStrField("power_output_file");
  classes = config.getIntField("classes");
  channel_width = (double)config.getIntField("channel_width");
  channel_sweep = (double)config.getIntField("channel_sweep");

  numVC = (double)config.getIntField("num_vcs");
  depthVC  = (double)config.getIntField("vc_buf_size");

  //////////////////////////////////Constants/////////////////////////////
  //wire length in (mm)
  wire_length = pconfig.getFloatField("wire_length");
  //////////Metal Parameters////////////
  // Wire left/right coupling capacitance [ F/mm ]
  Cw_cpl = pconfig.getFloatField("Cw_cpl"); 
  // Wire up/down groudn capacitance      [ F/mm ]
  Cw_gnd = pconfig.getFloatField("Cw_gnd");
  Cw = 2.0 * Cw_cpl + 2.0 * Cw_gnd ;
  Rw = pconfig.getFloatField("Rw");
  // metal pitch [mm]
  MetalPitch = pconfig.getFloatField("MetalPitch"); 
  
  //////////Device Parameters////////////
  
  LAMBDA =  pconfig.getFloatField("LAMBDA")  ;       // [um/LAMBDA]
  Cd     =  pconfig.getFloatField("Cd");           // [F/um] (for Delay)
  Cg     =  pconfig.getFloatField("Cg");           // [F/um] (for Delay)
  Cgdl   =  pconfig.getFloatField("Cgdl");           // [F/um] (for Delay)
  
  Cd_pwr =  pconfig.getFloatField("Cd_pwr") ;           // [F/um] (for Power)
  Cg_pwr =  pconfig.getFloatField("Cg_pwr") ;           // [F/um] (for Power)
			       
  IoffN  = pconfig.getFloatField("IoffN");            // [A/um]
  IoffP  = pconfig.getFloatField("IoffP");            // [A/um]
  // Leakage from bitlines, two-port cell  [A]
  IoffSRAM = pconfig.getFloatField("IoffSRAM");  
  // [Ohm] ( D1=1um Inverter)
  R        = pconfig.getFloatField("R");                  
  // [F]   ( D1=1um Inverter - for Power )
  Ci_delay = (1.0 + 2.0) * ( Cg + Cgdl );   
  // [F]   ( D1=1um Inverter - for Power )
  Co_delay = (1.0 + 2.0) * Cd ;              


  Ci = (1.0 + 2.0) * Cg_pwr ;
  Co = (1.0 + 2.0) * Cd_pwr ;

  Vdd    = pconfig.getFloatField("Vdd");
  FO4    = R * ( 3.0 * Cd + 12 * Cg + 12 * Cgdl);		     
  tCLK   = 20 * FO4;
  fCLK   = 1.0 / tCLK;              

  H_INVD2=(double)pconfig.getIntField("H_INVD2");
  W_INVD2=(double)pconfig.getIntField("W_INVD2") ;
  H_DFQD1=(double)pconfig.getIntField("H_DFQD1");
  W_DFQD1= (double)pconfig.getIntField("W_DFQD1");
  H_ND2D1= (double)pconfig.getIntField("H_ND2D1");
  W_ND2D1=(double)pconfig.getIntField("W_ND2D1");
  H_SRAM=(double)pconfig.getIntField("H_SRAM");
  W_SRAM=(double)pconfig.getIntField("W_SRAM");

  ChannelPitch = 2.0 * MetalPitch ;
  CrossbarPitch = 2.0 * MetalPitch ;
}

Power_Module::~Power_Module(){


}


//////////////////////////////////////////////
//Channels
//////////////////////////////////////////////

void Power_Module::calcChannel(const FlitChannel* f){
  double channelLength = f->getLatency()* wire_length;
  wire const this_wire = wireOptimize(channelLength);
  double const & K = this_wire.K;
  double const & N = this_wire.N;
  double const & M = this_wire.M;
  //area
  channelArea += areaChannel(K,N,M);

  //activity factor;
  const vector<int> temp = f->getActivity();
  vector<double> a(classes);
  for(int i = 0; i< classes; i++){

    a[i] = ((double)temp[i])/totalTime;
  }

  //power calculation
  double const bitPower = powerRepeatedWire(channelLength, K,M,N);

  channelClkPower += powerWireClk(M,channel_width);
  for(int i = 0; i< classes; i++){
    channelWirePower += bitPower * a[i]*channel_width;
    channelDFFPower += powerWireDFF(M, channel_width, a[i]);
  }
  channelLeakPower+= powerRepeatedWireLeak(K,M,N)*channel_width;
}

wire const & Power_Module::wireOptimize(double L){
  map<double, wire>::iterator iter = wire_map.find(L);
  if(iter == wire_map.end()){
    
    double W = 64;
    double bestMetric =  100000000 ;
    double bestK = -1;
    double bestM = -1;
    double bestN = -1;
    for (double K = 1.0 ; K < 10 ; K+=0.1 ) {
      for (double N = 1.0 ; N < 40 ; N += 1.0 ) {
	for (double M = 1.0 ; M < 40.0 ; M +=1.0 ) {
	  double l = 1.0 * L/( N * M) ;
	  
	  double k0 = R * (Co_delay + Ci_delay) ;
	  double k1 = R/K * Cw + K * Rw * Ci_delay ;
	  double k2 = 0.5 * Rw * Cw ;
	  double Tw = k0 + (k1 * l) + k2 * (l * l) ;
	  double alpha = 0.2 ;
	  double power = alpha * W * powerRepeatedWire( L, K, M, N) + powerWireDFF( M, W, alpha ) ;
	  double metric = M * M * M * M * power ;
	  if ( (N*Tw) < (0.8 * tCLK) ) {
	    if ( metric < bestMetric ) {
	      bestMetric = metric ;
	      bestK = K ;
	      bestM = M ;
	      bestN = N ;
	    }
	  }
	}
      }
    }
    *(net->context->gDumpFile)<<"L = "<<L<<" K = "<<bestK<<" M = "<<bestM<<" N = "<<bestN<<endl;
    
    wire const temp = {L, bestK, bestM, bestN};
    iter = wire_map.insert(make_pair(L, temp)).first;
  }
  return iter->second;
}

double Power_Module::powerRepeatedWire(double L, double K, double M, double N){
  
  double segments = 1.0 * M * N ;
  double Ca = K * (Ci + Co) + Cw * (L/segments) ;
  double Pa = 0.5 * Ca * Vdd * Vdd * fCLK;
  return Pa * M * N  ;

}

double Power_Module::powerRepeatedWireLeak (double K, double M, double N){
  double Pl = K * 0.5 * ( IoffN + 2.0 * IoffP ) * Vdd  ;
  return Pl * M * N ;

}

double Power_Module:: powerWireClk (double M, double W){
  // number of clock wires running down one repeater bank
  double columns = H_DFQD1 * MetalPitch /  ChannelPitch ;

  // length of clock wire
  double clockLength = W * ChannelPitch ;
  double Cclk = (1 + 5.0/16.0 * (1+Co_delay/Ci_delay)) * (clockLength * Cw * columns +W * Ci_delay);

  return M * Cclk * (Vdd * Vdd) * fCLK ;

}

double Power_Module::powerWireDFF(double M, double W, double alpha){
  double Cdin = 2 * 0.8 * (Ci + Co) + 2 * ( 2.0/3.0 * 0.8 * Co )  ;
  double Cclk = 2 * 0.8 * (Ci + Co) + 2 * ( 2.0/3.0 * 0.8 * Cg_pwr) ;
  double Cint = (alpha * 0.5) * Cdin + alpha * Cclk ;
  
  return Cint * M * W * (Vdd*Vdd) * fCLK ;
}


///////////////////////////////////////////////////////////////
//Memory
//////////////////////////////////////////////////////////////
void Power_Module::calcBuffer(const BufferMonitor *bm){
  double depth = numVC * depthVC  ;
  double Pleak = powerMemoryBitLeak( depth ) * channel_width ;
  //area

  const vector<int> reads = bm->GetReads();
  const vector<int> writes = bm->GetWrites();
  for(int i = 0; i<bm->NumInputs(); i++){
    inputArea += areaInputModule( depth );
    inputLeakagePower += Pleak ;
    for(int j = 0; j< classes; j++){
      double ar = ((double)reads[i* classes+j])/totalTime;
      double aw = ((double)writes[i* classes+j])/totalTime;
      if(ar>1 ||aw >1){
	*(net->context->gDumpFile)<<"activity factor is greater than one, soemthing is stomping memory\n"; exit(-1);
      }
      double Pwl =  powerWordLine( channel_width, depth) ;
      double Prd = powerMemoryBitRead( depth ) * channel_width ;
      double Pwr = powerMemoryBitWrite( depth ) * channel_width ; 
      inputReadPower    += ar * ( Pwl + Prd ) ;
      inputWritePower   += aw * ( Pwl + Pwr ) ;
    }
  }
}


double Power_Module::powerWordLine(double memoryWidth, double memoryDepth){
  // wordline capacitance
  double Ccell = 2 * ( 4.0 * LAMBDA ) * Cg_pwr +  6 * MetalPitch * Cw ;     
  double Cwl = memoryWidth * Ccell ; 

  // wordline circuits
  double Warray = 8 * MetalPitch + memoryDepth ;
  double x = 1.0 + (5.0/16.0) * (1 + Co/Ci)  ;
  double Cpredecode = x * (Cw * Warray  * Ci) ;
  double Cdecode    = x * Cwl ;

  // bitline circuits
  double Harray =  6 * memoryWidth * MetalPitch ;
  double y = (1 + 0.25) * (1 + Co/Ci) ;
  double Cprecharge = y * ( Cw * Harray + 3 * channel_width * Ci ) ;
  double Cwren      = y * ( Cw * Harray + 2 * channel_width * Ci ) ;

  double Cbd = Cprecharge + Cwren ;
  double Cwd = 2 * Cpredecode + Cdecode ;

  return ( Cbd + Cwd ) * Vdd * Vdd * fCLK ;
  
}

double Power_Module::powerMemoryBitRead(double memoryDepth){
  // bitline capacitance
  double Ccell  = 4.0 * LAMBDA * Cd_pwr + 8 * MetalPitch * Cw ; 
  double Cbl    = memoryDepth * Ccell ;
  double Vswing = Vdd  ;
  return ( Cbl ) * ( Vdd * Vswing ) * fCLK ;
}

double Power_Module:: powerMemoryBitWrite(double memoryDepth){
  // bitline capacitance
  double Ccell  = 4.0 * LAMBDA * Cd_pwr + 8 * MetalPitch * Cw ; 
  double Cbl    = memoryDepth * Ccell ;

  // internal capacitance
  double Ccc    = 2 * (Co + Ci) ;

  return (0.5 * Ccc * (Vdd*Vdd)) + ( Cbl ) * ( Vdd * Vdd ) * fCLK ;
}

double Power_Module::powerMemoryBitLeak(double memoryDepth ){
  
  return memoryDepth * IoffSRAM * Vdd ;
}

///////////////////////////////////////////////////////////////
//switch
//////////////////////////////////////////////////////////////

void Power_Module::calcSwitch(const SwitchMonitor* sm){

  switchArea += areaCrossbar(sm->NumInputs(), sm->NumOutputs());
  outputArea += areaOutputModule(sm->NumOutputs());
  switchPowerLeak += powerCrossbarLeak(channel_width, sm->NumInputs(), sm->NumOutputs());

  const vector<int> activity = sm->GetActivity();
  vector<double> type_activity(classes);

  for(int i = 0; i<sm->NumOutputs(); i++){
    for(int k = 0; k<classes; k++){
      type_activity[k] = 0;
    }
    for(int j = 0; j<sm->NumInputs(); j++){
      for(int k  = 0; k<classes; k++){
	double a = activity[k+classes*(i+sm->NumOutputs()*j)];
	a = a/totalTime;
	if(a>1){
	  *(net->context->gDumpFile)<<"Switcht activity factor is greater than 1!!!\n";exit(-1);
	}
	double Px = powerCrossbar(channel_width, sm->NumInputs(),sm->NumOutputs(),j,i);
	switchPower += a*channel_width*Px;
	switchPowerCtrl += a *powerCrossbarCtrl(channel_width,  sm->NumInputs(),sm->NumOutputs());
	type_activity[k]+=a;
      }
    }
    outputPowerClk += powerWireClk( 1, channel_width ) ;
    for(int k = 0; k<classes; k++){
      outputPower += type_activity[k] * powerWireDFF( 1, channel_width, 1.0 ) ;
      outputCtrlPower += type_activity[k] * powerOutputCtrl(channel_width ) ;
    }
  }

}

double Power_Module::powerCrossbar(double width, double inputs, double outputs, double from, double to){
  // datapath traversal power
  double Wxbar = width * outputs * CrossbarPitch ;
  double Hxbar = width * inputs  * CrossbarPitch ;

  // wires
  double CwIn  = Wxbar * Cw ;
  double CwOut = Hxbar * Cw ;

  // cross-points
  double Cxi = (1.0/16.0) * CwOut ;
  double Cxo = 4.0 * Cxi * (Co_delay/Ci_delay) ;

  // drivers
  double Cti = (1.0/16.0) * CwIn ;
  double Cto = 4.0 * Cti * (Co_delay/Ci_delay) ;

  double CinputDriver = 5.0/16.0 * (1 + Co_delay/Ci_delay) * (0.5 * Cw * Wxbar + Cti) ;

  // total switched capacitance
  
  //this maybe missing +Cto
  double Cin  = CinputDriver + CwIn + Cti + (outputs * Cxi) ;
  if ( to < outputs/2 ) {
    Cin -= ( 0.5 * CwIn + outputs/2 * Cxi) ;
  }
  //this maybe missing +cti
  double Cout = CwOut + Cto + (inputs * Cxo) ;
  if ( from < inputs/2) {
    Cout -= ( 0.5 * CwOut + (inputs/2 * Cxo)) ;
  }
  return 0.5 * (Cin + Cout) * (Vdd * Vdd * fCLK) ;
}


double Power_Module::powerCrossbarCtrl(double width, double inputs, double outputs){
 
  // datapath traversal power
  double Wxbar = width * outputs * CrossbarPitch ;
  double Hxbar = width * inputs  * CrossbarPitch ;

  // wires
  double CwIn  = Wxbar * Cw ;

  // drivers
  double Cti  = (5.0/16.0) * CwIn ;

  // need some estimate of how many control wires are required
  double Cctrl  = width * Cti + (Wxbar + Hxbar) * Cw  ; 
  double Cdrive = (5.0/16.0) * (1 + Co_delay/Ci_delay) * Cctrl ;

  return (Cdrive + Cctrl) * (Vdd*Vdd) * fCLK ;
  
}

double Power_Module::powerCrossbarLeak (double width, double inputs, double outputs){
  // datapath traversal power
    double Wxbar = width * outputs * CrossbarPitch ;
    double Hxbar = width * inputs  * CrossbarPitch ;

    // wires
    double CwIn  = Wxbar * Cw ;
    double CwOut = Hxbar * Cw ;
    // cross-points
    double Cxi = (1.0/16.0) * CwOut ;
    // drivers
    double Cti  = (1.0/16.0) * CwIn ;

    return 0.5 * (IoffN + 2 * IoffP)*width*(inputs*outputs*Cxi+inputs*Cti+outputs*Cti)/Ci;
}

//////////////////////////////////////////////////////////////////
//output module
//////////////////////////////////////////////////////////////////
double Power_Module:: powerOutputCtrl(double width) {

    double Woutmod = channel_width * ChannelPitch ;
    double Cen     = Ci ;

    double Cenable = (1 + 5.0/16.0)*(1.0+Co/Ci)*(Woutmod* Cw + width* Cen) ;

    return Cenable * (Vdd*Vdd) * fCLK ;
    
}

//////////////////////////////////////////////////////////////////
//area
//////////////////////////////////////////////////////////////////

double Power_Module:: areaChannel (double K, double N, double M){

    double Adff = M * W_DFQD1 * H_DFQD1 ;
    double Ainv = M * N * ( W_INVD2 + 3 * K) * H_INVD2 ;

    return channel_width * (Adff + Ainv) * MetalPitch * MetalPitch ;
}

double Power_Module:: areaCrossbar(double Inputs, double Outputs) {
    return (Inputs * channel_width * CrossbarPitch) * (Outputs * channel_width * CrossbarPitch) ;
}

double Power_Module:: areaInputModule(double Words) {
    double Asram =  ( channel_width * H_SRAM ) * (Words * W_SRAM) ;
    return Asram * (MetalPitch * MetalPitch) ;
}

double Power_Module:: areaOutputModule(double Outputs) {
    double Adff = Outputs * W_DFQD1 * H_DFQD1 ;
    return channel_width * Adff * MetalPitch * MetalPitch ;
}

void Power_Module::run(){
  totalTime = GetSimTime(net->context);
  channelWirePower=0;
  channelClkPower=0;
  channelDFFPower=0;
  channelLeakPower=0;
  inputReadPower=0;
  inputWritePower=0;
  inputLeakagePower=0;
  switchPower=0;
  switchPowerCtrl=0;
  switchPowerLeak=0;
  outputPower=0;
  outputPowerClk=0;
  outputCtrlPower=0;
  channelArea=0;
  switchArea=0;
  inputArea=0;
  outputArea=0;
  maxInputPort = 0;
  maxOutputPort = 0;

  vector<FlitChannel *> inject = net->GetInject();
  vector<FlitChannel *> eject = net->GetEject();
  vector<FlitChannel *> chan = net->GetChannels();
  
  for(int i = 0; i<net->NumNodes(); i++){
    calcChannel(inject[i]);
  }

  for(int i = 0; i<net->NumNodes(); i++){
    calcChannel(eject[i]);
  }

  for(int i = 0; i<net->NumChannels();i++){
    calcChannel(chan[i]);
  }

  vector<Router*> routers = net->GetRouters();
  for(size_t i = 0; i < routers.size(); i++){
    IQRouter* temp = dynamic_cast<IQRouter*>(routers[i]);
    const BufferMonitor * bm = temp->GetBufferMonitor();
    calcBuffer(bm);
    const SwitchMonitor * sm = temp->GetSwitchMonitor();
    calcSwitch(sm);
  }
  
  double totalpower =  channelWirePower+channelClkPower+channelDFFPower+channelLeakPower+ inputReadPower+inputWritePower+inputLeakagePower+ switchPower+switchPowerCtrl+switchPowerLeak+outputPower+outputPowerClk+outputCtrlPower;
  double totalarea =  channelArea+switchArea+inputArea+outputArea;
  *(net->context->gDumpFile)<< "-----------------------------------------\n";
  *(net->context->gDumpFile)<< "- OCN Power Summary\n" ;
  *(net->context->gDumpFile)<< "-----------------------------------------\n";
  *(net->context->gDumpFile)<< "- Vdd:                   "<<Vdd<<" V \n" ;
  *(net->context->gDumpFile)<< "- R:                     "<<R<<" Ohm \n" ;
  *(net->context->gDumpFile)<< "- fCLK:                  "<<fCLK<<" Hz \n" ;
  *(net->context->gDumpFile)<< "-----------------------------------------\n" ;
  *(net->context->gDumpFile)<< "- Completion Time:         "<<totalTime <<" cycles \n" ;
  *(net->context->gDumpFile)<< "- Flit Widths:            "<<channel_width<<" bits \n" ;
  *(net->context->gDumpFile)<< "- Channel Wire Power:      "<<channelWirePower <<" W \n" ;
  *(net->context->gDumpFile)<< "- Channel Clock Power:     "<<channelClkPower <<" W \n" ;
  *(net->context->gDumpFile)<< "- Channel Retiming Power:  "<<channelDFFPower <<" W \n" ;
  *(net->context->gDumpFile)<< "- Channel Leakage Power:   "<<channelLeakPower <<" W \n" ;
  
  *(net->context->gDumpFile)<< "- Input Read Power:        "<<inputReadPower <<" W \n" ;
  *(net->context->gDumpFile)<< "- Input Write Power:       "<<inputWritePower <<" W \n" ;
  *(net->context->gDumpFile)<< "- Input Leakage Power:     "<<inputLeakagePower <<" W \n" ;
  
  *(net->context->gDumpFile)<< "- Switch Power:            "<<switchPower <<" W \n" ;
  *(net->context->gDumpFile)<< "- Switch Control Power:    "<<switchPowerCtrl <<" W \n" ;
  *(net->context->gDumpFile)<< "- Switch Leakage Power:    "<<switchPowerLeak <<" W \n" ;
  
  *(net->context->gDumpFile)<< "- Output DFF Power:        "<<outputPower <<" W \n" ;
  *(net->context->gDumpFile)<< "- Output Clk Power:        "<<outputPowerClk <<" W \n" ;
  *(net->context->gDumpFile)<< "- Output Control Power:    "<<outputCtrlPower <<" W \n" ;
  *(net->context->gDumpFile)<< "- Total Power:             "<<totalpower <<" W \n";
  *(net->context->gDumpFile)<< "-----------------------------------------\n" ;
  *(net->context->gDumpFile)<< "\n" ;
  *(net->context->gDumpFile)<< "-----------------------------------------\n" ;
  *(net->context->gDumpFile)<< "- OCN Area Summary\n" ;
  *(net->context->gDumpFile)<< "- Channel Area:  "<<channelArea<<" mm^2\n" ;
  *(net->context->gDumpFile)<< "- Switch  Area:  "<<switchArea<<" mm^2\n" ;
  *(net->context->gDumpFile)<< "- Input  Area:   "<<inputArea<<" mm^2\n" ;
  *(net->context->gDumpFile)<< "- Output  Area:  "<<outputArea<<" mm^2\n" ;
  *(net->context->gDumpFile)<< "- Total Area:    "<<totalarea<<" mm^2\n" ;
  *(net->context->gDumpFile)<< "-----------------------------------------\n" ;

  // Store power summary in logger if logger is enabled
  if (net->context->logger != nullptr) {
    PowerSummary power_summary;
    power_summary.vdd = Vdd;
    power_summary.resistance = R;
    power_summary.fclk = fCLK;
    power_summary.completion_time_cycles = totalTime;
    power_summary.flit_width_bits = (int)channel_width;
    power_summary.channel_wire_power = channelWirePower;
    power_summary.channel_clock_power = channelClkPower;
    power_summary.channel_retiming_power = channelDFFPower;
    power_summary.channel_leakage_power = channelLeakPower;
    power_summary.input_read_power = inputReadPower;
    power_summary.input_write_power = inputWritePower;
    power_summary.input_leakage_power = inputLeakagePower;
    power_summary.switch_power = switchPower;
    power_summary.switch_control_power = switchPowerCtrl;
    power_summary.switch_leakage_power = switchPowerLeak;
    power_summary.output_dff_power = outputPower;
    power_summary.output_clk_power = outputPowerClk;
    power_summary.output_control_power = outputCtrlPower;
    power_summary.total_power = totalpower;

    net->context->logger->set_power_summary(power_summary);
  }

}
