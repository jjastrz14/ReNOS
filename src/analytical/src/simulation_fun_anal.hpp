///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: simulation_fun_anal.hpp
//  Description: High-level simulation functions for the analytical NoC model
//  Created by:  Jakub Jastrzebski
//  Date:  15/09/2025
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _SIMULATION_FUN_ANAL_HPP_
#define _SIMULATION_FUN_ANAL_HPP_

#include <string>
#include <tuple>
#include <memory>

// Forward declaration
class AnalyticalLogger;
class AnalyticalModel;

/**
 * High-level wrapper function to run analytical simulation
 *
 * @param config_file Path to JSON configuration file
 * @param output_file Path to output file ("-" for stdout, "" for no output)
 * @return Tuple of (simulation_time, logger_pointer)
 */
std::tuple<int, AnalyticalLogger*> SimulateAnalytical(const std::string& config_file,
                                                     const std::string& output_file = "");

/**
 * Create and configure an analytical model from config file
 *
 * @param config_file Path to JSON configuration file
 * @return Smart pointer to configured analytical model
 */
std::unique_ptr<AnalyticalModel> CreateAnalyticalModel(const std::string& config_file);

/**
 * Run analytical simulation with a pre-configured model
 *
 * @param model Pointer to configured analytical model
 * @param output_file Path to output file ("-" for stdout, "" for no output)
 * @return Simulation time in cycles
 */
int RunAnalyticalSimulation(AnalyticalModel* model, const std::string& output_file = "");

/**
 * Validate configuration file for analytical simulation
 *
 * @param config_file Path to JSON configuration file
 * @return True if configuration is valid, false otherwise
 */
bool ValidateAnalyticalConfig(const std::string& config_file);

/**
 * Get model information and capabilities
 *
 * @return String describing the analytical model version and features
 */
std::string GetAnalyticalModelInfo();

#endif // _SIMULATION_FUN_ANAL_HPP_