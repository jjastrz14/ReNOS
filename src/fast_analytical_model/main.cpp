///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: main.cpp
//  Description: Main entry point for fast analytical model testing
//  Created by:  Claude Code
//  Date:  23/09/2025
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <chrono>
#include "fast_model.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file.json>" << std::endl;
        return 1;
    }

    std::string config_file = argv[1];

    try {
        std::cout << "Fast Analytical NoC Simulator" << std::endl;
        std::cout << "Configuration file: " << config_file << std::endl;

        // Measure execution time
        auto start = std::chrono::high_resolution_clock::now();

        // Run simulation
        int simulation_time = simulate_fast_analytical(config_file);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end - start);

        std::cout << "Simulation completed!" << std::endl;
        std::cout << "Total simulation time: " << simulation_time << " cycles" << std::endl;
        std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}