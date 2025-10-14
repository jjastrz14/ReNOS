///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: main.cpp
//  Description: Main entry point for analytical NoC model
//  Created by:  Jakub Jastrzebski
//  Date:  15/09/2025
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "analytical_model.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file.json>" << std::endl;
        return 1;
    }

    std::string config_file = argv[1];

    try {
        AnalyticalModel model;
        model.configure(config_file);

        long long total_cycles = model.run_simulation();

        std::cout << "Simulation completed successfully!" << std::endl;
        std::cout << "Total execution time: " << total_cycles << " cycles" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}