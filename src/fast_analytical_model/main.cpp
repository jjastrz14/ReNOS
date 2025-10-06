///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: main.cpp
//  Description: Main entry point for fast analytical model testing
//  Created by:  Claude Code
//  Date:  23/09/2025
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <chrono>
#include "fast_model.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file.json> [output_file]" << std::endl;
        std::cerr << "  output_file: - for stdout, empty for no output, or filename" << std::endl;
        return 1;
    }

    std::string config_file = argv[1];
    std::ostream* dump_file = &std::cout;
    std::ofstream file_stream;
    std::ostream null_stream(nullptr);

    // Handle optional output file argument
    if (argc >= 3) {
        std::string output_arg = argv[2];
        if (output_arg == "") {
            dump_file = &null_stream;
        } else if (output_arg == "-") {
            dump_file = &std::cout;
        } else {
            file_stream.open(output_arg);
            dump_file = &file_stream;
        }
    }

    try {
        *dump_file << "Fast Analytical NoC Simulator" << std::endl;
        *dump_file << "Configuration file: " << config_file << std::endl;
        *dump_file << std::endl;

        // Measure execution time
        auto start = std::chrono::high_resolution_clock::now();

        // Run simulation
        int simulation_time = simulate_fast_analytical(config_file, dump_file);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end - start);

        *dump_file << "Simulation completed!" << std::endl;
        *dump_file << "Total simulation time: " << simulation_time << " cycles" << std::endl;
        *dump_file << "Execution time: " << duration.count() << " seconds" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}