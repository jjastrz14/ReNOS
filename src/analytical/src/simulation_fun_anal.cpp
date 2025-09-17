///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: simulation_fun_anal.cpp
//  Description: Implementation of high-level simulation functions for analytical model
//  Created by:  Jakub Jastrzebski
//  Date:  15/09/2025
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "simulation_fun_anal.hpp"
#include "analytical_model.hpp"
#include "../include/nlohmann/json.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/time.h>
#include <pybind11/pybind11.h>

std::tuple<int, AnalyticalLogger*> SimulateAnalytical(const std::string& config_file,
                                                     const std::string& output_file) {
    // Static storage for model and logger to ensure they persist
    static std::unique_ptr<AnalyticalModel> static_model;

    try {
        // Release GIL like BookSim2 does
        pybind11::gil_scoped_release release;

        // Create and configure the analytical model
        static_model = CreateAnalyticalModel(config_file);

        // Set up output stream
        std::ofstream file_stream;
        std::ofstream null_stream;  // Create a null stream
        std::ostream* output_stream = &std::cout;

        if (output_file == "") {
            // No output - use null stream (like BookSim2's nullStream)
            null_stream.setstate(std::ios_base::badbit);  // Make stream invalid to discard output
            output_stream = &null_stream;
        } else if (output_file == "-") {
            output_stream = &std::cout;
        } else {
            file_stream.open(output_file);
            if (!file_stream.is_open()) {
                throw std::runtime_error("Cannot open output file: " + output_file);
            }
            output_stream = &file_stream;
        }

        static_model->set_output_file(output_stream);

        // Measure execution time
        struct timeval start_time, end_time;
        gettimeofday(&start_time, nullptr);

        // Run the simulation
        int result = static_model->run_simulation();

        gettimeofday(&end_time, nullptr);
        double execution_time = ((double)(end_time.tv_sec) + (double)(end_time.tv_usec) / 1000000.0) -
                               ((double)(start_time.tv_sec) + (double)(start_time.tv_usec) / 1000000.0);

        *output_stream << "Analytical simulation execution time: " << execution_time << " seconds" << std::endl;

        // Re-acquire GIL like BookSim2 does
        pybind11::gil_scoped_acquire acquire;

        // Return the logger pointer directly (it's managed by the static model)
        // AnalyticalLogger* logger = static_model->get_logger();
        AnalyticalLogger* logger = nullptr;  // Disable logger for now

        return std::make_tuple(result, logger);

    } catch (const std::exception& e) {
        std::cerr << "Error in analytical simulation: " << e.what() << std::endl;
        return std::make_tuple(-1, nullptr);
    }
}

std::unique_ptr<AnalyticalModel> CreateAnalyticalModel(const std::string& config_file) {
    auto model = std::make_unique<AnalyticalModel>();
    model->configure(config_file);
    return model;
}

int RunAnalyticalSimulation(AnalyticalModel* model, const std::string& output_file) {
    if (!model) {
        std::cerr << "Error: Model pointer is null" << std::endl;
        return -1;
    }

    try {
        // Set up output stream
        std::ofstream file_stream;
        std::ostream* output_stream = &std::cout;

        if (!output_file.empty() && output_file != "-") {
            file_stream.open(output_file);
            if (!file_stream.is_open()) {
                throw std::runtime_error("Cannot open output file: " + output_file);
            }
            output_stream = &file_stream;
        }

        model->set_output_file(output_stream);

        // Run simulation
        return model->run_simulation();

    } catch (const std::exception& e) {
        std::cerr << "Error running analytical simulation: " << e.what() << std::endl;
        return -1;
    }
}

bool ValidateAnalyticalConfig(const std::string& config_file) {
    try {
        std::ifstream file(config_file);
        if (!file.is_open()) {
            std::cerr << "Cannot open configuration file: " << config_file << std::endl;
            return false;
        }

        nlohmann::json config;
        file >> config;

        // Check for required sections
        if (!config.contains("arch")) {
            std::cerr << "Configuration missing 'arch' section" << std::endl;
            return false;
        }

        // Validate architecture parameters
        const auto& arch = config["arch"];

        // Check topology
        if (arch.contains("topology")) {
            std::string topology = arch["topology"];
            if (topology != "torus" && topology != "mesh") {
                std::cerr << "Warning: Topology '" << topology << "' may not be fully supported" << std::endl;
            }
        }

        // Check required dimensions
        if (!arch.contains("k") || !arch.contains("n")) {
            std::cerr << "Configuration missing required topology dimensions (k, n)" << std::endl;
            return false;
        }

        int k = arch["k"];
        int n = arch["n"];
        if (k <= 0 || n <= 0) {
            std::cerr << "Invalid topology dimensions: k=" << k << ", n=" << n << std::endl;
            return false;
        }

        // Check timing parameters are non-negative
        std::vector<std::string> timing_params = {
            "routing_delay", "vc_alloc_delay", "sw_alloc_delay",
            "st_prepare_delay", "st_final_delay"
        };

        for (const auto& param : timing_params) {
            if (arch.contains(param)) {
                int value = arch[param];
                if (value < 0) {
                    std::cerr << "Invalid timing parameter " << param << ": " << value << std::endl;
                    return false;
                }
            }
        }

        // Check flit size
        if (arch.contains("flit_size")) {
            int flit_size = arch["flit_size"];
            if (flit_size <= 0) {
                std::cerr << "Invalid flit_size: " << flit_size << std::endl;
                return false;
            }
        }

        // Validate packets section if present
        if (config.contains("packets")) {
            const auto& packets = config["packets"];
            if (!packets.is_array()) {
                std::cerr << "Packets section must be an array" << std::endl;
                return false;
            }

            int total_nodes = 1;
            for (int i = 0; i < n; i++) {
                total_nodes *= k;
            }

            for (size_t i = 0; i < packets.size(); ++i) {
                const auto& packet = packets[i];

                if (!packet.contains("id") || !packet.contains("src") ||
                    !packet.contains("dst") || !packet.contains("size")) {
                    std::cerr << "Packet " << i << " missing required fields (id, src, dst, size)" << std::endl;
                    return false;
                }

                int src = packet["src"];
                int dst = packet["dst"];
                int size = packet["size"];

                if (src < 0 || src >= total_nodes || dst < 0 || dst >= total_nodes) {
                    std::cerr << "Packet " << i << " has invalid src/dst: " << src << "->" << dst
                              << " (total nodes: " << total_nodes << ")" << std::endl;
                    return false;
                }

                if (size <= 0) {
                    std::cerr << "Packet " << i << " has invalid size: " << size << std::endl;
                    return false;
                }
            }
        }

        std::cout << "Configuration file validation passed" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error validating configuration: " << e.what() << std::endl;
        return false;
    }
}

std::string GetAnalyticalModelInfo() {
    std::ostringstream info;

    info << "=== Analytical NoC Performance Model ===" << std::endl;
    info << "Version: 1.0.0" << std::endl;
    info << "Description: Fast latency estimation for Network-on-Chip architectures" << std::endl;
    info << std::endl;
    info << "Features:" << std::endl;
    info << "- Dependency-based packet injection" << std::endl;
    info << "- Analytical latency calculation using router pipeline model" << std::endl;
    info << "- Support for torus and mesh topologies" << std::endl;
    info << "- Configurable router microarchitecture parameters" << std::endl;
    info << "- Event logging and statistics collection" << std::endl;
    info << "- Python bindings for integration with optimization algorithms" << std::endl;
    info << std::endl;
    info << "Latency Model:" << std::endl;
    info << "  T_packet = n_routers * (t_head_hop + t_link + queuing_delay) + " << std::endl;
    info << "             (n_flits - 1) * max(t_body_hop, t_link)" << std::endl;
    info << "  where:" << std::endl;
    info << "    t_head_hop = routing_delay + vc_alloc_delay + sw_alloc_delay + " << std::endl;
    info << "                 st_prepare_delay + st_final_delay" << std::endl;
    info << "    t_body_hop = sw_alloc_delay + st_prepare_delay + st_final_delay" << std::endl;
    info << "    t_link = 1 cycle (assumed)" << std::endl;
    info << "    queuing_delay = 0 (not currently modeled)" << std::endl;
    info << std::endl;
    info << "Supported Configurations:" << std::endl;
    info << "- Topology: torus, mesh (k-ary n-dimensional)" << std::endl;
    info << "- Router delays: routing, VC allocation, switch allocation, crossbar traversal" << std::endl;
    info << "- Packet types: READ_REQ, WRITE_REQ with dependency tracking" << std::endl;
    info << "- Processing element workloads with computation time modeling" << std::endl;

    return info.str();
}