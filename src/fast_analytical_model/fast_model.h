///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: fast_model.h
//  Description: Fast analytical NoC performance model
//  Created by:  Jakub Jastrzebski
//  Date:  23/09/2025
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _FAST_MODEL_H_
#define _FAST_MODEL_H_

#include <vector>
#include <unordered_map>
#include <string>
#include <queue>

#include "../restart/include/nlohmann/json.hpp"

// Architecture configuration
struct FastArchConfig {
    std::string topology;
    int k;  // network radix
    int n;  // network dimensions
    int flit_size;  // in bits
    int routing_delay;
    int vc_alloc_delay;
    int sw_alloc_delay;
    int st_prepare_delay;
    int st_final_delay;
    int speculative;
    double ANY_comp_cycles;

    // Defaults
    FastArchConfig() : topology("torus"), k(4), n(2), flit_size(64),
                        routing_delay(1), vc_alloc_delay(1), sw_alloc_delay(1),
                        st_prepare_delay(1), st_final_delay(1), speculative(1),
                        ANY_comp_cycles(0.25) {}
};

// Task structure
struct FastTask {
    int id;
    std::string type;  // "COMP_OP", "WRITE", "WRITE_REQ"
    std::vector<int> dependencies;

    // Communication fields
    int src = -1;
    int dst = -1;
    int size = 0;
    int pt_required = 0;

    // Computation fields
    int node = -1;
    int ct_required = 0;

    FastTask() = default;
};

// Main fast analytical model class
class FastAnalyticalModel {
private:
    FastArchConfig _arch;
    std::unordered_map<int, FastTask> _tasks_by_id;
    std::unordered_map<int, double> _completion_times;
    std::unordered_map<int, double> _node_available_times;  // node_id -> when node becomes available
    int _total_nodes;

public:
    FastAnalyticalModel();
    ~FastAnalyticalModel() = default;

    // Configuration
    void configure(const std::string& config_file);
    void configure_from_json(const nlohmann::json& config);

    // Main simulation
    int run_simulation();

    // Topology calculations
    int calculate_hop_distance(int src, int dst) const;
    double calculate_message_latency(int src, int dst, int size_flits) const;
    int calculate_num_flits(int size_bytes) const;

    // Task completion calculations
    double calculate_task_completion(const FastTask& task, double start_time) const;
    double calculate_comp_op_completion(const FastTask& task, double start_time) const;
    double calculate_write_completion(const FastTask& task, double start_time) const;
    double calculate_write_req_completion(const FastTask& task, double start_time) const;

    // Dependency management
    std::vector<int> topological_sort() const;
    double get_max_dependency_completion_time(const std::vector<int>& dependencies) const;
    double calculate_task_start_time(const FastTask& task) const;

    // Getters
    const FastArchConfig& get_arch_config() const { return _arch; }
    int get_total_simulation_time() const;

private:
    // Helper functions
    void parse_tasks_from_json(const nlohmann::json& workload);
    void calculate_total_nodes();
};

// High-level simulation function
int simulate_fast_analytical(const std::string& config_file);

#endif // _FAST_MODEL_H_