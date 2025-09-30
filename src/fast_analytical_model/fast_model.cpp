///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: fast_model.cpp
//  Description: Implementation of fast analytical NoC performance model
//  Created by:  Jakub Jastrzebski
//  Date:  23/09/2025
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "fast_model.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// FastAnalyticalModel Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////////

FastAnalyticalModel::FastAnalyticalModel() : _total_nodes(0) {
}

void FastAnalyticalModel::configure(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open configuration file: " + config_file);
    }

    json config;
    file >> config;
    configure_from_json(config);
}

void FastAnalyticalModel::configure_from_json(const json& config) {
    // Parse architecture parameters
    if (!config.contains("arch")) {
        throw std::runtime_error("Configuration missing 'arch' section");
    }

    const json& arch = config["arch"];
    if (arch.contains("topology")) _arch.topology = arch["topology"];
    if (arch.contains("k")) _arch.k = arch["k"];
    if (arch.contains("n")) _arch.n = arch["n"];
    if (arch.contains("flit_size")) _arch.flit_size = arch["flit_size"];
    if (arch.contains("routing_delay")) _arch.routing_delay = arch["routing_delay"];
    if (arch.contains("vc_alloc_delay")) _arch.vc_alloc_delay = arch["vc_alloc_delay"];
    if (arch.contains("sw_alloc_delay")) _arch.sw_alloc_delay = arch["sw_alloc_delay"];
    if (arch.contains("st_prepare_delay")) _arch.st_prepare_delay = arch["st_prepare_delay"];
    if (arch.contains("st_final_delay")) _arch.st_final_delay = arch["st_final_delay"];
    if (arch.contains("speculative")) _arch.speculative = arch["speculative"];
    if (arch.contains("ANY_comp_cycles")) _arch.ANY_comp_cycles = arch["ANY_comp_cycles"];

    calculate_total_nodes();

    // Parse workload
    if (!config.contains("workload")) {
        throw std::runtime_error("Configuration missing 'workload' section");
    }

    parse_tasks_from_json(config["workload"]);
}

void FastAnalyticalModel::calculate_total_nodes() {
    _total_nodes = 1;
    for (int i = 0; i < _arch.n; i++) {
        _total_nodes *= _arch.k;
    }
}

void FastAnalyticalModel::parse_tasks_from_json(const json& workload) {
    _tasks_by_id.clear();

    for (const auto& task_json : workload) {
        FastTask task;
        task.id = task_json["id"];
        task.type = task_json["type"];

        // Parse dependencies
        if (task_json.contains("dep")) {
            for (int dep_id : task_json["dep"]) {
                if (dep_id != -1) {
                    task.dependencies.push_back(dep_id);
                }
            }
        }

        // Parse task-specific fields
        if (task.type == "COMP_OP") {
            if (task_json.contains("node")) task.node = task_json["node"];
            if (task_json.contains("ct_required")) task.ct_required = task_json["ct_required"];
        } else if (task.type == "WRITE" || task.type == "WRITE_REQ") {
            if (task_json.contains("src")) task.src = task_json["src"];
            if (task_json.contains("dst")) task.dst = task_json["dst"];
            if (task_json.contains("size")) task.size = task_json["size"];
            if (task_json.contains("pt_required")) task.pt_required = task_json["pt_required"];
        }

        _tasks_by_id[task.id] = task;
    }
}

int FastAnalyticalModel::run_simulation() {
    _completion_times.clear();
    _node_available_times.clear();

    // Initialize all nodes as available at time 0
    for (int node = 0; node < _total_nodes; node++) {
        _node_available_times[node] = 0.0;
    }

    // Get topological order of tasks
    std::vector<int> topo_order = topological_sort();

    // Calculate completion time for each task in dependency order
    for (int task_id : topo_order) {
        const FastTask& task = _tasks_by_id[task_id];

        // Calculate start time considering both dependencies AND node availability
        double start_time = calculate_task_start_time(task);

        // Calculate completion time based on task type
        double completion_time = calculate_task_completion(task, start_time);

        _completion_times[task_id] = completion_time;

        // Update node availability based on task type and completion
        if (task.type == "COMP_OP") {
            // COMP_OP blocks the computation node until completion
            _node_available_times[task.node] = completion_time;
        } else if (task.type == "WRITE" || task.type == "WRITE_REQ") {
            // Communication tasks block the source node until they can start sending
            // (minimal blocking - just injection time)
            _node_available_times[task.src] = std::max(_node_available_times[task.src], start_time + 1.0);
        }
    }

    return get_total_simulation_time();
}

std::vector<int> FastAnalyticalModel::topological_sort() const {
    std::unordered_map<int, int> in_degree;
    std::unordered_map<int, std::vector<int>> adj_list;

    // Initialize in-degree for all task IDs
    for (const auto& [task_id, task] : _tasks_by_id) {
        in_degree[task_id] = 0;
    }

    // Build adjacency list using actual IDs
    for (const auto& [task_id, task] : _tasks_by_id) {
        for (int dep_id : task.dependencies) {
            if (_tasks_by_id.count(dep_id)) {
                adj_list[dep_id].push_back(task_id);
                in_degree[task_id]++;
            }
        }
    }

    // Kahn's algorithm
    std::queue<int> ready_queue;
    for (const auto& [task_id, degree] : in_degree) {
        if (degree == 0) {
            ready_queue.push(task_id);
        }
    }

    std::vector<int> topo_order;
    while (!ready_queue.empty()) {
        int current_id = ready_queue.front();
        ready_queue.pop();
        topo_order.push_back(current_id);

        for (int neighbor_id : adj_list[current_id]) {
            in_degree[neighbor_id]--;
            if (in_degree[neighbor_id] == 0) {
                ready_queue.push(neighbor_id);
            }
        }
    }

    if (topo_order.size() != _tasks_by_id.size()) {
        throw std::runtime_error("Circular dependency detected in task graph");
    }

    return topo_order;
}

double FastAnalyticalModel::get_max_dependency_completion_time(const std::vector<int>& dependencies) const {
    double max_time = 0.0;
    for (int dep_id : dependencies) {
        if (_completion_times.count(dep_id)) {
            max_time = std::max(max_time, _completion_times.at(dep_id));
        }
    }
    return max_time;
}

double FastAnalyticalModel::calculate_task_start_time(const FastTask& task) const {
    // Start time is the maximum of:
    // 1. All dependency completion times
    // 2. Node availability (for COMP_OP tasks that need the computation unit)

    double dep_time = get_max_dependency_completion_time(task.dependencies);

    // For COMP_OP tasks, also wait for the node to be available
    if (task.type == "COMP_OP") {
        double node_time = _node_available_times.at(task.node);
        return std::max(dep_time, node_time);
    }

    // For communication tasks, just wait for dependencies
    return dep_time;
}

double FastAnalyticalModel::calculate_task_completion(const FastTask& task, double start_time) const {
    if (task.type == "COMP_OP") {
        return calculate_comp_op_completion(task, start_time);
    } else if (task.type == "WRITE") {
        return calculate_write_completion(task, start_time);
    } else if (task.type == "WRITE_REQ") {
        return calculate_write_req_completion(task, start_time);
    } else {
        throw std::runtime_error("Unknown task type: " + task.type);
    }
}

double FastAnalyticalModel::calculate_comp_op_completion(const FastTask& task, double start_time) const {
    // Pure computation - no network communication
    double computation_cycles = task.ct_required * _arch.ANY_comp_cycles;
    return start_time + computation_cycles;
}

double FastAnalyticalModel::calculate_write_completion(const FastTask& task, double start_time) const {
    // WRITE protocol: WRITE → WRITE_ACK

    if (task.src == task.dst) {
        // Local write - no network communication
        return start_time + task.pt_required;
    } else {
        // Step 1: WRITE (src → dst, task.size)
        int write_flits = calculate_num_flits(task.size);
        double write_latency = calculate_message_latency(task.src, task.dst, write_flits);
        double write_processing = task.pt_required;

        // Step 2: WRITE_ACK (dst → src, 1 flit)
        double ack_latency = calculate_message_latency(task.dst, task.src, 1);
        double ack_processing = 1; // 1 cycle processing

        return start_time + write_latency + write_processing + ack_latency + ack_processing;
    }
}

double FastAnalyticalModel::calculate_write_req_completion(const FastTask& task, double start_time) const {
    // WRITE_REQ protocol: WRITE_REQ → READ_REQ → WRITE → WRITE_ACK

    if (task.src == task.dst) {
        // Local operation - just processing time
        return start_time + task.pt_required;
    }

    // Step 1: WRITE_REQ (src → dst, 1 flit)
    double step1_latency = calculate_message_latency(task.src, task.dst, 1);
    double step1_processing = 1; // 1 cycle processing

    // Step 2: READ_REQ (dst → src, 1 flit)
    double step2_latency = calculate_message_latency(task.dst, task.src, 1);
    double step2_processing = 1; // 1 cycle processing

    // Step 3: WRITE bulk data (src → dst, task.size)
    int bulk_flits = calculate_num_flits(task.size);
    double step3_latency = calculate_message_latency(task.src, task.dst, bulk_flits);
    double step3_processing = task.pt_required; // From JSON

    // Step 4: WRITE_ACK (dst → src, 1 flit)
    double step4_latency = calculate_message_latency(task.dst, task.src, 1);
    double step4_processing = 1; // 1 cycle processing

    return start_time + step1_latency + step1_processing +
            step2_latency + step2_processing +
            step3_latency + step3_processing +
            step4_latency + step4_processing;
}

int FastAnalyticalModel::calculate_hop_distance(int src, int dst) const {
    if (_arch.topology == "torus" || _arch.topology == "mesh") {
        int hops = 0;
        int temp_src = src;
        int temp_dst = dst;

        for (int dim = 0; dim < _arch.n; dim++) {
            int src_coord = temp_src % _arch.k;
            int dst_coord = temp_dst % _arch.k;

            int distance = std::abs(src_coord - dst_coord);

            // For torus, consider wraparound
            if (_arch.topology == "torus") {
                distance = std::min(distance, _arch.k - distance);
            }

            hops += distance;
            temp_src /= _arch.k;
            temp_dst /= _arch.k;
        }

        return hops;
    }

    // Default: Manhattan distance for unknown topologies
    return std::abs(src - dst);
}

double FastAnalyticalModel::calculate_message_latency(int src, int dst, int size_flits) const {
    // Number of hops
    int n_routers = calculate_hop_distance(src, dst);

    // Link traversal time
    int t_link = (_arch.topology == "mesh") ? 1 : 2;

    // Account for speculative routing
    int alloc_delay = _arch.speculative ?
        std::max(_arch.vc_alloc_delay, _arch.sw_alloc_delay) :
        (_arch.vc_alloc_delay + _arch.sw_alloc_delay);

    // Head flit processing time per hop
    int t_head_hop = _arch.routing_delay + alloc_delay +
                    _arch.st_prepare_delay + _arch.st_final_delay;

    // Body flit processing time per hop
    int t_body_hop = _arch.sw_alloc_delay + _arch.st_prepare_delay + _arch.st_final_delay;

    // Queuing delay (simplified)
    int queuing_delay = 1; // Assume 1 cycle queuing delay per hop

    // Total packet latency
    double T_packet = n_routers * (t_head_hop + t_link + queuing_delay) +
                      (size_flits - 1) * std::max(t_body_hop, t_link);

    return T_packet;
}

int FastAnalyticalModel::calculate_num_flits(int size_bytes) const {
    if (size_bytes <= 0) return 1;

    // Convert bytes to flits
    double scaling = 8.0 / _arch.flit_size;
    int flits = static_cast<int>(std::ceil(size_bytes * scaling));

    return std::max(1, flits);  // At least 1 flit
}

int FastAnalyticalModel::get_total_simulation_time() const {
    if (_completion_times.empty()) {
        return 0;
    }

    double max_time = 0.0;
    for (const auto& [task_id, completion_time] : _completion_times) {
        max_time = std::max(max_time, completion_time);
    }

    return static_cast<int>(std::ceil(max_time));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// High-level simulation function
///////////////////////////////////////////////////////////////////////////////////////////////////////////

int simulate_fast_analytical(const std::string& config_file) {
    FastAnalyticalModel model;
    model.configure(config_file);
    return model.run_simulation();
}