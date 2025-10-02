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
    int vc_buf_size;  // buffer size in flits
    int routing_delay;
    int vc_alloc_delay;
    int sw_alloc_delay;
    int st_prepare_delay;
    int st_final_delay;
    int speculative;
    double ANY_comp_cycles;

    // Queuing model parameters
    double C_S;  // Coefficient of variation for service time
    int traffic_burstiness_k;  // MMPP k parameter (λ₁/λ₀) for burstiness modeling

    // Defaults
    FastArchConfig() : topology("torus"), k(4), n(2), flit_size(64), vc_buf_size(8),
                        routing_delay(1), vc_alloc_delay(1), sw_alloc_delay(1),
                        st_prepare_delay(1), st_final_delay(1), speculative(1),
                        ANY_comp_cycles(0.25), C_S(0.5), traffic_burstiness_k(10) {}
};

// Port enumeration for routing
enum Port {
    PORT_NORTH = 0,
    PORT_EAST = 1,
    PORT_SOUTH = 2,
    PORT_WEST = 3,
    PORT_LOCAL = 4  // Injection/Ejection
};

// Routing hop structure
struct Hop {
    int router_id;
    int input_port;   // Port where packet enters router
    int output_port;  // Port where packet exits router

    Hop(int r, int in, int out) : router_id(r), input_port(in), output_port(out) {}
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

// Channel key for tracking traffic (router, input_port, output_port)
struct ChannelKey {
    int router;
    int input_port;
    int output_port;

    bool operator<(const ChannelKey& other) const {
        if (router != other.router) return router < other.router;
        if (input_port != other.input_port) return input_port < other.input_port;
        return output_port < other.output_port;
    }
};

// Output channel key (router, output_port)
struct OutputChannelKey {
    int router;
    int output_port;

    bool operator<(const OutputChannelKey& other) const {
        if (router != other.router) return router < other.router;
        return output_port < other.output_port;
    }
};

// Main fast analytical model class
class FastAnalyticalModel {
private:
    FastArchConfig _arch;
    std::unordered_map<int, FastTask> _tasks_by_id;
    std::unordered_map<int, double> _completion_times;
    std::unordered_map<int, double> _node_available_times;  // node_id -> when node becomes available
    int _total_nodes;

    // Traffic parameters for queuing model
    double _mu_service_rate;  // Average service rate (packets/cycle)
    double _C_A;              // Coefficient of variation for interarrival time (from MMPP)
    std::map<ChannelKey, double> _lambda_ic_oc;           // λ^N_{i→j}: arrival rate per IC→OC
    std::map<OutputChannelKey, double> _rho_total;        // ρ^N_j: total utilization per output channel

    // Per-channel service time parameters (computed dynamically from paper's method)
    std::map<OutputChannelKey, int> _channel_index;       // Channel group index (0=ejection, 1+=routing)
    std::map<OutputChannelKey, double> _service_time_mean; // μ: mean service time per channel
    std::map<OutputChannelKey, double> _service_time_C_S;  // C_S: service time CV per channel
    std::map<OutputChannelKey, double> _waiting_time;      // W: average waiting time per channel

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

    // Routing helpers
    std::pair<int, int> node_to_coords(int node_id) const;
    int coords_to_node(int x, int y) const;
    std::vector<Hop> trace_routing_path(int src, int dst) const;
    int get_opposite_port(int port) const;

    // Queuing model helpers
    double calculate_C_A_mmpp(int k_param) const;

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

    // Traffic parameter calculation
    void calculate_traffic_parameters();
    double calculate_average_packet_size_flits() const;
    double calculate_critical_path_time() const;
    void calculate_arrival_rates();
    void calculate_utilizations();
    int estimate_k_from_dependency_graph() const;

    // Per-channel service time calculation (paper's method)
    void calculate_channel_indices();
    void calculate_service_times_and_C_S();
    void initialize_ejection_channels();
    void calculate_group_service_times(int group_index);
    int get_max_distance_from_channel(int router, int port) const;
    std::map<OutputChannelKey, double> get_routing_probabilities(OutputChannelKey channel) const;
};

// High-level simulation function
int simulate_fast_analytical(const std::string& config_file);

#endif // _FAST_MODEL_H_