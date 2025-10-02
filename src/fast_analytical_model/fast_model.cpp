///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: fast_model.cpp
//  Description: Implementation of fast analytical NoC performance model
//  Created by:  Jakub Jastrzebski
//  Date:  23/09/2025
//  all references to Paper refer to:
//  "Abbas Eslami Kiasari, Zhonghai Lu, Member, and Axel Jantsch 
//  An Analytical Latency Model for Networks-on-Chip 
//  10.1109/TVLSI.2011.2178620"
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
    if (arch.contains("vc_buf_size")) _arch.vc_buf_size = arch["vc_buf_size"];
    if (arch.contains("routing_delay")) _arch.routing_delay = arch["routing_delay"];
    if (arch.contains("vc_alloc_delay")) _arch.vc_alloc_delay = arch["vc_alloc_delay"];
    if (arch.contains("sw_alloc_delay")) _arch.sw_alloc_delay = arch["sw_alloc_delay"];
    if (arch.contains("st_prepare_delay")) _arch.st_prepare_delay = arch["st_prepare_delay"];
    if (arch.contains("st_final_delay")) _arch.st_final_delay = arch["st_final_delay"];
    if (arch.contains("speculative")) _arch.speculative = arch["speculative"];
    if (arch.contains("ANY_comp_cycles")) _arch.ANY_comp_cycles = arch["ANY_comp_cycles"];
    if (arch.contains("C_S")) _arch.C_S = arch["C_S"];
    if (arch.contains("traffic_burstiness_k")) _arch.traffic_burstiness_k = arch["traffic_burstiness_k"];

    calculate_total_nodes();

    // Parse workload
    if (!config.contains("workload")) {
        throw std::runtime_error("Configuration missing 'workload' section");
    }

    parse_tasks_from_json(config["workload"]);

    // Calculate traffic parameters for queuing model
    calculate_traffic_parameters();
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Routing Helper Functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////

std::pair<int, int> FastAnalyticalModel::node_to_coords(int node_id) const {
    // Convert node ID to (x, y) coordinates
    // Assumes row-major ordering: node_id = y * k + x
    int x = node_id % _arch.k;
    int y = node_id / _arch.k;
    return {x, y};
}

int FastAnalyticalModel::coords_to_node(int x, int y) const {
    // Convert (x, y) coordinates to node ID
    return y * _arch.k + x;
}

int FastAnalyticalModel::get_opposite_port(int port) const {
    // Return the opposite direction port
    switch (port) {
        case PORT_NORTH: return PORT_SOUTH;
        case PORT_SOUTH: return PORT_NORTH;
        case PORT_EAST:  return PORT_WEST;
        case PORT_WEST:  return PORT_EAST;
        case PORT_LOCAL: return PORT_LOCAL;
        default: return PORT_LOCAL;
    }
}

// Here we implement dimension-order routing for 2D torus/mesh networks (XY routing)
// as the preliminary study is okay, but feel free to extend for other routing algorithms / topologies
std::vector<Hop> FastAnalyticalModel::trace_routing_path(int src, int dst) const {
    std::vector<Hop> path;

    if (src == dst) {
        // Local communication - no hops through network
        return path;
    }

    // Only support 2D torus/mesh for now
    if (_arch.n != 2) {
        throw std::runtime_error("Routing only implemented for 2D networks (n=2)");
    }

    auto [src_x, src_y] = node_to_coords(src);
    auto [dst_x, dst_y] = node_to_coords(dst);

    int current_node = src;
    int current_x = src_x;
    int current_y = src_y;
    int input_port = PORT_LOCAL;  // Start from injection port

    // Dimension-order routing: X dimension first, then Y dimension

    // Route in X dimension
    while (current_x != dst_x) {
        int output_port;
        int next_x = current_x;

        if (_arch.topology == "torus") {
            // For torus, choose shortest path (may wrap around)
            int dist_east = (dst_x - current_x + _arch.k) % _arch.k;
            int dist_west = (current_x - dst_x + _arch.k) % _arch.k;

            if (dist_east <= dist_west) {
                output_port = PORT_EAST;
                next_x = (current_x + 1) % _arch.k;
            } else {
                output_port = PORT_WEST;
                next_x = (current_x - 1 + _arch.k) % _arch.k;
            }
        } else {
            // For mesh, simple comparison
            if (dst_x > current_x) {
                output_port = PORT_EAST;
                next_x = current_x + 1;
            } else {
                output_port = PORT_WEST;
                next_x = current_x - 1;
            }
        }

        // Add hop
        path.push_back(Hop(current_node, input_port, output_port));

        // Move to next node
        current_x = next_x;
        current_node = coords_to_node(current_x, current_y);
        input_port = get_opposite_port(output_port);
    }

    // Route in Y dimension
    while (current_y != dst_y) {
        int output_port;
        int next_y = current_y;

        if (_arch.topology == "torus") {
            // For torus, choose shortest path (may wrap around)
            int dist_north = (dst_y - current_y + _arch.k) % _arch.k;
            int dist_south = (current_y - dst_y + _arch.k) % _arch.k;

            if (dist_north <= dist_south) {
                output_port = PORT_NORTH;
                next_y = (current_y + 1) % _arch.k;
            } else {
                output_port = PORT_SOUTH;
                next_y = (current_y - 1 + _arch.k) % _arch.k;
            }
        } else {
            // For mesh, simple comparison
            if (dst_y > current_y) {
                output_port = PORT_NORTH;
                next_y = current_y + 1;
            } else {
                output_port = PORT_SOUTH;
                next_y = current_y - 1;
            }
        }

        // Add hop
        path.push_back(Hop(current_node, input_port, output_port));

        // Move to next node
        current_y = next_y;
        current_node = coords_to_node(current_x, current_y);
        input_port = get_opposite_port(output_port);
    }

    // Final hop: enter destination node and eject
    path.push_back(Hop(current_node, input_port, PORT_LOCAL));

    return path;
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
    // Local communication - no network hops
    if (src == dst) {
        return 0.0;
    }

    // Trace the routing path
    std::vector<Hop> path = trace_routing_path(src, dst);

    if (path.empty()) {
        return 0.0;  // No hops (shouldn't happen if src != dst)
    }

    // Link traversal time
    int t_link = (_arch.topology == "mesh") ? 1 : 2;

    // Account for speculative routing
    int alloc_delay = _arch.speculative ?
        std::max(_arch.vc_alloc_delay, _arch.sw_alloc_delay) :
        (_arch.vc_alloc_delay + _arch.sw_alloc_delay);

    // Router delays
    int t_r = _arch.routing_delay;
    int t_s = alloc_delay + _arch.st_prepare_delay + _arch.st_final_delay;
    int t_body_hop = _arch.sw_alloc_delay + _arch.st_prepare_delay + _arch.st_final_delay;

    // Head flit latency calculation (equation 5 from paper)
    // L_h = t_inj + Σ(t_r + W^N_{i→j} + t_s + t_w) + t_ej

    double L_h = 0.0;

    // Injection delay (first hop uses LOCAL port as input)
    // No waiting time for injection
    L_h += t_r + t_s;

    // For each hop along the path
    for (const auto& hop : path) {
        // Calculate waiting time W^N_{i→j} using equation (13) + (12)
        // W^N_{i→j} = [ρ^N_j * (C²_A + C²_S)] / [2(μ^N_j - λ^N_{i→j})]

        double W = 0.0;  // Default no waiting

        ChannelKey ch_key{hop.router_id, hop.input_port, hop.output_port};
        OutputChannelKey out_key{hop.router_id, hop.output_port};

        // Look up pre-computed traffic parameters
        auto lambda_it = _lambda_ic_oc.find(ch_key);
        auto rho_it = _rho_total.find(out_key);

        if (lambda_it != _lambda_ic_oc.end() && rho_it != _rho_total.end()) {
            double lambda_ij = lambda_it->second;
            double rho_j = rho_it->second;

            // Check if channel is not saturated
            double denominator = _mu_service_rate - lambda_ij;

            if (denominator > 0.0001 && rho_j < 0.99) {
                // Apply waiting time formula (equation 13 for i=1)
                // W^N_{i→j} = [ρ^N_j * (C²_A + C²_S)] / [2(μ - λ_{i→j})]
                // C_A from MMPP, C_S from per-channel calculation (paper's equations 16-19)

                // Get channel-specific C_S (if available, otherwise fallback to config default)
                double C_S_channel = _arch.C_S;  // Default fallback
                auto cs_it = _service_time_C_S.find(out_key);
                if (cs_it != _service_time_C_S.end()) {
                    C_S_channel = cs_it->second;  // Use dynamically calculated C_S
                }
                else {
                    std::cout << "DEBUG: Using default C_S for channel at router " << hop.router_id
                    << " port " << hop.output_port << std::endl;
                }

                double numerator = rho_j * (_C_A * _C_A + C_S_channel * C_S_channel);
                W = numerator / (2.0 * denominator);
                //std::cout << "DEBUG: Channel at router " << hop.router_id
                //        << " port " << hop.output_port
                //        << " λ=" << lambda_ij
                //        << " ρ=" << rho_j
                //        << " W=" << W << std::endl;
            } else {
                // Channel saturated - use large delay
                std::cout << "WARN: Channel saturated at router " << hop.router_id
                        << " port " << hop.output_port << ", setting high waiting time." << std::endl;
                W = 100.0;
            }
        }

        // Add hop delay: routing + waiting + switch + wire
        if (hop.output_port == PORT_LOCAL) {
            // Last hop - ejection, no wire delay
            L_h += t_r + W + t_s;
        } else {
            // Regular hop
            L_h += t_r + W + t_s + t_link;
        }
    }

    // Body flit latency (equation 6 from paper)
    // L_b = (m - 1) × max(t_s, t_w)
    double L_b = (size_flits - 1) * std::max(t_body_hop, t_link);

    // Total packet latency
    double T_packet = L_h + L_b;

    return T_packet;
}

int FastAnalyticalModel::calculate_num_flits(int size_bytes) const {
    if (size_bytes <= 0) return 1;

    // Convert bytes to flits
    double scaling = 8.0 / _arch.flit_size;
    int flits = static_cast<int>(std::ceil(size_bytes * scaling));

    return std::max(1, flits);  // At least 1 flit
}

double FastAnalyticalModel::calculate_C_A_mmpp(int k_param) const {
    // MMPP-based C_A calculation using Table II from the paper
    // k = lambda_1 / lambda_0 (ratio of high to low arrival rates)
    // This table maps k values to C_A (coefficient of variation of interarrival time)

    // Table II from paper: MMPP parameters and resulting C_A values
    // additional values estimated for finer granularity via fitting a power function
    // Format: {k, C_A}
    static const std::vector<std::pair<int, double>> ca_table = {
        {1, 1.0},    // k=1: Poisson (exponential interarrival) -> C_A = 1.0
        {5, 1.16},  // k=5
        {10, 1.55},  // k=10
        {15, 1.882},  // k=15
        {20, 2.04},  // k=20
        {25, 2.357},  // k=25
        {30, 2.554}, // k=30
        {35, 2.734}, // k=35
        {40, 2.899},  // k=40
        {45, 3.054},  // k=45
        {50, 3.08},  // k=50
        {55, 3.336}, // k=55
        {60, 3.466},  // k=60
        {65, 3.591},  // k=65
        {70, 3.710},  // k=70
        {75, 3.824},  // k=75
        {80, 3.934},  // k=80
        {85, 4.041},  // k=85
        {90, 4.144},  // k=90
        {95, 4.244},  // k=95
        {100, 4.28},  // k=100
        {105, 4.435},  // k=105
        {110, 4.527},  // k=110
        {115, 4.616},  // k=115
        {120, 4.704},  // k=120
        {125, 4.789},  // k=125
        {130, 4.873},  // k=130
        {135, 4.954},  // k=135
        {140, 5.034},  // k=140
        {145, 5.113},  // k=145
        {150, 5.190},  // k=150
        {155, 5.265},  // k=155
        {160, 5.339},  // k=160
        {165, 5.412},  // k=165
        {170, 5.484},  // k=170
        {175, 5.554},  // k=175
        {180, 5.624},  // k=180
        {185, 5.692},  // k=185
        {190, 5.759},  // k=190
        {195, 5.826},  // k=195
        {200, 6.00}    // k=200
    };

    // Find closest k value in table
    if (k_param <= 1) {
        return 1.0;  // Poisson traffic
    }
    // For k > 300, use linear scaling to handle network congestion
    else if (k_param > 300) {
        // Linear model: C_A = 1.44982296 + 0.034281 * k
        double C_A_linear = 1.44982296 + 0.0325 * k_param;
        std::cout << "WARN: calculate_C_A_mmpp: k_param = " << k_param
                << " indicates severe network congestion. Using linear C_A scaling (C_A = "
                << C_A_linear << ") as artificial workaround. "
                << "This solution should NOT be used for accurate modeling." << std::endl;
        return C_A_linear;
    }
    // For 200 < k <= 300, use power law fit from curve fitting
    else if (k_param > 200) {
        // Power law model: C_A = 0.57092083 * k^0.44049964
        double C_A_extrapolated = 0.57092083 * pow(k_param, 0.44049964);
        std::cout << "INFO: calculate_C_A_mmpp: k_param = " << k_param
                << " exceeds table range (max=200), using power law extrapolation: C_A = "
                << C_A_extrapolated << std::endl;
        return C_A_extrapolated;
    }

    // Round k_param to nearest multiple of 5 for finer interpolation
    int k_rounded = ((k_param + 2) / 5) * 5;
    if (k_rounded < 1) k_rounded = 1;
    if (k_rounded > 200) k_rounded = 200;

    // Find the bounding entries in the table
    for (size_t i = 0; i < ca_table.size() - 1; i++) {
        if (k_rounded >= ca_table[i].first && k_rounded <= ca_table[i+1].first) {
            // Linear interpolation between table entries
            double k1 = ca_table[i].first;
            double k2 = ca_table[i+1].first;
            double ca1 = ca_table[i].second;
            double ca2 = ca_table[i+1].second;

            double t = (k_rounded - k1) / (k2 - k1);
            return ca1 + t * (ca2 - ca1);
        }
    }

    // Exact match case (k_rounded == 1, 10, 20, 50, 100, or 200)
    for (const auto& entry : ca_table) {
        if (k_rounded == entry.first) {
            return entry.second;
        }
    }

    return 1.0;  // Default to Poisson
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
// Traffic Parameter Calculation
///////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastAnalyticalModel::calculate_traffic_parameters() {
    // Estimate k from dependency graph if not explicitly set
    int k_param = _arch.traffic_burstiness_k;

    // If k is set to a special value (e.g., -1 or 0), estimate from graph
    if (k_param <= 0) {
        k_param = estimate_k_from_dependency_graph();
        std::cout << "Estimated k from dependency graph: " << k_param << std::endl;
    }

    // Pre-compute C_A from MMPP
    _C_A = calculate_C_A_mmpp(k_param);

    // Calculate average service rate μ
    double avg_packet_flits = calculate_average_packet_size_flits();
    int t_body_hop = _arch.sw_alloc_delay + _arch.st_prepare_delay + _arch.st_final_delay;
    int t_link = (_arch.topology == "mesh") ? 1 : 2;
    double avg_service_time = avg_packet_flits * std::max(t_body_hop, t_link);

    if (avg_service_time > 0) {
        _mu_service_rate = 1.0 / avg_service_time;
    } else {
        _mu_service_rate = 1.0;  // Default
    }

    // Calculate arrival rates λ^N_{i→j}
    calculate_arrival_rates();

    // Calculate total utilizations ρ^N_j
    calculate_utilizations();

    // Calculate per-channel service times and C_S using paper's method
    calculate_channel_indices();
    initialize_ejection_channels();
    calculate_service_times_and_C_S();

    // C_A is interarrival time variability coefficient from MMPP
    // C_S is now calculated per-channel based on paper's equations 16-19

    std::cout << "Traffic parameters calculated:" << std::endl;
    std::cout << "  C_A = " << _C_A << std::endl;
    std::cout << "  C_S from arch.json = " << _arch.C_S << std::endl;
    std::cout << "  μ (service rate) = " << _mu_service_rate << " packets/cycle" << std::endl;
    std::cout << "  Average packet size = " << avg_packet_flits << " flits" << std::endl;
    std::cout << "  Number of IC→OC channels with traffic = " << _lambda_ic_oc.size() << std::endl;
}

double FastAnalyticalModel::calculate_average_packet_size_flits() const {
    double total_flits = 0.0;
    int num_comm_tasks = 0;

    for (const auto& [task_id, task] : _tasks_by_id) {
        if (task.type == "WRITE" || task.type == "WRITE_REQ") {
            int flits = calculate_num_flits(task.size);
            total_flits += flits;
            num_comm_tasks++;
        }
    }

    if (num_comm_tasks > 0) {
        return total_flits / num_comm_tasks;
    }

    return 1.0;  // Default to 1 flit if no communication tasks
}

double FastAnalyticalModel::calculate_critical_path_time() const {
    // Calculate earliest start and finish times for each task
    // considering dependency graph and parallel execution
    std::map<int, double> earliest_start;
    std::map<int, double> earliest_finish;

    // Process tasks in dependency order (topological sort implicit through iteration)
    // We may need multiple passes to resolve all dependencies
    bool changed = true;
    int max_iterations = _tasks_by_id.size() + 10;  // Safety limit
    int iteration = 0;

    while (changed && iteration < max_iterations) {
        changed = false;
        iteration++;

        for (const auto& [task_id, task] : _tasks_by_id) {
            double start_time = 0.0;

            // Task can't start until all dependencies finish
            for (int dep_id : task.dependencies) {
                if (dep_id >= 0 && earliest_finish.count(dep_id) > 0) {
                    start_time = std::max(start_time, earliest_finish[dep_id]);
                }
            }

            // Update if changed
            if (earliest_start.count(task_id) == 0 || earliest_start[task_id] != start_time) {
                earliest_start[task_id] = start_time;

                // Calculate finish time based on task type
                double exec_time = task.ct_required;  // Computation time

                // Communication tasks also have transmission time
                if (task.type == "WRITE" || task.type == "WRITE_REQ") {
                    // Use pt_required if available, otherwise estimate
                    exec_time += (task.pt_required > 0) ? task.pt_required : task.size;
                }

                earliest_finish[task_id] = start_time + exec_time;
                changed = true;
            }
        }
    }

    // Critical path is the maximum finish time across all tasks
    double critical_path = 0.0;
    for (const auto& [task_id, finish_time] : earliest_finish) {
        critical_path = std::max(critical_path, finish_time);
    }

    // Ensure non-zero critical path
    if (critical_path <= 0.0) {
        std::cout << "WARN: calculate_critical_path_time: critical path is zero or negative, using default=1000.0" << std::endl;
        return 1000.0;
    }

    return critical_path;
}

void FastAnalyticalModel::calculate_arrival_rates() {
    _lambda_ic_oc.clear();

    // Count communication tasks
    int num_comm_tasks = 0;
    for (const auto& [task_id, task] : _tasks_by_id) {
        if ((task.type == "WRITE" || task.type == "WRITE_REQ") && task.src != task.dst) {
            num_comm_tasks++;
        }
    }

    if (num_comm_tasks == 0) {
        return;  // No network communication
    }

    // Calculate total execution time (critical path through dependency graph)
    double critical_path_time = calculate_critical_path_time();

    std::cout << "INFO: Critical path time = " << critical_path_time << " cycles" << std::endl;

    // Step 1: Accumulate relative traffic (packet counts) per channel
    std::map<ChannelKey, double> packet_counts;
    double total_packets = 0.0;

    for (const auto& [task_id, task] : _tasks_by_id) {
        if (task.type == "WRITE" || task.type == "WRITE_REQ") {
            if (task.src == task.dst) {
                // Local communication - no network hops
                continue;
            }

            // Trace routing path
            std::vector<Hop> path = trace_routing_path(task.src, task.dst);

            // Count packets on each channel
            for (const auto& hop : path) {
                ChannelKey key{hop.router_id, hop.input_port, hop.output_port};
                packet_counts[key] += 1.0;
                total_packets += 1.0;
            }

            // WRITE_REQ has additional reply packets (ACKs)
            if (task.type == "WRITE_REQ") {
                // Account for reverse path traffic (4 packets: WRITE_REQ, READ_REQ, WRITE, ACK)
                std::vector<Hop> reverse_path = trace_routing_path(task.dst, task.src);

                for (const auto& hop : reverse_path) {
                    ChannelKey key{hop.router_id, hop.input_port, hop.output_port};
                    packet_counts[key] += 3.0;  // 3 reverse packets per WRITE_REQ
                    total_packets += 3.0;
                }
            }
        }
    }

    // Step 2: Convert packet counts to arrival rates using critical path time
    // λ = (number of packets) / (critical path time - without contention)
    // Higher traffic will naturally lead to higher λ and thus higher waiting times
    for (const auto& [key, count] : packet_counts) {
        _lambda_ic_oc[key] = count / critical_path_time;
    }

    // Step 4: Check if any OUTPUT CHANNEL would saturate (sum of all input flows)
    std::map<OutputChannelKey, double> total_lambda_per_output;
    for (const auto& [key, lambda] : _lambda_ic_oc) {
        OutputChannelKey out_key{key.router, key.output_port};
        total_lambda_per_output[out_key] += lambda;
    }

    double max_utilization = 0.0;
    for (const auto& [out_key, total_lambda] : total_lambda_per_output) {
        double util = total_lambda / _mu_service_rate;
        max_utilization = std::max(max_utilization, util);
    }

    std::cout << "INFO: Max utilization = " << max_utilization << std::endl;

    // Only scale if extremely high to prevent numerical instability
    // Higher threshold (0.98 vs 0.85) allows queueing formula to capture more contention
    if (max_utilization > 0.98) {
        double scale_factor = 0.98 / max_utilization;
        std::cout << "INFO: Scaling arrival rates by " << scale_factor
                << " for numerical stability (max_util was " << max_utilization << ")" << std::endl;
        for (auto& [key, lambda] : _lambda_ic_oc) {
            lambda *= scale_factor;
        }
    }
}

void FastAnalyticalModel::calculate_utilizations() {
    _rho_total.clear();

    // For each output channel, sum arrival rates from all input channels
    for (const auto& [channel_key, lambda_ij] : _lambda_ic_oc) {
        OutputChannelKey out_key{channel_key.router, channel_key.output_port};
        _rho_total[out_key] += lambda_ij;
    }

    // Convert total arrival rate to utilization (ρ = λ / μ)
    for (auto& [out_key, lambda_total] : _rho_total) {
        lambda_total = lambda_total / _mu_service_rate;

        // Cap utilization at 0.99 to avoid division by zero
        if (lambda_total >= 0.99) {
            std::cout << "WARN: High utilization (" << lambda_total
                        << ") at router " << out_key.router
                        << " port " << out_key.output_port << std::endl;
            lambda_total = 0.99;
        }
    }
}

int FastAnalyticalModel::estimate_k_from_dependency_graph() const {
    // Estimate k (burstiness parameter) from dependency graph structure
    // k = λ₁/λ₀ represents ratio of high vs low arrival rates in MMPP

    if (_tasks_by_id.empty()) {
        return 10;  // Default moderate burstiness
    }

    // Metric 1: Calculate parallelism variance
    // - High parallelism → many tasks at same depth → bursty traffic
    // - Low parallelism → sequential tasks → smooth traffic

    std::unordered_map<int, int> depth_map;  // task_id → depth
    std::unordered_map<int, int> depth_count; // depth → num_tasks

    // Calculate depth for each task (longest path from root)
    std::function<int(int)> get_depth = [&](int task_id) -> int {
        if (depth_map.count(task_id)) {
            return depth_map[task_id];
        }

        const auto& task = _tasks_by_id.at(task_id);
        int max_dep_depth = 0;

        for (int dep_id : task.dependencies) {
            if (_tasks_by_id.count(dep_id)) {
                max_dep_depth = std::max(max_dep_depth, get_depth(dep_id) + 1);
            }
        }

        depth_map[task_id] = max_dep_depth;
        return max_dep_depth;
    };

    // Calculate depth for all tasks
    for (const auto& [task_id, task] : _tasks_by_id) {
        int depth = get_depth(task_id);
        depth_count[depth]++;
    }

    // Calculate parallelism statistics
    if (depth_count.empty()) {
        return 10;
    }

    int max_parallelism = 0;
    int min_parallelism = INT_MAX;
    double avg_parallelism = 0.0;

    for (const auto& [depth, count] : depth_count) {
        max_parallelism = std::max(max_parallelism, count);
        min_parallelism = std::min(min_parallelism, count);
        avg_parallelism += count;
    }
    avg_parallelism /= depth_count.size();

    // Metric 2: Communication pattern variance
    // Count messages per source node
    std::unordered_map<int, int> msgs_per_src;
    int total_comm_tasks = 0;

    for (const auto& [task_id, task] : _tasks_by_id) {
        if ((task.type == "WRITE" || task.type == "WRITE_REQ") && task.src != task.dst) {
            msgs_per_src[task.src]++;
            total_comm_tasks++;
        }
    }

    double comm_variance = 0.0;
    if (total_comm_tasks > 0 && !msgs_per_src.empty()) {
        double avg_msgs = static_cast<double>(total_comm_tasks) / msgs_per_src.size();

        for (const auto& [src, count] : msgs_per_src) {
            double diff = count - avg_msgs;
            comm_variance += diff * diff;
        }
        comm_variance /= msgs_per_src.size();
    }

    // Estimate k based on metrics
    // Higher parallelism variation → higher k
    // Higher communication variance → higher k

    double parallelism_ratio = (min_parallelism > 0) ?
        static_cast<double>(max_parallelism) / min_parallelism : 1.0;

    double comm_cv = (avg_parallelism > 0) ?
        std::sqrt(comm_variance) / avg_parallelism : 0.0;

    // Continuous mapping to k values (can be any value, interpolated in lookup table)
    // Combine both metrics with weights
    // 0.8 because parallelism is more indicative of burstiness
    // 0.2 because communication pattern also contributes
    // Scaling factor needed because comm_cv ∈ [0, 1] while parallelism_ratio ∈ [2, 100+]. Without scaling, comm_cv would have negligible impact on the combined score.
    double burstiness_score = 2.0 * parallelism_ratio + 1.0 * (comm_cv * 100.0);

    int estimated_k = static_cast<int>(burstiness_score);

    std::cout << "  Parallelism ratio (max/min): " << parallelism_ratio << std::endl;
    std::cout << "  Communication CV: " << comm_cv << std::endl;
    std::cout << "  Max parallelism at depth: " << max_parallelism << std::endl;
    std::cout << "  Burstiness score: " << burstiness_score << std::endl;
    std::cout << "  Estimated k: " << estimated_k << std::endl;
    
    return estimated_k;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Per-Channel Service Time Calculation (Paper's Method - Section D)
///////////////////////////////////////////////////////////////////////////////////////////////////////////

int FastAnalyticalModel::get_max_distance_from_channel(int router, int port) const {
    // For ejection channel (local port), distance is 0
    if (port == PORT_LOCAL) {
        return 0;
    }

    // For routing channels, calculate max distance to any reachable destination
    // Using dimension-order routing, determine which nodes are reachable via this port

    int k = _arch.k;  // Network radix
    int my_x = router % k;
    int my_y = router / k;

    int max_dist = 0;

    // Check all possible destination nodes
    for (int dest = 0; dest < k * k; dest++) {
        if (dest == router) continue;  // Skip self

        int dest_x = dest % k;
        int dest_y = dest / k;

        // Trace dimension-order (XY) routing from this router
        std::vector<Hop> path = trace_routing_path(router, dest);

        // Check if this path uses the given output port as first hop
        if (!path.empty() && path[0].output_port == port) {
            // This destination is reachable via this port
            // Calculate distance (number of hops)
            int dist = path.size();
            max_dist = std::max(max_dist, dist);
        }
    }

    return max_dist;
}

void FastAnalyticalModel::calculate_channel_indices() {
    _channel_index.clear();

    int num_routers = _arch.k * _arch.k;

    // For each router and output port, calculate its index (distance group)
    for (int router = 0; router < num_routers; router++) {
        for (int port = 0; port < 5; port++) {  // 5 ports: N, E, S, W, Local
            OutputChannelKey key{router, port};

            int index = get_max_distance_from_channel(router, port);
            _channel_index[key] = index;
        }
    }

    std::cout << "Channel indexing complete" << std::endl;
}

void FastAnalyticalModel::initialize_ejection_channels() {
    // Initialize Group 0 (ejection channels) with simple service time model
    // Equation (12) from paper

    int t_body_hop = _arch.sw_alloc_delay + _arch.st_prepare_delay + _arch.st_final_delay;
    int t_link = (_arch.topology == "mesh") ? 1 : 2;
    int t_body = std::max(t_body_hop, t_link);

    // Calculate packet size statistics
    double avg_packet_flits = calculate_average_packet_size_flits();

    // Calculate variance of packet sizes
    double total_size_sq = 0.0;
    int count = 0;
    for (const auto& [task_id, task] : _tasks_by_id) {
        if (task.type == "WRITE" || task.type == "WRITE_REQ") {
            double size_bytes = task.size;
            double scaling = 8.0 / _arch.flit_size;
            int flits = static_cast<int>(std::ceil(size_bytes * scaling));
            total_size_sq += flits * flits;
            count++;
        }
    }

    double avg_size_sq = (count > 0) ? (total_size_sq / count) : (avg_packet_flits * avg_packet_flits);
    double packet_size_variance = avg_size_sq - (avg_packet_flits * avg_packet_flits);

    // For each ejection channel
    for (const auto& [key, index] : _channel_index) {
        if (index == 0) {  // Ejection channel
            // Mean service time: header flit + (avg_size - 1) body flits
            // Header includes routing and allocation delays
            double t_header = _arch.routing_delay + _arch.vc_alloc_delay + _arch.sw_alloc_delay;
            double mu = t_header + (avg_packet_flits - 1) * t_body;

            // Service time variance comes from packet size variance
            double service_variance = packet_size_variance * (t_body * t_body);

            // C_S = σ / μ
            double C_S = (mu > 0) ? sqrt(service_variance) / mu : 0.5;

            _service_time_mean[key] = mu;
            _service_time_C_S[key] = C_S;

            // Calculate initial waiting time for this ejection channel
            // W = (ρ * (C_A² + C_S²)) / (2 * (μ - λ))
            double lambda_total = 0.0;
            for (const auto& [ch_key, lambda] : _lambda_ic_oc) {
                if (ch_key.router == key.router && ch_key.output_port == key.output_port) {
                    lambda_total += lambda;
                }
            }

            double rho = lambda_total / _mu_service_rate;
            double denominator = _mu_service_rate - lambda_total;

            if (denominator > 0.0001 && rho < 0.99) {
                double numerator = rho * (_C_A * _C_A + C_S * C_S);
                _waiting_time[key] = numerator / (2.0 * denominator);
            } else {
                _waiting_time[key] = 0.0;
            }
        }
    }

    std::cout << "Ejection channels initialized (Group 0)" << std::endl;
}

std::map<OutputChannelKey, double> FastAnalyticalModel::get_routing_probabilities(OutputChannelKey channel) const {
    // Calculate probability distribution of which downstream channels packets use
    // Based on actual traffic pattern and routing algorithm

    std::map<OutputChannelKey, double> next_channel_traffic;
    double total_traffic = 0.0;

    int router = channel.router;
    int out_port = channel.output_port;

    // For each communication task that passes through this channel
    for (const auto& [task_id, task] : _tasks_by_id) {
        if (task.type != "WRITE" && task.type != "WRITE_REQ") continue;
        if (task.src == task.dst) continue;

        // Trace routing path
        std::vector<Hop> path = trace_routing_path(task.src, task.dst);

        // Find if this channel is used in the path
        for (size_t i = 0; i < path.size(); i++) {
            const Hop& hop = path[i];

            if (hop.router_id == router && hop.output_port == out_port) {
                // This task uses this channel
                // Determine next channel (if not last hop)
                if (i + 1 < path.size()) {
                    const Hop& next_hop = path[i + 1];
                    OutputChannelKey next_key{next_hop.router_id, next_hop.output_port};

                    // Weight by task traffic (for now, assume each task contributes equally)
                    double weight = 1.0;
                    if (task.type == "WRITE_REQ") {
                        weight = 4.0;  // WRITE_REQ has reverse path traffic too so more weight because of the handshake protocol
                    }

                    next_channel_traffic[next_key] += weight;
                    total_traffic += weight;
                }
                break;  // Found this channel in path, move to next task
            }
        }
    }

    // Normalize to probabilities
    std::map<OutputChannelKey, double> probabilities;
    if (total_traffic > 0.0) {
        for (const auto& [next_key, traffic] : next_channel_traffic) {
            probabilities[next_key] = traffic / total_traffic;
        }
    }

    return probabilities;
}

void FastAnalyticalModel::calculate_service_times_and_C_S() {
    // Iterative calculation of service times and C_S for all routing channels
    // Following paper's equations 16, 18, 19

    // Find max group index
    int max_group = 0;
    for (const auto& [key, index] : _channel_index) {
        max_group = std::max(max_group, index);
    }

    std::cout << "Calculating service times for groups 1 to " << max_group << std::endl;

    // Buffer parameters
    int t_body_hop = _arch.sw_alloc_delay + _arch.st_prepare_delay + _arch.st_final_delay;
    int t_link = (_arch.topology == "mesh") ? 1 : 2;
    int t_body = std::max(t_body_hop, t_link);

    // Buffer delay calculation (from paper's equation 18)
    // ASSUMPTION: Input-output buffer model where both input and output buffers exist
    // If changing to input-only buffer model, also update t_body calculation to:
    //   t_body = (n_flits - 1) * (t_s + t_w)
    // where t_s is serialization time and t_w is wire delay
    int buffer_size = _arch.vc_buf_size;  // in flits
    double buffer_delay = (buffer_size + buffer_size) * t_link;  // (IB + OB) × max(t_s, t_w)

    // Iterate until convergence
    const int max_iterations = 100;
    const double epsilon = 0.001;

    // Track which iteration each channel converged at
    std::map<OutputChannelKey, int> channel_convergence_iter;

    bool converged = false;  // Declare outside loop scope

    for (int iter = 0; iter < max_iterations; iter++) {
        converged = true;

        // Process each group in ascending order
        for (int group = 1; group <= max_group; group++) {

            // For each channel in this group
            for (const auto& [key, index] : _channel_index) {
                if (index != group) continue;

                // Get routing probabilities to downstream channels
                std::map<OutputChannelKey, double> next_probs = get_routing_probabilities(key);

                if (next_probs.empty()) {
                    // No traffic through this channel, use defaults
                    _service_time_mean[key] = _mu_service_rate > 0 ? (1.0 / _mu_service_rate) : 10.0;
                    _service_time_C_S[key] = 0.5;
                    _waiting_time[key] = 0.0;
                    continue;
                }

                double old_mu = _service_time_mean[key];
                double old_C_S = _service_time_C_S[key];

                // Equation (16): Mean service time
                double mu = 0.0;
                for (const auto& [next_key, P] : next_probs) {
                    double mu_next = _service_time_mean[next_key];  // From lower group
                    double W_next = _waiting_time[next_key];        // From lower group

                    mu += P * (mu_next + W_next - buffer_delay);
                }

                // Equation (18): Second moment
                double mu2 = 0.0;
                for (const auto& [next_key, P] : next_probs) {
                    double mu_next = _service_time_mean[next_key];
                    double C_S_next = _service_time_C_S[next_key];
                    double W_next = _waiting_time[next_key];

                    // Variance = (C_S * μ)²
                    double variance_next = (C_S_next * mu_next) * (C_S_next * mu_next);
                    // Second moment E[X²] = Var[X] + E[X]²
                    double second_moment_next = variance_next + (mu_next * mu_next);

                    double combined = mu_next + W_next - buffer_delay;
                    mu2 += P * (second_moment_next + combined * combined);
                }

                // Equation (19): C_S calculation
                double variance = mu2 - (mu * mu);
                if (variance < 0) variance = 0;  // Numerical safety

                double C_S = (mu > 0) ? sqrt(variance) / mu : 0.5;

                _service_time_mean[key] = mu;
                _service_time_C_S[key] = C_S;

                // Update waiting time using Equation (13)
                double lambda_total = 0.0;
                for (const auto& [ch_key, lambda] : _lambda_ic_oc) {
                    if (ch_key.router == key.router && ch_key.output_port == key.output_port) {
                        lambda_total += lambda;
                    }
                }

                double rho = lambda_total / _mu_service_rate;
                double denominator = _mu_service_rate - lambda_total;

                if (denominator > 0.0001 && rho < 0.99) {
                    double numerator = rho * (_C_A * _C_A + C_S * C_S);
                    _waiting_time[key] = numerator / (2.0 * denominator);
                } else {
                    _waiting_time[key] = mu * 10.0;  // High waiting time for saturated channels
                }

                // Check convergence for this channel
                bool channel_converged = (abs(old_mu - mu) <= epsilon && abs(old_C_S - C_S) <= epsilon);

                if (!channel_converged) {
                    converged = false;
                } else if (channel_convergence_iter.find(key) == channel_convergence_iter.end()) {
                    // First time this channel converged
                    channel_convergence_iter[key] = iter + 1;
                }
            }
        }

        if (converged) {
            std::cout << "Service time calculation converged in " << (iter + 1) << " iterations" << std::endl;
            break;
        }
    }

    // Check if loop exited due to max iterations (not convergence)
    if (!converged) {
        // Calculate average final C_S
        double avg_final_C_S = 0.0;
        int count = 0;
        for (const auto& [key, C_S] : _service_time_C_S) {
            avg_final_C_S += C_S;
            count++;
        }
        if (count > 0) avg_final_C_S /= count;

        std::cout << "WARN: Service time calculation did not converge after "
                  << max_iterations << " iterations. Using last computed values "
                  << "(average C_S = " << avg_final_C_S << ")" << std::endl;
    }

    // Print statistics per group
    std::map<int, double> group_C_S_sum;
    std::map<int, int> group_counts;
    std::map<int, double> group_convergence_sum;

    for (const auto& [key, C_S] : _service_time_C_S) {
        int group = _channel_index[key];
        group_C_S_sum[group] += C_S;
        group_counts[group]++;

        // Track convergence iteration
        auto conv_it = channel_convergence_iter.find(key);
        if (conv_it != channel_convergence_iter.end()) {
            group_convergence_sum[group] += conv_it->second;
        }
    }

    std::cout << "Per-group C_S statistics:" << std::endl;
    for (const auto& [group, count] : group_counts) {
        double avg_C_S = group_C_S_sum[group] / count;
        double avg_convergence = group_convergence_sum[group] / count;
        std::cout << "  Group " << group << ": " << count << " channels, "
            << "average C_S = " << std::fixed << std::setprecision(2) << avg_C_S
            << ", average convergence iteration = " << std::fixed << std::setprecision(2) << avg_convergence
            << std::endl;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// High-level simulation function
///////////////////////////////////////////////////////////////////////////////////////////////////////////

int simulate_fast_analytical(const std::string& config_file) {
    FastAnalyticalModel model;
    model.configure(config_file);
    return model.run_simulation();
}