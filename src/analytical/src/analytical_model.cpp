///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: analytical_model.cpp
//  Description: Implementation of analytical NoC performance model
//  Created by:  Claude Code
//  Date:  15/09/2025
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "analytical_model.hpp"
#include <algorithm>
#include <cmath>
#include <cassert>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// commType Helper Functions
///////////////////////////////////////////////////////////////////////////////////////////////////////////

commType intToCommType(int i) {
    switch(i) {
        case 0: return ANY;
        case 1: return READ_REQ;
        case 2: return WRITE_REQ;
        case 3: return READ_ACK;
        case 4: return WRITE_ACK;
        case 5: return READ;
        case 6: return WRITE;
        default: throw std::invalid_argument("Invalid integer value for commType: " + std::to_string(i));
    }
}

std::string commTypeToString(commType type) {
    switch(type) {
        case ANY: return "ANY";
        case READ_REQ: return "READ_REQ";
        case WRITE_REQ: return "WRITE_REQ";
        case READ_ACK: return "READ_ACK";
        case WRITE_ACK: return "WRITE_ACK";
        case READ: return "READ";
        case WRITE: return "WRITE";
        default: return "UNKNOWN";
    }
}

bool isRequestType(commType type) {
    return (type == READ_REQ || type == WRITE_REQ);
}

bool isReplyType(commType type) {
    return (type == READ_ACK || type == WRITE_ACK);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// AnalyticalLogger Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////////

void AnalyticalLogger::log_event(double time, const std::string& type, int packet_id,
                                int node, const std::string& description) {
    Event event;
    event.time = time;
    event.type = type;
    event.packet_id = packet_id;
    event.node = node;
    event.description = description;
    events.push_back(event);
}

void AnalyticalLogger::print_events(std::ostream& out) const {
    out << "Time\tType\tPacket_ID\tNode\tDescription" << std::endl;
    for (const auto& event : events) {
        out << event.time << "\t" << event.type << "\t" << event.packet_id
            << "\t" << event.node << "\t" << event.description << std::endl;
    }
}

void AnalyticalLogger::clear() {
    events.clear();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// AnalyticalModel Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////////

AnalyticalModel::AnalyticalModel()
    : _current_time(0.0), _nodes(0), _logger(nullptr), _output_file(&std::cout) {
    _logger = new AnalyticalLogger();
}

AnalyticalModel::~AnalyticalModel() {
    if (_logger) {
        delete _logger;
    }
}

void AnalyticalModel::configure(const std::string& config_file) {
    // Parse JSON configuration file
    std::ifstream file(config_file);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open configuration file: " + config_file);
    }

    json config;
    file >> config;

    // Parse architecture parameters
    if (config.contains("arch")) {
        const json& arch = config["arch"];

        if (arch.contains("topology")) _arch.topology = arch["topology"];
        if (arch.contains("k")) _arch.k = arch["k"];
        if (arch.contains("n")) _arch.n = arch["n"];
        if (arch.contains("routing_delay")) _arch.routing_delay = arch["routing_delay"];
        if (arch.contains("vc_alloc_delay")) _arch.vc_alloc_delay = arch["vc_alloc_delay"];
        if (arch.contains("sw_alloc_delay")) _arch.sw_alloc_delay = arch["sw_alloc_delay"];
        if (arch.contains("st_prepare_delay")) _arch.st_prepare_delay = arch["st_prepare_delay"];
        if (arch.contains("st_final_delay")) _arch.st_final_delay = arch["st_final_delay"];
        if (arch.contains("flit_size")) _arch.flit_size = arch["flit_size"];
        if (arch.contains("num_vcs")) _arch.num_vcs = arch["num_vcs"];
        if (arch.contains("vc_buf_size")) _arch.vc_buf_size = arch["vc_buf_size"];
    }

    // Calculate number of nodes
    _nodes = 1;
    for (int i = 0; i < _arch.n; i++) {
        _nodes *= _arch.k;
    }

    // Initialize per-node data structures
    _executed.resize(_nodes);
    _waiting_packets.resize(_nodes);
    _waiting_workloads.resize(_nodes);
    _pending_workloads.resize(_nodes, nullptr);
    _npu_free_time.resize(_nodes, 0.0);

    // Parse packets
    if (config.contains("packets")) {
        const json& packets = config["packets"];
        _packets.reserve(packets.size());

        for (const auto& packet_json : packets) {
            AnalyticalPacket packet;
            packet.id = packet_json["id"];
            packet.src = packet_json["src"];
            packet.dst = packet_json["dst"];
            packet.size = packet_json["size"];
            packet.cl = packet_json.value("cl", 0);

            // Parse packet type and convert to commType
            int type_int = packet_json.value("type", 1);
            packet.type = intToCommType(type_int);

            packet.pt_required = packet_json.value("pt_required", 0);

            // Parse additional handshake protocol fields
            packet.data_size = packet_json.value("data_size", packet.size);
            packet.data_ptime_expected = packet_json.value("data_ptime_expected", packet.pt_required);
            packet.data_dep = packet_json.value("data_dep", -1);
            packet.rpid = packet_json.value("rpid", packet.id);

            // Parse dependencies
            if (packet_json.contains("dep")) {
                if (packet_json["dep"].is_number()) {
                    int dep_val = packet_json["dep"];
                    if (dep_val != -1) {
                        packet.dep.push_back(dep_val);
                    }
                } else if (packet_json["dep"].is_array()) {
                    for (const auto& dep_val : packet_json["dep"]) {
                        if (dep_val != -1) {
                            packet.dep.push_back(dep_val);
                        }
                    }
                }
            }

            _packets.push_back(packet);
        }
    }

    // Parse workloads (if present)
    if (config.contains("workloads")) {
        const json& workloads = config["workloads"];
        _workloads.reserve(workloads.size());

        for (const auto& workload_json : workloads) {
            AnalyticalWorkload workload;
            workload.id = workload_json["id"];
            workload.pe = workload_json["pe"];
            workload.cycles_required = workload_json.value("cycles_required", 0);

            // Parse dependencies
            if (workload_json.contains("dep")) {
                if (workload_json["dep"].is_number()) {
                    int dep_val = workload_json["dep"];
                    if (dep_val != -1) {
                        workload.dep.push_back(dep_val);
                    }
                } else if (workload_json["dep"].is_array()) {
                    for (const auto& dep_val : workload_json["dep"]) {
                        if (dep_val != -1) {
                            workload.dep.push_back(dep_val);
                        }
                    }
                }
            }

            _workloads.push_back(workload);
        }
    }

    // Preprocess packets (convert bytes to flits)
    preprocess_packets();

    *_output_file << "Analytical model configured with " << _packets.size()
                << " packets and " << _workloads.size() << " workloads" << std::endl;
    *_output_file << "Network: " << _arch.topology << " " << _arch.k << "^" << _arch.n
                << " (" << _nodes << " nodes)" << std::endl;
}

int AnalyticalModel::calculate_hop_distance(int src, int dst) const {
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

double AnalyticalModel::calculate_message_latency(int src, int dst, int size, bool is_reply) const {
    // Number of hops
    int n_routers = calculate_hop_distance(src, dst);

    // Calculate number of flits
    int n_flits = is_reply ? 1 : calculate_num_flits(size);

    // Link traversal time (assumed 1 cycle)
    int t_link = 1;

    // Head flit processing time per hop
    int t_head_hop = _arch.routing_delay + _arch.vc_alloc_delay + _arch.sw_alloc_delay +
                    _arch.st_prepare_delay + _arch.st_final_delay;

    // Body flit processing time per hop
    int t_body_hop = _arch.sw_alloc_delay + _arch.st_prepare_delay + _arch.st_final_delay;

    // Queuing delay (not modeled for now)
    int queuing_delay = 0;

    // Total packet latency: head + body term
    double T_packet = n_routers * (t_head_hop + t_link + queuing_delay) +
                      (n_flits - 1) * std::max(t_body_hop, t_link);

    return T_packet;
}

int AnalyticalModel::calculate_num_flits(int size_bytes) const {
    if (size_bytes <= 0) return 1;

    // Convert bytes to flits based on flit size
    double scaling = 8.0 / _arch.flit_size;
    int flits = static_cast<int>(std::ceil(size_bytes * scaling));

    return std::max(1, flits);  // At least 1 flit
}

bool AnalyticalModel::check_dependencies_satisfied(const AnalyticalPacket* packet, int node) const {
    if (packet->dep.empty()) {
        return true;  // No dependencies
    }

    // Check if all dependencies are satisfied
    for (int dep_id : packet->dep) {
        bool found = false;

        // Search in executed packets at this node
        for (const auto& executed : _executed[node]) {
            if (std::get<0>(executed) == dep_id) {
                found = true;
                break;
            }
        }

        if (!found) {
            return false;  // Dependency not satisfied
        }
    }

    return true;  // All dependencies satisfied
}

bool AnalyticalModel::check_dependencies_satisfied(const AnalyticalWorkload* workload, int node) const {
    if (workload->dep.empty()) {
        return true;  // No dependencies
    }

    // Check if all dependencies are satisfied
    for (int dep_id : workload->dep) {
        bool found = false;

        // Search in executed packets/workloads at this node
        for (const auto& executed : _executed[node]) {
            if (std::get<0>(executed) == dep_id) {
                found = true;
                break;
            }
        }

        if (!found) {
            return false;  // Dependency not satisfied
        }
    }

    return true;  // All dependencies satisfied
}

double AnalyticalModel::get_dependency_completion_time(const std::vector<int>& deps, int node) const {
    if (deps.empty()) {
        return 0.0;
    }

    double max_time = 0.0;

    for (int dep_id : deps) {
        for (const auto& executed : _executed[node]) {
            if (std::get<0>(executed) == dep_id) {
                max_time = std::max(max_time, std::get<2>(executed));
                break;
            }
        }
    }

    return max_time;
}

void AnalyticalModel::preprocess_packets() {
    for (auto& packet : _packets) {
        packet.size_flits = calculate_num_flits(packet.size);
    }

    *_output_file << "Preprocessed " << _packets.size() << " packets (converted sizes to flits)" << std::endl;
}

int AnalyticalModel::run_simulation() {
    *_output_file << "Starting analytical simulation..." << std::endl;

    // Initialize waiting queues
    for (const auto& packet : _packets) {
        _waiting_packets[packet.src].push_back(&packet);
    }

    for (const auto& workload : _workloads) {
        _waiting_workloads[workload.pe].push_back(&workload);
    }

    // Main simulation loop
    while (true) {
        bool activity = false;

        // Process packets and workloads
        inject_ready_packets();
        process_workloads();

        // Check if any packets are still waiting or being processed
        for (int node = 0; node < _nodes; node++) {
            if (!_waiting_packets[node].empty() || !_waiting_workloads[node].empty() ||
                _pending_workloads[node] != nullptr) {
                activity = true;
                break;
            }
        }

        if (!activity) {
            break;  // Simulation complete
        }

        advance_time();
    }

    *_output_file << "Analytical simulation completed at time: " << _current_time << std::endl;
    print_statistics(*_output_file);

    return static_cast<int>(_current_time);
}

void AnalyticalModel::inject_ready_packets() {
    for (int node = 0; node < _nodes; node++) {
        auto& waiting = _waiting_packets[node];

        for (auto it = waiting.begin(); it != waiting.end();) {
            const AnalyticalPacket* packet = *it;

            if (check_dependencies_satisfied(packet, node)) {
                // Dependencies satisfied, inject packet
                double dep_time = get_dependency_completion_time(packet->dep, node);
                double injection_time = std::max(_current_time, dep_time);
                double latency = calculate_message_latency(packet->src, packet->dst, packet->size);
                double completion_time = injection_time + latency;

                // Log injection
                if (_logger) {
                    _logger->log_event(injection_time, "INJECT", packet->id, packet->src,
                                     "Packet injected to destination " + std::to_string(packet->dst));
                }

                // Log completion at destination
                if (_logger) {
                    _logger->log_event(completion_time, "COMPLETE", packet->id, packet->dst,
                                     "Packet completed with latency " + std::to_string(latency));
                }

                // Mark packet as executed at destination
                _executed[packet->dst].insert(std::make_tuple(packet->id, packet->type, completion_time));

                // Generate handshake packets based on received packet type
                if (packet->type == WRITE_REQ || packet->type == READ_REQ) {
                    generate_handshake_packets(packet);
                } else if (packet->type == READ || packet->type == WRITE) {
                    generate_reply_packet(packet);
                }

                // Update current time if necessary
                _current_time = std::max(_current_time, completion_time);

                // Remove from waiting queue
                it = waiting.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void AnalyticalModel::process_workloads() {
    for (int node = 0; node < _nodes; node++) {
        // Check if NPU is free and there's a pending workload that completed
        if (_pending_workloads[node] != nullptr && _current_time >= _npu_free_time[node]) {
            const AnalyticalWorkload* completed_workload = _pending_workloads[node];

            // Mark workload as executed
            _executed[node].insert(std::make_tuple(completed_workload->id, 0, _current_time));

            // Log completion
            if (_logger) {
                _logger->log_event(_current_time, "WORKLOAD_COMPLETE", completed_workload->id, node,
                                "Workload processing completed");
            }

            _pending_workloads[node] = nullptr;
        }

        // Try to start new workload if NPU is free
        if (_pending_workloads[node] == nullptr && !_waiting_workloads[node].empty()) {
            auto& waiting = _waiting_workloads[node];

            for (auto it = waiting.begin(); it != waiting.end(); ++it) {
                const AnalyticalWorkload* workload = *it;

                if (check_dependencies_satisfied(workload, node)) {
                    // Dependencies satisfied, start processing
                    double start_time = std::max(_current_time,
                                            get_dependency_completion_time(workload->dep, node));
                    double end_time = start_time + workload->cycles_required;

                    _npu_free_time[node] = end_time;
                    _pending_workloads[node] = workload;

                    // Log start
                    if (_logger) {
                        _logger->log_event(start_time, "WORKLOAD_START", workload->id, node,
                                        "Started processing workload");
                    }

                    // Remove from waiting queue
                    waiting.erase(it);
                    break;
                }
            }
        }
    }
}

void AnalyticalModel::advance_time() {
    // Find next event time
    double next_time = _current_time + 1.0;  // Default advance

    // Check for NPU completion times
    for (int node = 0; node < _nodes; node++) {
        if (_pending_workloads[node] != nullptr && _npu_free_time[node] > _current_time) {
            next_time = std::min(next_time, _npu_free_time[node]);
        }
    }

    _current_time = next_time;
}

void AnalyticalModel::print_statistics(std::ostream& out) const {
    out << "\n=== Analytical Model Statistics ===" << std::endl;
    out << "Total simulation time: " << _current_time << " cycles" << std::endl;
    out << "Total packets processed: " << _packets.size() << std::endl;
    out << "Total workloads processed: " << _workloads.size() << std::endl;

    // Calculate average latencies
    double total_latency = 0.0;
    int completed_packets = 0;

    for (int node = 0; node < _nodes; node++) {
        for (const auto& executed : _executed[node]) {
            // Find corresponding packet to calculate latency
            int packet_id = std::get<0>(executed);
            double completion_time = std::get<2>(executed);

            for (const auto& packet : _packets) {
                if (packet.id == packet_id) {
                    double latency = calculate_message_latency(packet.src, packet.dst, packet.size);
                    total_latency += latency;
                    completed_packets++;
                    break;
                }
            }
        }
    }

    if (completed_packets > 0) {
        out << "Average packet latency: " << (total_latency / completed_packets) << " cycles" << std::endl;
    }
}

void AnalyticalModel::generate_handshake_packets(const AnalyticalPacket* received_packet) {
    // Based on restart/trafficmanager.cpp:905-925
    // AUTOMATIC GENERATION OF READ_REQ AND WRITEs AFTER RECEIVED MESSAGES

    if ((received_packet->type == WRITE_REQ || received_packet->type == READ_REQ)) {

        int dest = received_packet->dst;
        int src = received_packet->src;

        // Create new packet based on handshake protocol
        AnalyticalPacket* new_packet = new AnalyticalPacket();

        new_packet->id = received_packet->rpid;  // Use request packet ID
        new_packet->src = dest;  // Reverse direction: dst becomes src
        new_packet->dst = src;   // src becomes dst
        new_packet->size = received_packet->data_size;  // Size from data field
        new_packet->cl = received_packet->cl;
        new_packet->pt_required = received_packet->data_ptime_expected;
        new_packet->rpid = received_packet->rpid;
        new_packet->data_dep = received_packet->data_dep;
        new_packet->auto_generated = true;

        // Set dependencies
        if (received_packet->type == READ_REQ) {
            // READ_REQ -> WRITE packet from dst to src
            new_packet->type = WRITE;
            new_packet->dep.clear();
            if (received_packet->data_dep != -1) {
                new_packet->dep.push_back(received_packet->data_dep);
            }

            if (_logger) {
                _logger->log_event(_current_time, "GENERATE_WRITE", new_packet->id, dest,
                                "Generated WRITE packet in response to READ_REQ");
            }
        } else {
            // WRITE_REQ -> READ_REQ packet from dst to src
            new_packet->type = READ_REQ;
            new_packet->dep.clear();  // No dependencies

            if (_logger) {
                _logger->log_event(_current_time, "GENERATE_READ_REQ", new_packet->id, dest,
                                "Generated READ_REQ packet in response to WRITE_REQ");
            }
        }

        // Add to waiting queue for injection
        _waiting_packets[dest].push_back(new_packet);

        *_output_file << "Generated " << commTypeToString(new_packet->type)
                    << " (id: " << new_packet->id << ") packet at time: " << _current_time
                    << " from node: " << dest << " to node: " << src << std::endl;
    }
}

void AnalyticalModel::generate_reply_packet(const AnalyticalPacket* request_packet) {
    // Generate automatic reply packets (READ/WRITE -> READ_ACK/WRITE_ACK)

    if (request_packet->type == READ || request_packet->type == WRITE) {
        AnalyticalPacket* reply_packet = new AnalyticalPacket();

        reply_packet->id = request_packet->rpid;  // Use request packet ID
        reply_packet->src = request_packet->dst;  // Reply from destination
        reply_packet->dst = request_packet->src;  // Reply to source
        reply_packet->size = 1;  // Reply packets are typically small (1 flit)
        reply_packet->cl = request_packet->cl;
        reply_packet->pt_required = 0;  // Minimal processing for ACK
        reply_packet->type = get_reply_type(request_packet->type);
        reply_packet->rpid = request_packet->rpid;
        reply_packet->auto_generated = true;
        reply_packet->dep.clear();  // No dependencies for replies

        // Add to waiting queue
        _waiting_packets[reply_packet->src].push_back(reply_packet);

        if (_logger) {
            _logger->log_event(_current_time, "GENERATE_REPLY", reply_packet->id, reply_packet->src,
                            "Generated " + commTypeToString(reply_packet->type) + " reply");
        }

        *_output_file << "Generated " << commTypeToString(reply_packet->type)
                    << " reply (id: " << reply_packet->id << ") at time: " << _current_time
                    << " from node: " << reply_packet->src << " to node: " << reply_packet->dst << std::endl;
    }
}

commType AnalyticalModel::get_reply_type(commType request_type) {
    switch(request_type) {
        case READ:
        case READ_REQ:
            return READ_ACK;
        case WRITE:
        case WRITE_REQ:
            return WRITE_ACK;
        default:
            return ANY;  // No reply needed
    }
}