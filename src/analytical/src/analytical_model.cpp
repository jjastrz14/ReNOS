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
    : _current_time(0.0), _nodes(0), _logger(nullptr), _output_file(&std::cout), _last_generated_processed(0), _debug_output(false) {
    // _logger = new AnalyticalLogger();  // Disable logger for now
    _logger = nullptr;
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
        if (arch.contains("output_delay")) _arch.output_delay = arch["output_delay"];
        if (arch.contains("credit_delay")) _arch.credit_delay = arch["credit_delay"];
        if (arch.contains("speculative")) _arch.speculative = arch["speculative"];
        if (arch.contains("routing_delay")) _arch.routing_delay = arch["routing_delay"];
        if (arch.contains("vc_alloc_delay")) _arch.vc_alloc_delay = arch["vc_alloc_delay"];
        if (arch.contains("sw_alloc_delay")) _arch.sw_alloc_delay = arch["sw_alloc_delay"];
        if (arch.contains("st_prepare_delay")) _arch.st_prepare_delay = arch["st_prepare_delay"];
        if (arch.contains("st_final_delay")) _arch.st_final_delay = arch["st_final_delay"];
        if (arch.contains("flit_size")) _arch.flit_size = arch["flit_size"];
        if (arch.contains("num_vcs")) _arch.num_vcs = arch["num_vcs"];
        if (arch.contains("vc_buf_size")) _arch.vc_buf_size = arch["vc_buf_size"];
        if (arch.contains("ANY_comp_cycles")) _arch.ANY_comp_cycles = arch["ANY_comp_cycles"];
        if (arch.contains("threshold_pe_mem")) _arch.threshold_pe_mem = arch["threshold_pe_mem"];
    } else {
        throw std::runtime_error("Configuration missing 'arch' section");
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

    // Initialize event-driven reordering tracking
    _node_needs_packet_reorder.resize(_nodes, false);
    _node_needs_workload_reorder.resize(_nodes, false);


    // Parse workloads (check if "workload" contains packet-like entries or actual workloads)
    if (config.contains("workload")) {
        const json& workload_array = config["workload"];

        // Parse workload array - separate packets from PE computations based on type
        for (const auto& item_json : workload_array) {
            std::string type_str = item_json.value("type", "");

            if (type_str == "COMP_OP") {
                // This is a PE computation workload
                AnalyticalWorkload workload;
                workload.id = item_json["id"];
                workload.pe = item_json.value("node", -1); // Use node field for PE
                workload.cycles_required = static_cast<int>(std::ceil(item_json.value("ct_required", 0) * item_json.value("ANY_comp_cycles", 0.25))); // Use ct_required * by factor as cycles, always round up
                workload.size = item_json.value("size", 0); // Use size field for data size if available

                // Parse dependencies
                if (item_json.contains("dep")) {
                    if (item_json["dep"].is_array()) {
                        for (const auto& dep_val : item_json["dep"]) {
                            if (dep_val != -1) {
                                workload.dep.push_back(dep_val);
                            }
                        }
                    }
                }
                else {
                    //throw error
                    throw std::runtime_error("No dependencies found for workload: " + type_str);
                }

                _workloads.push_back(workload);

            } else if (type_str == "WRITE") {
                // This is a network packet with only WRITE_ACK generation
                AnalyticalPacket packet;
                packet.type = WRITE;
                packet.id = item_json["id"];
                packet.src = item_json["src"];
                packet.dst = item_json["dst"];
                packet.size = item_json["size"];
                packet.pt_required = item_json.value("pt_required", 0);
                packet.data_size = item_json.value("data_size", packet.size);
                packet.data_dep = item_json.value("data_dep", -1);

                // Parse dependencies
                if (item_json.contains("dep")) {
                    if (item_json["dep"].is_array()) {
                        for (const auto& dep_val : item_json["dep"]) {
                            if (dep_val != -1) {
                                packet.dep.push_back(dep_val);
                            }
                        }
                    }
                }
                else {
                    //throw error
                    throw std::runtime_error("No dependencies found for workload: " + type_str);
                }

                _packets.push_back(packet);
            } else if (type_str == "WRITE_REQ"){
                // this creates a WRITE_REQ packet that will trigger a handshake protocol
                /*  1. WRITE_REQ from src -> dst
                    2. READ_REQ from dst -> src
                    3. WRITE (actuall bulk data) from src -> dst
                    4. WRITE_ACK from dst -> src 
                */
                AnalyticalPacket packet;
                packet.type = WRITE_REQ;
                packet.id = item_json["id"];
                packet.src = item_json["src"];
                packet.dst = item_json["dst"];
                packet.size = item_json["size"];
                packet.pt_required = item_json.value("pt_required", 0);
                packet.data_size = item_json.value("data_size", packet.size);
                packet.data_dep = item_json.value("data_dep", -1);

                //for handshake protocol:
                packet.bulk_data = item_json["size"];
                packet.bulk_pt_required = item_json.value("pt_required", 0);


                if (item_json.contains("dep")) {
                    if (item_json["dep"].is_array()) {
                        for (const auto& dep_val : item_json["dep"]) {
                            if (dep_val != -1) {
                                packet.dep.push_back(dep_val);
                            }
                        }
                    }
                _packets.push_back(packet);
                }
                else {
                    //throw error
                    throw std::runtime_error("No dependencies found for workload: " + type_str);
                }

            } else {
                // unknown type packet
                throw std::runtime_error("Unknown workload type: " + type_str);
            }
        }

    }

    // Preprocess packets (convert bytes to flits)
    preprocess_packets();

    //*_output_file << "Analytical model configured with " << _packets.size()
    //            << " packets and " << _workloads.size() << " workloads" << std::endl;
    //*_output_file << "Network: " << _arch.topology << " " << _arch.k << "^" << _arch.n
    //            << " (" << _nodes << " nodes)" << std::endl;
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
    *_output_file << "Warning: Unknown topology '" << _arch.topology
                << "', using Manhattan distance as default." << std::endl;
    return std::abs(src - dst);
}

double AnalyticalModel::calculate_message_latency(int src, int dst, int size_flits) const {
    // Number of hops
    int n_routers = calculate_hop_distance(src, dst);

    // Use the already calculated number of flits
    int n_flits = size_flits;

    // Link traversal time (assumed 1 cycle)
    int t_link = _arch.topology == "mesh" ? 1 : 2; // mesh has 1 cycle link delay, torus has 2 cycles this is implemented in the BookSim2 simulator

    // account for speculative routing in IQ router (like in BookSim2, look: IQRouter::AddOutputChannel 203)
    int alloc_delay = _arch.speculative ? std::max(_arch.vc_alloc_delay, _arch.sw_alloc_delay) : (_arch.vc_alloc_delay + _arch.sw_alloc_delay);

    // Head flit processing time per hop
    int t_head_hop = _arch.routing_delay + alloc_delay +
                    _arch.st_prepare_delay + _arch.st_final_delay;

    // Body flit processing time per hop
    int t_body_hop = _arch.sw_alloc_delay + _arch.st_prepare_delay + _arch.st_final_delay;

    // Queuing delay (not modeled for now)
    int queuing_delay = 1; // Assume 1 cycle queuing delay per hop for simplicity

    // Total packet latency: head + body term
    double T_packet = n_routers * (t_head_hop + t_link + queuing_delay) +
                      (n_flits - 1) * std::max(t_body_hop, t_link);
    //debug
    //*_output_file << "Calculated latency from " << src << " to " << dst << " for size " << size << " (" << n_flits << " flits, " << n_routers << " hops): " << T_packet << std::endl;

    return T_packet;
}

int AnalyticalModel::calculate_num_flits(int size_bytes) const {
    if (size_bytes <= 0) return 1;

    // Convert bytes to flits based on flit size assuming that field size is in Bytes!
    double scaling = 8.0 / _arch.flit_size;
    int flits = static_cast<int>(std::ceil(size_bytes * scaling));

    return std::max(1, flits);  // At least 1 flit
}

bool AnalyticalModel::check_dependencies_satisfied(const AnalyticalPacket* packet, int node) const {
    return check_dependencies_helper(packet->dep, node);
}

bool AnalyticalModel::check_dependencies_satisfied(const AnalyticalWorkload* workload, int node) const {
    // For COMP_OP workloads, dependencies should only be satisfied by WRITE (type 6) packets,
    // not WRITE_REQ (type 2) packets. This ensures handshake completes before computation starts.
    if (workload->dep.empty()) {
        return true;  // No dependencies
    }

    // Check if all dependencies are satisfied by WRITE packets (bulk data transfer complete)
    for (int dep_id : workload->dep) {
        bool found = false;

        // Search for executed packets at this node with matching ID and type = WRITE (6)
        // AND ensure the completion time has already passed
        for (const auto& executed : _executed[node]) {
            if (std::get<0>(executed) == dep_id && std::get<1>(executed) == WRITE) {
                double completion_time = std::get<2>(executed);
                // Only consider dependency satisfied if the packet has actually completed processing
                if (completion_time <= _current_time) {
                    found = true;
                    break;
                }
            }
        }

        if (!found) {
            return false;  // Dependency not satisfied (no WRITE packet with this ID completed yet)
        }
    }

    return true;  // All dependencies satisfied by WRITE packets
}

bool AnalyticalModel::check_dependencies_helper(const std::vector<int>& deps, int node) const {
    if (deps.empty()) {
        return true;  // No dependencies
    }

    // Check if all dependencies are satisfied
    for (int dep_id : deps) {
        bool found = false;

        // Search in executed packets/workloads at this node
        // AND ensure the completion time has already passed
        for (const auto& executed : _executed[node]) {
            if (std::get<0>(executed) == dep_id) {
                double completion_time = std::get<2>(executed);
                // Only consider dependency satisfied if the packet/workload has actually completed
                if (completion_time <= _current_time) {
                    found = true;
                    break;
                }
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
                double completion_time = std::get<2>(executed);
                max_time = std::max(max_time, completion_time);
                break;
            }
        }
    }

    return max_time;
}

void AnalyticalModel::preprocess_packets() {
    // Only preprocess original packets from JSON - generated packets already have size in flits
    for (auto& packet : _packets) {
        packet.size_flits = calculate_num_flits(packet.size);
    }

    //*_output_file << "Preprocessed " << _packets.size() << " packets (converted sizes to flits)" << std::endl;
}

long long AnalyticalModel::run_simulation() {
    *_output_file << "Starting analytical simulation..." << std::endl;

    // Reserve space for generated packets to prevent vector reallocation and pointer invalidation
    // Estimate: each original packet might generate 2-3 additional packets (handshake + reply)
    _generated_packets.reserve(_packets.size() * 5);

    // Initialize waiting queues
    for (const auto& packet : _packets) {
        _waiting_packets[packet.src].push_back(&packet);

        // Log packet insertion
        *_output_file << "Packet " << packet.id << " with size " << packet.size_flits
                    << " inserted in queue at node " << packet.src << std::endl;
    }

    for (const auto& workload : _workloads) {
        _waiting_workloads[workload.pe].push_back(&workload);

        // Log workload insertion
        *_output_file << "Workload " << workload.id << " with size " << workload.size
                    << " inserted in queue at node " << workload.pe << std::endl;
    }

    // Main simulation loop
    while (true) {
        bool activity = false;

        // Process packets and workloads
        inject_ready_packets();
        process_workloads();
        process_generated_packets();

        // Check if any packets are still waiting or being processed
        for (int node = 0; node < _nodes; node++) {
            if (!_waiting_packets[node].empty()) {
                activity = true;
                if (_debug_output) {
                    *_output_file << "Activity detected: node " << node << " has " << _waiting_packets[node].size() << " waiting packets" << std::endl;
                    for (const auto* packet : _waiting_packets[node]) {
                        *_output_file << "  Packet " << packet->id << " type " << static_cast<int>(packet->type)
                                    << " auto_generated=" << packet->auto_generated
                                    << " injection_time=" << packet->injection_time << std::endl;
                    }
                }
                break;
            }
            if (!_waiting_workloads[node].empty() || _pending_workloads[node] != nullptr) {
                activity = true;
                if (_debug_output) {
                    *_output_file << "Activity detected: node " << node << " has workload activity" << std::endl;
                }
                break;
            }
        }

        if (!activity) {
            break;  // Simulation complete
        }

        advance_time();
    }

    // Calculate final simulation time including pending reply packet completions
    double final_time = _current_time;
    if (!_pending_reply_completions.empty()) {
        double max_reply_time = *std::max_element(_pending_reply_completions.begin(), _pending_reply_completions.end());
        final_time = std::max(final_time, max_reply_time);
    }

    *_output_file << "Analytical simulation completed at time: " << final_time << std::endl;
    print_statistics(*_output_file, final_time);

    return static_cast<int>(final_time);
}

void AnalyticalModel::inject_ready_packets() {
    bool any_local_processing = false;
    double next_time = _current_time; // Track the next time we should advance to

    // BookSim2-style optimization: only check front of queue per node per cycle
    for (int node = 0; node < _nodes; node++) {
        // Event-driven reordering: only reorder when something meaningful happened
        if (_node_needs_packet_reorder[node]) {
            reorder_packet_queue(node);
            _node_needs_packet_reorder[node] = false;  // Reset flag
        }

        // Skip empty queues
        if (_waiting_packets[node].empty()) {
            continue;
        }

        // ONLY check the front packet (BookSim2 approach)
        const AnalyticalPacket* packet = _waiting_packets[node].front();

        // Check if front packet is ready to inject
        bool ready_to_inject = false;
        double injection_time;

        if (packet->auto_generated && packet->injection_time > 0) {
            // Generated packets with specific injection time
            injection_time = packet->injection_time;
            ready_to_inject = (injection_time <= _current_time);
        } else {
            // Regular packets - check dependencies
            if (check_dependencies_satisfied(packet, node)) {
                double dep_time = get_dependency_completion_time(packet->dep, node);
                injection_time = std::max(_current_time, dep_time);
                ready_to_inject = (injection_time <= _current_time);
            }
        }

        if (ready_to_inject) {
                // Handle local packets (src == dst) separately
                if (packet->src == packet->dst) {
                        // Convert packet type for local processing (like BookSim2)
                        commType processed_type = packet->type;
                        if (packet->type == WRITE_REQ) processed_type = WRITE;
                        else if (packet->type == READ_REQ) processed_type = READ;

                        *_output_file << " --( No communication )--  Packet with ID:" << packet->id
                                    << " and type " << static_cast<int>(packet->type)
                                    << " at node " << packet->src << " has been processed at time "
                                    << static_cast<int>(_current_time) << std::endl;

                        // Mark packet as executed locally with converted type (no network latency)
                        _executed[packet->dst].insert(std::make_tuple(packet->id, processed_type, _current_time));

                        // Track that we had local processing
                        any_local_processing = true;

                        // Remove from front of queue (BookSim2 style)
                        _waiting_packets[node].pop_front();

                        // Don't generate handshake packets for local communications
                        continue;
                    }
                
                double latency = 0.0; 
                int temp_write_req = 1;  // 1 flit   
                //change locally the size of WRITE_REQ packet for handshake protocol
                if (packet->type == WRITE_REQ) {
                    latency = calculate_message_latency(packet->src, packet->dst, temp_write_req);
                }
                else {
                    // Calculate latency for network packets
                    latency = calculate_message_latency(packet->src, packet->dst, packet->size_flits);
                }

                // Handle network packets (src != dst)
                double completion_time = _current_time + latency;
                
                // debugging outputs
                *_output_file << "Packet with ID:" << packet->id << " and type " << static_cast<int>(packet->type)
                            << " at node " << packet->src << " has been injected at time "
                            << static_cast<int>(_current_time) << std::endl;

                if (packet->type == WRITE) {
                    // Log arrival at destination
                    *_output_file << " Bulk packet with id: " << packet->id << " and type: " << static_cast<int>(packet->type)
                                << " arrived at: " << packet->dst << " at time: " << static_cast<int>(completion_time)
                                << " from node: " << packet->src << ", size: " << packet->size_flits << std::endl;
                    *_output_file << "Processing time: " << packet->pt_required << std::endl;
                }
                else if (packet->type == READ_REQ) {
                    // Log arrival at destination
                    *_output_file << " READ_REQ with id: " << packet->id << " and type: " << static_cast<int>(packet->type)
                                << " arrived at: " << packet->dst << " at time: " << static_cast<int>(completion_time)
                                << " from node: " << packet->src << ", size: " << packet->size_flits << std::endl;
                    *_output_file << "Processing time: " << packet->pt_required << std::endl;
                }
                else if (packet->type == WRITE_REQ) {
                    // Log arrival at destination
                    *_output_file << " WRITE_REQ with id: " << packet->id << " and type: " << static_cast<int>(packet->type)
                                << " arrived at: " << packet->dst << " at time: " << static_cast<int>(completion_time)
                                << " from node: " << packet->src << ", size: " << temp_write_req<< std::endl;
                    *_output_file << "Processing time: " << 1 << std::endl;
                }
                else if (packet->type == WRITE_ACK) {
                    // Log arrival at destination
                    *_output_file << " WRITE_ACK packet with id: " << packet->id << " and type: " << static_cast<int>(packet->type)
                                << " arrived at: " << packet->dst << " at time: " << static_cast<int>(completion_time)
                                << " from node: " << packet->src << ", size: " << packet->size_flits << std::endl;
                    *_output_file << "Processing time: " << packet->pt_required << std::endl;
                }
                else {
                    // Unknown packet type
                    *_output_file << "Packet with ID:" << packet->id << " has unknown type "
                                << static_cast<int>(packet->type) << " at node " << packet->src << std::endl;
                }

                // Log injection
                // if (_logger) {
                //     _logger->log_event(_current_time, "INJECT", packet->id, packet->src,
                //                     "Packet injected to destination " + std::to_string(packet->dst));
                // }

                // Log completion at destination
                // if (_logger) {
                //     _logger->log_event(completion_time, "COMPLETE", packet->id, packet->dst,
                //                     "Packet completed with latency " + std::to_string(latency));
                // }

                // Calculate actual processing completion time
                int temp_pt_required = (packet->type == WRITE_REQ) ? 1 : packet->pt_required;
                double processing_completion_time = completion_time + temp_pt_required;

                // Mark packet as executed at destination
                // All packets get marked as executed for tracking, but dependency satisfaction
                // is determined by the dependency checking logic
                _executed[packet->dst].insert(std::make_tuple(packet->id, packet->type, processing_completion_time));

                // Generate handshake packets based on received packet type (only for network packets)
                if (packet->type == WRITE_REQ || packet->type == READ_REQ) {
                    generate_handshake_packets(packet, processing_completion_time);
                } else if (packet->type == READ || packet->type == WRITE) {
                    generate_reply_packet(packet, processing_completion_time);
                }
                
                // Event-driven reordering: mark injection events for WRITE and WRITE_REQ packets
                if (packet->type == WRITE || packet->type == WRITE_REQ) {
                    mark_packet_injection_event(node);
                }

                // Track future completion time (including processing time)
                next_time = std::max(next_time, processing_completion_time);

                // Remove from front of queue (BookSim2 style)
                _waiting_packets[node].pop_front();
        }
        // If front packet not ready, we skip this node (BookSim2 behavior)
        // The packet will be checked again next cycle
    }

    // Update time based on packet processing
    if (any_local_processing) {
        // Add +1 cycle overhead if any local processing occurred (BookSim2 behavior)
        _current_time += 1;
    }
}

void AnalyticalModel::process_workloads() {
    // First, complete any workloads that finished at current time
    for (int node = 0; node < _nodes; node++) {
        if (_pending_workloads[node] != nullptr && _current_time >= _npu_free_time[node]) {
            const AnalyticalWorkload* completed_workload = _pending_workloads[node];

            // Mark workload as executed
            _executed[node].insert(std::make_tuple(completed_workload->id, 0, _current_time));

            // Log workload completion in BookSim2 format
            *_output_file << "Workload with ID:" << completed_workload->id << " at node " << node
                        << " has been processed at time " << static_cast<int>(_current_time) << std::endl;

            // Log completion
            // if (_logger) {
            //     _logger->log_event(_current_time, "WORKLOAD_COMPLETE", completed_workload->id, node,
            //                     "Workload processing completed");
            // }

            // Event-driven reordering: mark workload end event
            mark_workload_event(node);

            _pending_workloads[node] = nullptr;
        }
    }

    // Then, start workloads that can start at current time - BookSim2 style optimization
    for (int node = 0; node < _nodes; node++) {
        // Event-driven reordering: only reorder when something meaningful happened
        if (_node_needs_workload_reorder[node]) {
            reorder_workload_queue(node);
            _node_needs_workload_reorder[node] = false;  // Reset flag
        }

        // Try to start new workload if NPU is free and queue not empty
        if (_pending_workloads[node] == nullptr && !_waiting_workloads[node].empty()) {
            // ONLY check the front workload (BookSim2 approach)
            const AnalyticalWorkload* workload = _waiting_workloads[node].front();

            if (check_dependencies_satisfied(workload, node)) {
                // Check if ready at current time
                double dep_completion_time = get_dependency_completion_time(workload->dep, node);
                double start_time = std::max(_current_time, dep_completion_time);

                // Only start if ready at current time
                if (start_time <= _current_time) {
                    double end_time = _current_time + workload->cycles_required;

                    _npu_free_time[node] = end_time;
                    _pending_workloads[node] = workload;

                    // Log workload start in BookSim2 format
                    *_output_file << "Workload with ID:" << workload->id << " at node " << node
                                << " has started processing at time " << static_cast<int>(_current_time) << std::endl;

                    // Log start
                    // if (_logger) {
                    //     _logger->log_event(_current_time, "WORKLOAD_START", workload->id, node,
                    //                     "Started processing workload");
                    // }

                    // Event-driven reordering: mark workload start event
                    mark_workload_event(node);

                    // Remove from front of queue (BookSim2 style)
                    _waiting_workloads[node].pop_front();
                }
            }
            // If front workload not ready, we skip this node (BookSim2 behavior)
        }
    }
}

void AnalyticalModel::process_generated_packets() {
    // Process any newly generated packets and add them to waiting queues
    // This avoids pointer invalidation issues with vector reallocation

    for (size_t i = _last_generated_processed; i < _generated_packets.size(); i++) {
        const AnalyticalPacket& packet = _generated_packets[i];

        // Add generated packets to waiting queue
        _waiting_packets[packet.src].push_back(&packet);

        // Event-driven reordering: mark that this node has new packets to consider
        _node_needs_packet_reorder[packet.src] = true;

        if (_debug_output) {
            *_output_file << "Added generated packet " << packet.id << " (type " << static_cast<int>(packet.type)
                        << ") to waiting queue at node " << packet.src
                        << " with injection_time " << packet.injection_time << std::endl;
        }
    }

    _last_generated_processed = _generated_packets.size();
}

void AnalyticalModel::advance_time() {
    // Find next event time
    double next_time = _current_time + 1.0;  // Default advance
    if (_debug_output) {
        *_output_file << "advance_time: current_time=" << _current_time << ", initial next_time=" << next_time << std::endl;
    }

    // Check for NPU completion times
    for (int node = 0; node < _nodes; node++) {
        if (_pending_workloads[node] != nullptr && _npu_free_time[node] > _current_time) {
            next_time = std::min(next_time, _npu_free_time[node]);
            if (_debug_output) {
                *_output_file << "  NPU completion at node " << node << " time " << _npu_free_time[node] << ", next_time=" << next_time << std::endl;
            }
        }

        // Check for generated packet injection times
        for (const auto* packet : _waiting_packets[node]) {
            if (packet->auto_generated && packet->injection_time > _current_time) {
                next_time = std::min(next_time, packet->injection_time);
                if (_debug_output) {
                    *_output_file << "  Generated packet " << packet->id << " at node " << node << " injection_time " << packet->injection_time << ", next_time=" << next_time << std::endl;
                }
            }
        }
    }

    if (_debug_output) {
        *_output_file << "advance_time: advancing from " << _current_time << " to " << next_time << std::endl;
    }
    _current_time = next_time;
}

void AnalyticalModel::generate_handshake_packets(const AnalyticalPacket* received_packet, double arrival_time) {
    // Based on restart/trafficmanager.cpp:905-925
    // AUTOMATIC GENERATION OF READ_REQ AND WRITEs AFTER RECEIVED MESSAGES

    int dest = received_packet->dst;
    int src = received_packet->src;

    // Create new packet based on handshake protocol
    AnalyticalPacket new_packet;

    new_packet.id = received_packet->id;  // Use same ID for simplicity
    new_packet.src = dest;  // Reverse direction: dst becomes src
    new_packet.dst = src;   // src becomes dst
    new_packet.data_dep = received_packet->data_dep;
    new_packet.auto_generated = true;

    // Set dependencies
    if (received_packet->type == READ_REQ) {
        // READ_REQ -> WRITE packet from dst to src
        new_packet.type = WRITE;
        // Size already in flits for generated packets
        new_packet.size_flits = calculate_num_flits(received_packet->bulk_data);
        new_packet.pt_required = received_packet->bulk_pt_required;
        new_packet.dep.clear();
        if (received_packet->data_dep != -1) {
            new_packet.dep.push_back(received_packet->data_dep);
        }

        //if (_logger) {
        //    _logger->log_event(_current_time, "GENERATE_WRITE", new_packet.id, dest,
        //                    "Generated WRITE packet in response to READ_REQ");
        //}
    } else {
        // WRITE_REQ -> READ_REQ packet from dst to src
        new_packet.type = READ_REQ;
        new_packet.size = 1;  // READ_REQ packets are typically small (1 byte)
        new_packet.size_flits = 1;  // 1 flit for control packets
        new_packet.pt_required = 1;  // 1 cycle of processing at destination
        new_packet.dep.clear();  // No dependencies
        new_packet.bulk_data = received_packet->bulk_data;
        new_packet.bulk_pt_required = received_packet->bulk_pt_required;

        //if (_logger) {
        //    _logger->log_event(_current_time, "GENERATE_READ_REQ", new_packet.id, dest,
        //                    "Generated READ_REQ packet in response to WRITE_REQ");
        //}
    }

    // Set injection time to be 1 cycle after generation (BookSim2 behavior)
    new_packet.injection_time = arrival_time + 1;

    _generated_packets.push_back(new_packet);
    // Don't add to waiting queue immediately - let next iteration find it

    *_output_file << "Generated " << commTypeToString(new_packet.type)
                << " (id: " << new_packet.id << ") packet at time: " << static_cast<int>(arrival_time)
                << " from node: " << dest << " to node: " << src << std::endl;
}


void AnalyticalModel::generate_reply_packet(const AnalyticalPacket* request_packet, double reply_time) {
    // Generate automatic reply packets (READ/WRITE -> READ_ACK/WRITE_ACK)

    if (request_packet->type == WRITE || request_packet->type == READ) {
        AnalyticalPacket reply_packet;

        reply_packet.id = request_packet->id;  // Use same ID for simplicity
        reply_packet.src = request_packet->dst;  // Reply from destination
        reply_packet.dst = request_packet->src;  // Reply to source
        reply_packet.size = 1;  // Reply packets are typically small (1 byte)
        reply_packet.size_flits = 1;  // 1 flit for control packets
        reply_packet.pt_required = 1;  // 1 cycle for processing in BookSim2 might be 0 Minimal processing for ACK
        reply_packet.type = get_reply_type(request_packet->type);
        reply_packet.auto_generated = true;
        reply_packet.dep.clear();  // No dependencies for replies

        // Set injection time to be 1 cycle after generation (BookSim2 behavior)
        reply_packet.injection_time = reply_time + 1;

        // Calculate when this reply packet will actually complete (injection + network latency + processing)
        double reply_network_latency = calculate_message_latency(reply_packet.src, reply_packet.dst, reply_packet.size_flits);
        double reply_completion_time = reply_packet.injection_time + reply_network_latency + reply_packet.pt_required;
        reply_packet.completion_time = reply_completion_time;

        // Track this completion time for final simulation time calculation
        _pending_reply_completions.push_back(reply_completion_time);

        _generated_packets.push_back(reply_packet);
        // Don't add to waiting queue immediately - let next iteration find it

        //if (_logger) {
        //   _logger->log_event(reply_time, "GENERATE_REPLY", reply_packet.id, reply_packet.src,
        //                    "Generated " + commTypeToString(reply_packet.type) + " reply");
        //}

        *_output_file << "Generated " << commTypeToString(reply_packet.type)
                    << " reply (id: " << reply_packet.id << ") at time: " << static_cast<int>(reply_time)
                    << " from node: " << reply_packet.src << " to node: " << reply_packet.dst << std::endl;
    }
    else {
        throw std::runtime_error("Attempted to generate reply for non-request packet type");
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

void AnalyticalModel::reorder_packet_queue(int node) {
    // BookSim2-style queue reordering: move ready packets to front
    if (_waiting_packets[node].size() < 2) {
        return;  // No need to reorder if less than 2 packets
    }

    std::vector<const AnalyticalPacket*> ready_packets;
    std::vector<const AnalyticalPacket*> waiting_packets;

    // Separate ready and waiting packets
    for (const auto* packet : _waiting_packets[node]) {
        if (check_dependencies_satisfied(packet, node)) {
            ready_packets.push_back(packet);
        } else {
            waiting_packets.push_back(packet);
        }
    }

    // Rebuild queue: ready packets first, then waiting packets
    _waiting_packets[node].clear();
    for (const auto* packet : ready_packets) {
        _waiting_packets[node].push_back(packet);
    }
    for (const auto* packet : waiting_packets) {
        _waiting_packets[node].push_back(packet);
    }
}

void AnalyticalModel::reorder_workload_queue(int node) {
    // BookSim2-style queue reordering: move ready workloads to front
    if (_waiting_workloads[node].size() < 2) {
        return;  // No need to reorder if less than 2 workloads
    }

    std::vector<const AnalyticalWorkload*> ready_workloads;
    std::vector<const AnalyticalWorkload*> waiting_workloads;

    // Separate ready and waiting workloads
    for (const auto* workload : _waiting_workloads[node]) {
        if (check_dependencies_satisfied(workload, node)) {
            ready_workloads.push_back(workload);
        } else {
            waiting_workloads.push_back(workload);
        }
    }

    // Rebuild queue: ready workloads first, then waiting workloads
    _waiting_workloads[node].clear();
    for (const auto* workload : ready_workloads) {
        _waiting_workloads[node].push_back(workload);
    }
    for (const auto* workload : waiting_workloads) {
        _waiting_workloads[node].push_back(workload);
    }
}

void AnalyticalModel::mark_workload_event(int node) {
    // When a workload starts or ends, it may satisfy dependencies for:
    // 1. Other workloads at this node
    // 2. Packets at this node waiting for workload completion
    _node_needs_workload_reorder[node] = true;
    _node_needs_packet_reorder[node] = true;
}

void AnalyticalModel::mark_packet_injection_event(int node) {
    // When WRITE/WRITE_REQ packets are injected, they may eventually
    // complete and satisfy dependencies across the network
    // For simplicity, mark all nodes since cross-node dependencies are possible
    for (int n = 0; n < _nodes; n++) {
        _node_needs_packet_reorder[n] = true;
        _node_needs_workload_reorder[n] = true;
    }
    //_node_needs_packet_reorder[node] = true;
    //_node_needs_workload_reorder[node] = true;
}

void AnalyticalModel::print_statistics(std::ostream& out, double final_time) const {
    // Use final_time if provided, otherwise fall back to _current_time
    double time_to_use = (final_time > 0) ? final_time : _current_time;

    out << "\n=== Analytical Model Statistics ===" << std::endl;
    out << "Total simulation time: " << time_to_use << " cycles" << std::endl;

    // Count original packets from JSON
    //int original_packets = _packets.size();

    // Estimate total packets: original + generated (assume each WRITE generates WRITE_ACK, each WRITE_REQ generates READ_REQ)
    // Simple approximation: double the packet count
    //int estimated_total_packets = original_packets * 2;

    //out << "Original packets from JSON: " << original_packets << std::endl;
    //out << "Estimated total packets (including generated): " << estimated_total_packets << std::endl;
    //out << "Total workloads processed: " << _workloads.size() << std::endl;

    // Calculate throughput: packets per cycle
    //if (time_to_use > 0) {
    //    double throughput = static_cast<double>(estimated_total_packets) / time_to_use;
    //    out << "Average throughput: " << throughput << " packets/cycle" << std::endl;
    //    out << "Average latency per packet: " << (time_to_use / estimated_total_packets) << " cycles" << std::endl;
    //} else {
        //out << "No simulation time elapsed, cannot calculate throughput." << std::endl;
    //}
}