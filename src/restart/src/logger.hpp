///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: logger.hpp
//  Description: Header file for the logger class
//  Created by:  Edoardo Cabiati
//  Date:  22/01/2025
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <vector>
#include <tuple>
#include <iterator>
#include <map>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <deque>
#include <cassert>
#include <algorithm>
#include "packet.hpp"


enum class EventType {
    START_SIMULATION,
    END_SIMULATION,
    IN_TRAFFIC,
    OUT_TRAFFIC,
    START_RECONFIGURATION,
    END_RECONFIGURATION,
    START_COMPUTATION,
    END_COMPUTATION,
};

struct HistoryBit {
    int rsource;
    int rsink;
    int start;
    int end;
};

// Power summary structure to store Booksim2 power analysis results
struct PowerSummary {
    double vdd;                      // Voltage (V)
    double resistance;               // Resistance (Ohm)
    double fclk;                     // Clock frequency (Hz)
    int completion_time_cycles;      // Completion time (cycles)
    int flit_width_bits;            // Flit width (bits)
    double channel_wire_power;       // Channel wire power (W)
    double channel_clock_power;      // Channel clock power (W)
    double channel_retiming_power;   // Channel retiming power (W)
    double channel_leakage_power;    // Channel leakage power (W)
    double input_read_power;         // Input read power (W)
    double input_write_power;        // Input write power (W)
    double input_leakage_power;      // Input leakage power (W)
    double switch_power;             // Switch power (W)
    double switch_control_power;     // Switch control power (W)
    double switch_leakage_power;     // Switch leakage power (W)
    double output_dff_power;         // Output DFF power (W)
    double output_clk_power;         // Output clock power (W)
    double output_control_power;     // Output control power (W)
    double total_power;              // Total power (W)

    PowerSummary() : vdd(0), resistance(0), fclk(0), completion_time_cycles(0),
                     flit_width_bits(0), channel_wire_power(0), channel_clock_power(0),
                     channel_retiming_power(0), channel_leakage_power(0), input_read_power(0),
                     input_write_power(0), input_leakage_power(0), switch_power(0),
                     switch_control_power(0), switch_leakage_power(0), output_dff_power(0),
                     output_clk_power(0), output_control_power(0), total_power(0) {}
};

// Statistics summary structure to store Booksim2 latency/throughput stats
struct StatsSummary {
    double packet_latency_avg;
    int packet_latency_min;
    int packet_latency_max;
    double network_latency_avg;
    int network_latency_min;
    int network_latency_max;
    double flit_latency_avg;
    int flit_latency_min;
    int flit_latency_max;
    double fragmentation_avg;
    int fragmentation_min;
    int fragmentation_max;
    double injected_packet_rate_avg;
    double accepted_packet_rate_avg;
    double injected_flit_rate_avg;
    double accepted_flit_rate_avg;
    double injected_packet_length_avg;
    double accepted_packet_length_avg;
    int total_in_flight_flits;
    int time_elapsed_cycles;

    StatsSummary() : packet_latency_avg(0), packet_latency_min(0), packet_latency_max(0),
                     network_latency_avg(0), network_latency_min(0), network_latency_max(0),
                     flit_latency_avg(0), flit_latency_min(0), flit_latency_max(0),
                     fragmentation_avg(0), fragmentation_min(0), fragmentation_max(0),
                     injected_packet_rate_avg(0), accepted_packet_rate_avg(0),
                     injected_flit_rate_avg(0), accepted_flit_rate_avg(0),
                     injected_packet_length_avg(0), accepted_packet_length_avg(0),
                     total_in_flight_flits(0), time_elapsed_cycles(0) {}
};

// the base class used to store important information on the events
class EventInfo {
    public:
        EventInfo(int id) : _id(id) {}
        virtual ~EventInfo() = default;

        int get_id() const {
            return _id;
        }
    private:
        int _id; // this id refers to the packet/workload/reconfiguration id

};

class TrafficEventInfo : public EventInfo {
    public:
        TrafficEventInfo(int id, commType type, int source, int dest, int size) : EventInfo(id), _type(type), _source(source), _dest(dest), _size(size) {}

        int get_id() const {
            return EventInfo::get_id();
        }

        int get_type() const {
            return _type;
        }

        int get_source() const {
            return _source;
        }

        int get_dest() const {
            return _dest;
        }

        int get_size() const {
            return _size;
        }


        const std::vector<HistoryBit> & get_history() const {
            return _history;
        }

        void add_history(int rsource, int rsink, int start) {
            int end = -1;
            _history.push_back({rsource, rsink, start, end});
        }

        void modify_history(int rsource, int rsink, int end) {
            // search for the key in the history starting from the end of the history
            auto it = std::find_if(_history.rbegin(), _history.rend(), [rsource, rsink](const HistoryBit & elem) {
                return elem.rsource == rsource && elem.rsink == rsink;
            });
            assert(it != _history.rend());
            it->end = end;
        }


    private:
        commType _type;
        int _source;
        int _dest;
        int _size;
        // a vector to store the history of the packet
        std::vector<HistoryBit> _history;
};

class ComputationEventInfo : public EventInfo {
    public:
        ComputationEventInfo(int id, int node, int ctime) : EventInfo(id), _node(node), _ctime(ctime) {}

        int get_id() const {
            return EventInfo::get_id();
        }

        int get_node() const {
            return _node;
        }

        int get_ctime() const {
            return _ctime;
        }

    private:
        int _node;
        int _ctime;
};


class Event {
    public:
        Event(int id, EventType type, int start_cycle, int additional_info, int ctype = -1) : _id(id), _type(type), _cycle(start_cycle), _additional_info(additional_info), _ctype(ctype), _info(nullptr) {}

        int get_id() const {
            return _id;
        }

        EventType get_type() const {
            return _type;
        }

        int get_cycle() const {
            return _cycle;
        }

        int get_additional_info() const {
            return _additional_info;
        }

        int get_ctype() const {
            return _ctype;
        }

        EventInfo & get_event_info() const {
            return *_info;
        }

         void print(std::ostream& os) const {
            
            std::string type;
            switch (_type) {
                case EventType::START_SIMULATION:
                    type = "START_SIMULATION";
                    break;
                case EventType::END_SIMULATION:
                    type = "END_SIMULATION";
                    break;
                case EventType::IN_TRAFFIC:
                    type = "IN_TRAFFIC";
                    break;
                case EventType::OUT_TRAFFIC:
                    type = "OUT_TRAFFIC";
                    break;
                case EventType::START_RECONFIGURATION:
                    type = "START_RECONFIGURATION";
                    break;
                case EventType::END_RECONFIGURATION:
                    type = "END_RECONFIGURATION";
                    break;
                case EventType::START_COMPUTATION:
                    type = "START_COMPUTATION";
                    break;
                case EventType::END_COMPUTATION:
                    type = "END_COMPUTATION";
                    break;
            }


            std::ostringstream oss;
            oss << "ID: " << _id << "\n"
                << "--------------------\n"
                << "Type: " << type << "\n"
                << "# Cycle: " << _cycle << "\n"
                << "Additional Info: " << _additional_info << "\n";
            
            if (_ctype != -1) {
                oss << "Comm Type: " << _ctype << "\n";
            }

            std::string content = oss.str();
            std::istringstream iss(content);
            std::string line;
            size_t max_length = 0;

            // Calcola la lunghezza massima del contenuto
            while (std::getline(iss, line)) {
                if (line.length() > max_length) {
                    max_length = line.length();
                }
            }

            // Stampa la scatola
            os << "+" << std::string(max_length + 2, '-') << "+\n";
            iss.clear();
            iss.seekg(0, std::ios::beg);
            while (std::getline(iss, line)) {
                os << "| " << std::left << std::setw(max_length) << line << " |\n";
            }
            os << "+" << std::string(max_length + 2, '-') << "+\n";
        }

        void set_event_info(EventInfo* info) {
            _info = info;
        }

    private:
        int _id; // the id of the event
        EventType _type; // the type of the event
        int _cycle; // the cycle in which the event starts
        int _additional_info; // the id of the relative packet/workload/reconfiguration
        int _ctype; // the type of the transmission (if the event is a traffic event)
        EventInfo* _info; // the info of the event
};



class EventLogger {
    public:
        EventLogger() : _events(), _id_counter(0), _has_power_summary(false), _has_stats_summary(false) {}

        ~EventLogger() {
            for (auto& [id, info] : _event_info) {
                for (auto& [ctype, event_info] : info) {
                    delete event_info;
                }
            }
        }

        void initialize_event_info() {
            _events.emplace_back(Event(_id_counter, EventType::START_SIMULATION, 0, -1));
            _id_counter++;
        }

        void end_simulation(int clock_cycle) {
            _events.emplace_back(Event(_id_counter, EventType::END_SIMULATION, clock_cycle , -1));
            _id_counter++;
        }

        void add_tevent_info(EventInfo* info) {
            // check if the EventInfo is TrafficEventInfo or ComputationEventInfo
            if (dynamic_cast<TrafficEventInfo*>(info)) {
                auto tinfo = dynamic_cast<TrafficEventInfo*>(info);
                _event_info[info->get_id()][tinfo->get_type()] = info;
            } else if (dynamic_cast<ComputationEventInfo*>(info)) {
                _event_info[info->get_id()][0] = info;
            }
        }

        void add_tevent_history(int id, commType ctype, int source, int sink, int start_cycle) {
            if (!_event_info[id][ctype]) {
                TrafficEventInfo* info = new TrafficEventInfo(id, ctype, source, sink, 0);
                _event_info[id][ctype] = info;
            }
            TrafficEventInfo* info = dynamic_cast<TrafficEventInfo*>(_event_info[id][ctype]);
            info->add_history(source, sink, start_cycle);
        }

        void modify_tevent_history(int id, commType ctype, int ch_source, int ch_sink, int end_cycle) {
            assert(_event_info[id][ctype]);
            TrafficEventInfo* info = dynamic_cast<TrafficEventInfo*>(_event_info[id][ctype]);
            info->modify_history(ch_source, ch_sink, end_cycle);
        }

        void register_event(EventType type, int start_cycle, int additional_info = -1, int ctype = -1) {
            _events.emplace_back(Event(_id_counter, type, start_cycle, additional_info, ctype));
            if (type == EventType::IN_TRAFFIC || type == EventType::OUT_TRAFFIC || type == EventType::START_COMPUTATION || type == EventType::END_COMPUTATION) {
                assert (_event_info[additional_info][ctype > -1 ? ctype : 0]);
                _events.back().set_event_info(_event_info[additional_info][ctype > -1 ? ctype : 0]);
            }
            _id_counter++;
        }

        void pop_event() {
            assert(!_events.empty());
            _events.pop_front();
        }

        // getters

        std::deque<Event> const& get_events() const {
            return _events;
        }

        std::map<int,EventInfo*> get_event_info(int id) {
            return _event_info[id];
        }

        void print_events(std::ostream& os) const {
            assert(!_events.empty());
            for (const auto& event : _events) {
                event.print(os);
            }
        }

        // Power and statistics summary setters
        void set_power_summary(const PowerSummary& summary) {
            _power_summary = summary;
            _has_power_summary = true;
        }

        void set_stats_summary(const StatsSummary& summary) {
            _stats_summary = summary;
            _has_stats_summary = true;
        }

        // Power and statistics summary getters
        const PowerSummary& get_power_summary() const {
            return _power_summary;
        }

        const StatsSummary& get_stats_summary() const {
            return _stats_summary;
        }

        bool has_power_summary() const {
            return _has_power_summary;
        }

        bool has_stats_summary() const {
            return _has_stats_summary;
        }

    private:
        std::deque<Event> _events; // chrono timeline
        int _id_counter;
        std::map<int,std::map<int,EventInfo*>> _event_info; // to store the info of the events
        bool _has_power_summary;
        bool _has_stats_summary;
        PowerSummary _power_summary;
        StatsSummary _stats_summary;

};



#endif // LOGGER_HPP