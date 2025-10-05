///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: simulation_bindings.cpp
//  Description: Define the pybind11 wrappers for the simulation functions
//  Created by:  Edoardo Cabiati
//  Date:  24/01/2025
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "simulation_fun.hpp"

namespace py = pybind11;

PYBIND11_MODULE(nocsim, m) {
    m.doc() = "Reconfigurable Network-on-Chip simulator";
    m.def("simulate", &SimulateWrapper, "Run the simulation with the given configuration file");

    py::class_<HistoryBit>(m, "HistoryBit")
        .def(py::init<int, int, int, int>())
        .def("__repr__", [](const HistoryBit &bit) {
            std::ostringstream oss;
            oss << "HistoryBit(rsource=" << bit.rsource << ", rsink=" << bit.rsink << ", start=" << bit.start << ", end=" << bit.end << ")";
            return oss.str();
        })
        .def_readwrite("rsource", &HistoryBit::rsource)
        .def_readwrite("rsink", &HistoryBit::rsink)
        .def_readwrite("start", &HistoryBit::start)
        .def_readwrite("end", &HistoryBit::end);

    py::class_<EventInfo>(m, "EventInfo")
        .def(py::init<int>())
        .def_property_readonly("id", &EventInfo::get_id);

    py::class_<TrafficEventInfo, EventInfo>(m, "TrafficEventInfo")
        .def(py::init<int, commType, int, int, int>())
        .def("__repr__", [](const TrafficEventInfo &info) {
            std::ostringstream oss;
            oss << "TrafficEventInfo(id=" << info.get_id() << ", type=" << info.get_type() << ", source=" << info.get_source() << ", dest=" << info.get_dest() << ", size=" << info.get_size() << ")";
            return oss.str();
        })
        .def_property_readonly("id", &TrafficEventInfo::get_id)
        .def_property_readonly("type", &TrafficEventInfo::get_type)
        .def_property_readonly("source", &TrafficEventInfo::get_source)
        .def_property_readonly("dest", &TrafficEventInfo::get_dest)
        .def_property_readonly("size", &TrafficEventInfo::get_size)
        .def_property_readonly("history", &TrafficEventInfo::get_history);

    py::class_<ComputationEventInfo, EventInfo>(m, "ComputationEventInfo")
        .def(py::init<int, int, int>())
        .def("__repr__", [](const ComputationEventInfo &info) {
            std::ostringstream oss;
            oss << "ComputationEventInfo(id=" << info.get_id() << ", node=" << info.get_node() << ", ctime=" << info.get_ctime() << ")";
            return oss.str();
        })
        .def_property_readonly("id", &ComputationEventInfo::get_id)
        .def_property_readonly("node", &ComputationEventInfo::get_node)
        .def_property_readonly("ctime", &ComputationEventInfo::get_ctime);

    py::class_<Event>(m, "Event")
        .def(py::init<int, EventType, int, int>())
        .def("__repr__", [](const Event &event) {
            std::ostringstream oss;
            oss << "Event(id=" << event.get_id() << ", type=" << (int)event.get_type() << ", cycle=" << event.get_cycle() << ", additional_info=" << event.get_additional_info() << ", ctype=" << event.get_ctype() << ")";
            return oss.str();
        })
        .def_property_readonly("id", &Event::get_id)
        .def_property_readonly("type", &Event::get_type)
        .def_property_readonly("cycle", &Event::get_cycle)
        .def_property_readonly("additional_info", &Event::get_additional_info)
        .def_property_readonly("ctype", &Event::get_ctype)
        .def_property_readonly("info", &Event::get_event_info)
        .def("print", [](const Event &event) {
            std::ostringstream oss;
            event.print(oss);
            return oss.str();
        });

    py::enum_<EventType>(m, "EventType")
        .value("START_SIMULATION", EventType::START_SIMULATION)
        .value("END_SIMULATION", EventType::END_SIMULATION)
        .value("IN_TRAFFIC", EventType::IN_TRAFFIC)
        .value("OUT_TRAFFIC", EventType::OUT_TRAFFIC)
        .value("START_RECONFIGURATION", EventType::START_RECONFIGURATION)
        .value("END_RECONFIGURATION", EventType::END_RECONFIGURATION)
        .value("START_COMPUTATION", EventType::START_COMPUTATION)
        .value("END_COMPUTATION", EventType::END_COMPUTATION)
        .export_values();

    py::class_<PowerSummary>(m, "PowerSummary")
        .def(py::init<>())
        .def("__repr__", [](const PowerSummary &ps) {
            std::ostringstream oss;
            oss << "PowerSummary(total_power=" << ps.total_power << "W, fclk=" << ps.fclk << "Hz, cycles=" << ps.completion_time_cycles << ")";
            return oss.str();
        })
        .def_readonly("vdd", &PowerSummary::vdd)
        .def_readonly("resistance", &PowerSummary::resistance)
        .def_readonly("fclk", &PowerSummary::fclk)
        .def_readonly("completion_time_cycles", &PowerSummary::completion_time_cycles)
        .def_readonly("flit_width_bits", &PowerSummary::flit_width_bits)
        .def_readonly("channel_wire_power", &PowerSummary::channel_wire_power)
        .def_readonly("channel_clock_power", &PowerSummary::channel_clock_power)
        .def_readonly("channel_retiming_power", &PowerSummary::channel_retiming_power)
        .def_readonly("channel_leakage_power", &PowerSummary::channel_leakage_power)
        .def_readonly("input_read_power", &PowerSummary::input_read_power)
        .def_readonly("input_write_power", &PowerSummary::input_write_power)
        .def_readonly("input_leakage_power", &PowerSummary::input_leakage_power)
        .def_readonly("switch_power", &PowerSummary::switch_power)
        .def_readonly("switch_control_power", &PowerSummary::switch_control_power)
        .def_readonly("switch_leakage_power", &PowerSummary::switch_leakage_power)
        .def_readonly("output_dff_power", &PowerSummary::output_dff_power)
        .def_readonly("output_clk_power", &PowerSummary::output_clk_power)
        .def_readonly("output_control_power", &PowerSummary::output_control_power)
        .def_readonly("total_power", &PowerSummary::total_power);

    py::class_<StatsSummary>(m, "StatsSummary")
        .def(py::init<>())
        .def("__repr__", [](const StatsSummary &ss) {
            std::ostringstream oss;
            oss << "StatsSummary(packet_latency_avg=" << ss.packet_latency_avg << ", cycles=" << ss.time_elapsed_cycles << ")";
            return oss.str();
        })
        .def_readonly("packet_latency_avg", &StatsSummary::packet_latency_avg)
        .def_readonly("packet_latency_min", &StatsSummary::packet_latency_min)
        .def_readonly("packet_latency_max", &StatsSummary::packet_latency_max)
        .def_readonly("network_latency_avg", &StatsSummary::network_latency_avg)
        .def_readonly("network_latency_min", &StatsSummary::network_latency_min)
        .def_readonly("network_latency_max", &StatsSummary::network_latency_max)
        .def_readonly("flit_latency_avg", &StatsSummary::flit_latency_avg)
        .def_readonly("flit_latency_min", &StatsSummary::flit_latency_min)
        .def_readonly("flit_latency_max", &StatsSummary::flit_latency_max)
        .def_readonly("fragmentation_avg", &StatsSummary::fragmentation_avg)
        .def_readonly("fragmentation_min", &StatsSummary::fragmentation_min)
        .def_readonly("fragmentation_max", &StatsSummary::fragmentation_max)
        .def_readonly("injected_packet_rate_avg", &StatsSummary::injected_packet_rate_avg)
        .def_readonly("accepted_packet_rate_avg", &StatsSummary::accepted_packet_rate_avg)
        .def_readonly("injected_flit_rate_avg", &StatsSummary::injected_flit_rate_avg)
        .def_readonly("accepted_flit_rate_avg", &StatsSummary::accepted_flit_rate_avg)
        .def_readonly("injected_packet_length_avg", &StatsSummary::injected_packet_length_avg)
        .def_readonly("accepted_packet_length_avg", &StatsSummary::accepted_packet_length_avg)
        .def_readonly("total_in_flight_flits", &StatsSummary::total_in_flight_flits)
        .def_readonly("time_elapsed_cycles", &StatsSummary::time_elapsed_cycles);

    py::class_<EventLogger>(m, "EventLogger")
        .def(py::init<>())
        .def("__repr__", [](const EventLogger &logger) {
            std::ostringstream oss;
            oss << "EventLogger(events=" << logger.get_events().size() << ")";
            return oss.str();
        })
        .def("print_events", [](const EventLogger &logger) {
            std::ostringstream oss;
            logger.print_events(oss);
            return oss.str();
        })
        .def_property_readonly("events", &EventLogger::get_events)
        .def("get_event_info", &EventLogger::get_event_info, py::arg("id"))
        .def_property_readonly("power_summary", &EventLogger::get_power_summary)
        .def_property_readonly("stats_summary", &EventLogger::get_stats_summary)
        .def_property_readonly("has_power_summary", &EventLogger::has_power_summary)
        .def_property_readonly("has_stats_summary", &EventLogger::has_stats_summary);

    // Version and model info
    m.attr("__version__") = "2.1.0";
    m.attr("__model__") = "BookSim2";
    m.attr("__description__") = "BookSim2-based cycle-accurate NoC simulator with reconfiguration support, supoorting trace-like traffic patterns and JSON inputs";
}


