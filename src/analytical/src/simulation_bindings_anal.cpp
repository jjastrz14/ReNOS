///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: simulation_bindings_anal.cpp
//  Description: Python bindings for the analytical NoC model using pybind11
//  Created by:  Jakub Jastrzebski
//  Date:  15/09/2025
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <sstream>
#include <memory>

#include "analytical_model.hpp"
#include "simulation_fun_anal.hpp"

namespace py = pybind11;

PYBIND11_MODULE(analytical_simulator, m) {
    m.doc() = "Analytical NoC Performance Model - Fast latency estimation";

    // commType enum
    py::enum_<commType>(m, "commType")
        .value("READ_REQ", READ_REQ, "Read request from PE to memory")
        .value("READ_ACK", READ_ACK, "Read acknowledgment from memory to PE")
        .value("READ", READ, "Read data transfer from memory to PE")
        .value("WRITE_REQ", WRITE_REQ, "Write request from PE to memory")
        .value("WRITE_ACK", WRITE_ACK, "Write acknowledgment from memory to PE")
        .value("WRITE", WRITE, "Write data transfer from PE to memory")
        .value("ANY", ANY, "Any packet type (generic handling)")
        .export_values();

    // Helper functions for commType
    m.def("intToCommType", &intToCommType, "Convert integer to commType", py::arg("i"));
    m.def("commTypeToString", &commTypeToString, "Convert commType to string", py::arg("type"));
    m.def("isRequestType", &isRequestType, "Check if commType is a request type", py::arg("type"));
    m.def("isReplyType", &isReplyType, "Check if commType is a reply type", py::arg("type"));

    // AnalyticalLogger class
    py::class_<AnalyticalLogger>(m, "AnalyticalLogger")
        .def(py::init<>())
        .def("log_event", &AnalyticalLogger::log_event,
            "Log an event with timestamp, type, packet_id, node, and description")
        .def("print_events", [](const AnalyticalLogger& logger) {
            std::ostringstream oss;
            logger.print_events(oss);
            return oss.str();
        }, "Get all logged events as a string")
        .def("clear", &AnalyticalLogger::clear, "Clear all logged events")
        .def("get_events", [](const AnalyticalLogger& logger) {
            return logger.events;
        }, "Get list of all events")
        .def_readonly("events", &AnalyticalLogger::events);

    // Event structure
    py::class_<AnalyticalLogger::Event>(m, "Event")
        .def_readonly("time", &AnalyticalLogger::Event::time)
        .def_readonly("type", &AnalyticalLogger::Event::type)
        .def_readonly("packet_id", &AnalyticalLogger::Event::packet_id)
        .def_readonly("node", &AnalyticalLogger::Event::node)
        .def_readonly("description", &AnalyticalLogger::Event::description);

    // ArchConfig structure
    py::class_<ArchConfig>(m, "ArchConfig")
        .def(py::init<>())
        .def_readwrite("topology", &ArchConfig::topology)
        .def_readwrite("k", &ArchConfig::k)
        .def_readwrite("n", &ArchConfig::n)
        .def_readwrite("routing_delay", &ArchConfig::routing_delay)
        .def_readwrite("vc_alloc_delay", &ArchConfig::vc_alloc_delay)
        .def_readwrite("sw_alloc_delay", &ArchConfig::sw_alloc_delay)
        .def_readwrite("st_prepare_delay", &ArchConfig::st_prepare_delay)
        .def_readwrite("st_final_delay", &ArchConfig::st_final_delay)
        .def_readwrite("flit_size", &ArchConfig::flit_size)
        .def_readwrite("num_vcs", &ArchConfig::num_vcs)
        .def_readwrite("vc_buf_size", &ArchConfig::vc_buf_size);

    // AnalyticalPacket structure
    py::class_<AnalyticalPacket>(m, "AnalyticalPacket")
        .def(py::init<>())
        .def_readwrite("id", &AnalyticalPacket::id)
        .def_readwrite("src", &AnalyticalPacket::src)
        .def_readwrite("dst", &AnalyticalPacket::dst)
        .def_readwrite("size", &AnalyticalPacket::size)
        .def_readwrite("dep", &AnalyticalPacket::dep)
        .def_readwrite("cl", &AnalyticalPacket::cl)
        .def_readwrite("type", &AnalyticalPacket::type)
        .def_readwrite("pt_required", &AnalyticalPacket::pt_required)
        .def_readwrite("data_size", &AnalyticalPacket::data_size)
        .def_readwrite("data_ptime_expected", &AnalyticalPacket::data_ptime_expected)
        .def_readwrite("data_dep", &AnalyticalPacket::data_dep)
        .def_readwrite("rpid", &AnalyticalPacket::rpid)
        .def_readwrite("size_flits", &AnalyticalPacket::size_flits)
        .def_readwrite("injection_time", &AnalyticalPacket::injection_time)
        .def_readwrite("completion_time", &AnalyticalPacket::completion_time)
        .def_readwrite("auto_generated", &AnalyticalPacket::auto_generated)
        .def("__repr__", [](const AnalyticalPacket &p) {
            std::ostringstream oss;
            oss << "AnalyticalPacket(id=" << p.id << ", type=" << commTypeToString(p.type)
                << ", src=" << p.src << ", dst=" << p.dst << ", size=" << p.size << ")";
            return oss.str();
        });

    // AnalyticalWorkload structure
    py::class_<AnalyticalWorkload>(m, "AnalyticalWorkload")
        .def(py::init<>())
        .def_readwrite("id", &AnalyticalWorkload::id)
        .def_readwrite("pe", &AnalyticalWorkload::pe)
        .def_readwrite("dep", &AnalyticalWorkload::dep)
        .def_readwrite("cycles_required", &AnalyticalWorkload::cycles_required);

    // AnalyticalModel class
    py::class_<AnalyticalModel>(m, "AnalyticalModel")
        .def(py::init<>())
        .def("configure", &AnalyticalModel::configure,
             "Configure the model from a JSON configuration file",
             py::arg("config_file"))
        .def("run_simulation", &AnalyticalModel::run_simulation,
             "Run the analytical simulation and return total simulation time")
        .def("calculate_hop_distance", &AnalyticalModel::calculate_hop_distance,
             "Calculate hop distance between two nodes",
             py::arg("src"), py::arg("dst"))
        .def("calculate_message_latency", &AnalyticalModel::calculate_message_latency,
             "Calculate latency for a message using analytical formula",
             py::arg("src"), py::arg("dst"), py::arg("size"), py::arg("is_reply") = false)
        .def("calculate_num_flits", &AnalyticalModel::calculate_num_flits,
             "Calculate number of flits for a given message size in bytes",
             py::arg("size_bytes"))
        .def("get_simulation_time", &AnalyticalModel::get_simulation_time,
             "Get current simulation time")
        .def("get_total_packets", &AnalyticalModel::get_total_packets,
             "Get total number of packets")
        .def("get_logger", &AnalyticalModel::get_logger,
             "Get the logger instance", py::return_value_policy::reference)
        .def("print_statistics", [](const AnalyticalModel& model) {
            std::ostringstream oss;
            model.print_statistics(oss);
            return oss.str();
        }, "Get simulation statistics as a string")
        .def("set_output_stream", [](AnalyticalModel& model, const std::string& filename) {
            static std::ofstream file_stream;
            if (!filename.empty() && filename != "-") {
                file_stream.open(filename);
                if (file_stream.is_open()) {
                    model.set_output_file(&file_stream);
                } else {
                    throw std::runtime_error("Cannot open output file: " + filename);
                }
            }
        }, "Set output file for simulation messages", py::arg("filename"));

    // High-level simulation wrapper function
    m.def("simulate_analytical", &SimulateAnalytical,
          "Run analytical simulation with config file and output file",
          py::arg("config_file"), py::arg("output_file") = "");

    // Utility functions for topology calculations
    m.def("calculate_manhattan_distance", [](int src, int dst, int k, int n) {
        int hops = 0;
        int temp_src = src;
        int temp_dst = dst;

        for (int dim = 0; dim < n; dim++) {
            int src_coord = temp_src % k;
            int dst_coord = temp_dst % k;
            hops += std::abs(src_coord - dst_coord);
            temp_src /= k;
            temp_dst /= k;
        }
        return hops;
    }, "Calculate Manhattan distance between nodes",
       py::arg("src"), py::arg("dst"), py::arg("k"), py::arg("n"));

    m.def("calculate_torus_distance", [](int src, int dst, int k, int n) {
        int hops = 0;
        int temp_src = src;
        int temp_dst = dst;

        for (int dim = 0; dim < n; dim++) {
            int src_coord = temp_src % k;
            int dst_coord = temp_dst % k;
            int distance = std::abs(src_coord - dst_coord);
            // Consider wraparound for torus
            distance = std::min(distance, k - distance);
            hops += distance;
            temp_src /= k;
            temp_dst /= k;
        }
        return hops;
    }, "Calculate torus distance between nodes with wraparound",
       py::arg("src"), py::arg("dst"), py::arg("k"), py::arg("n"));

    // Version and model info
    m.attr("__version__") = "1.0.0";
    m.attr("__model__") = "Analytical NoC Performance Model";
    m.attr("__description__") = "Fast latency estimation for Network-on-Chip architectures";
}