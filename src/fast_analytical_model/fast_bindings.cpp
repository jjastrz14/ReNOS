///////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  File name: fast_bindings.cpp
//  Description: Python bindings for the fast analytical NoC model using pybind11
//  Created by:  Jakub Jastrzebski
//  Date:  23/09/2025
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <sstream>
#include <fstream>
#include <memory>

#include "fast_model.h"

namespace py = pybind11;

PYBIND11_MODULE(fast_nocsim, m) {
    m.doc() = "Fast Analytical NoC Performance Model - Pure mathematical computation";

    // FastArchConfig structure
    py::class_<FastArchConfig>(m, "FastArchConfig")
        .def(py::init<>())
        .def_readwrite("topology", &FastArchConfig::topology)
        .def_readwrite("k", &FastArchConfig::k)
        .def_readwrite("n", &FastArchConfig::n)
        .def_readwrite("flit_size", &FastArchConfig::flit_size)
        .def_readwrite("routing_delay", &FastArchConfig::routing_delay)
        .def_readwrite("vc_alloc_delay", &FastArchConfig::vc_alloc_delay)
        .def_readwrite("sw_alloc_delay", &FastArchConfig::sw_alloc_delay)
        .def_readwrite("st_prepare_delay", &FastArchConfig::st_prepare_delay)
        .def_readwrite("st_final_delay", &FastArchConfig::st_final_delay)
        .def_readwrite("speculative", &FastArchConfig::speculative)
        .def_readwrite("ANY_comp_cycles", &FastArchConfig::ANY_comp_cycles);

    // FastTask structure
    py::class_<FastTask>(m, "FastTask")
        .def(py::init<>())
        .def_readwrite("id", &FastTask::id)
        .def_readwrite("type", &FastTask::type)
        .def_readwrite("dependencies", &FastTask::dependencies)
        .def_readwrite("src", &FastTask::src)
        .def_readwrite("dst", &FastTask::dst)
        .def_readwrite("size", &FastTask::size)
        .def_readwrite("pt_required", &FastTask::pt_required)
        .def_readwrite("node", &FastTask::node)
        .def_readwrite("ct_required", &FastTask::ct_required)
        .def("__repr__", [](const FastTask &t) {
            std::ostringstream oss;
            oss << "FastTask(id=" << t.id << ", type=" << t.type
                << ", src=" << t.src << ", dst=" << t.dst << ", size=" << t.size << ")";
            return oss.str();
        });

    // FastAnalyticalModel class
    py::class_<FastAnalyticalModel>(m, "FastAnalyticalModel")
        .def(py::init<>())
        .def("configure", &FastAnalyticalModel::configure,
            "Configure model from JSON file", py::arg("config_file"))
        .def("run_simulation", &FastAnalyticalModel::run_simulation,
            "Run fast analytical simulation and return total time")
        .def("calculate_hop_distance", &FastAnalyticalModel::calculate_hop_distance,
            "Calculate hop distance between two nodes", py::arg("src"), py::arg("dst"))
        .def("calculate_message_latency", &FastAnalyticalModel::calculate_message_latency,
            "Calculate message latency", py::arg("src"), py::arg("dst"), py::arg("size_flits"))
        .def("calculate_num_flits", &FastAnalyticalModel::calculate_num_flits,
            "Calculate number of flits from size in bytes", py::arg("size_bytes"))
        .def("get_total_simulation_time", &FastAnalyticalModel::get_total_simulation_time,
            "Get total simulation time")
        .def("get_arch_config", &FastAnalyticalModel::get_arch_config,
            "Get architecture configuration", py::return_value_policy::reference_internal)
        .def("topological_sort", &FastAnalyticalModel::topological_sort,
            "Get topological order of tasks");

    // High-level simulation wrapper function with output control
    m.def("simulate_fast_analytical", [](const std::string& config_file, const std::string& output_file = "") {
        // Handle output file like Booksim2
        std::ostream* dump_file;
        std::ofstream file_stream;
        std::ostream null_stream(nullptr);

        if (output_file == "") {
            dump_file = &null_stream;  // Empty string = no output
        } else if (output_file == "-") {
            dump_file = &std::cout;    // "-" = stdout
        } else {
            file_stream.open(output_file);
            dump_file = &file_stream;  // filename = file output
        }

        return simulate_fast_analytical(config_file, dump_file);
    }, "Run fast analytical simulation with config file",
        py::arg("config_file"), py::arg("output_file") = "");

    // Wrapper function that matches the interface of the original analytical simulator
    m.def("simulate_analytical", [](const std::string& config_file, const std::string& output_file = "") {
        // Handle output file like Booksim2
        std::ostream* dump_file;
        std::ofstream file_stream;
        std::ostream null_stream(nullptr);

        if (output_file == "") {
            dump_file = &null_stream;  // Empty string = no output
        } else if (output_file == "-") {
            dump_file = &std::cout;    // "-" = stdout
        } else {
            file_stream.open(output_file);
            dump_file = &file_stream;  // filename = file output
        }

        int simulation_time = simulate_fast_analytical(config_file, dump_file);

        // Return tuple (simulation_time, logger) to match original interface
        // Logger is None since we don't generate events in fast mode
        return std::make_tuple(simulation_time, py::none());
    }, "Run fast analytical simulation with original interface",
        py::arg("config_file"), py::arg("output_file") = "");

    // Version and model info
    m.attr("__version__") = "1.0.0";
    m.attr("__model__") = "Fast Analytical NoC Performance Model";
    m.attr("__description__") = "Mathematical latency estimation for Network-on-Chip architectures";
}