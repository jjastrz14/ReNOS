# CMakeLists.txt for Analytical NoC Performance Model
cmake_minimum_required(VERSION 3.12)
project(analytical_noc_model)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(pybind11 REQUIRED)

# Include directories
include_directories(src)
include_directories(include)

# Source files
set(SOURCES
    src/analytical_model.cpp
    src/simulation_fun_anal.cpp
)

# Headers
set(HEADERS
    src/analytical_model.hpp
    src/simulation_fun_anal.hpp
)

# Create the Python module
pybind11_add_module(analytical_simulator
    src/simulation_bindings_anal.cpp
    ${SOURCES}
)

# Compiler-specific options
target_compile_definitions(analytical_simulator PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Set properties for the module
set_target_properties(analytical_simulator PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN YES
)

# Optional: Create a standalone executable for testing
add_executable(analytical_test
    src/analytical_model.cpp
    src/simulation_fun_anal.cpp
)

# Link libraries if needed
# target_link_libraries(analytical_test ${SOME_LIBS})

message(STATUS "Analytical NoC Model configured successfully")
message(STATUS "  - Python module: analytical_simulator")
message(STATUS "  - Test executable: analytical_test")