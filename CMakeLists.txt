cmake_minimum_required(VERSION 3.5.0)
set(CMAKE_CXX_STANDARD 17)
project(restart VERSION 0.1.0 LANGUAGES CXX)

# check the system, if its apple and arm64, then set the architecture to arm64,
# also activate debug mode
if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(CMAKE_OSX_ARCHITECTURES "arm64")
        set(CMAKE_BUILD_TYPE Debug)
    endif()
endif()

# function to prepend the current source directory to a list of files
# this is used to make the files in the subdirectories visible to the main executable
function(PREPEND invar)
    set(listVar "")
    foreach(f ${${invar}})
        list(APPEND listVar ${CMAKE_CURRENT_SOURCE_DIR}/${f})
    endforeach()
    set(${invar} ${listVar} PARENT_SCOPE)
endfunction()


# Set output directories for binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")

# Include directories for header files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/restart/include)

# Find and include the nlohmann JSON library
# find_path(nlohmann_json_INCLUDE_DIR nlohmann/json.hpp PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/restart/include)

# Find Python and pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

# Add subdirectories for each module
add_subdirectory(src/restart/src)
add_subdirectory(src/restart/src/routers)
add_subdirectory(src/restart/src/arbiters)
add_subdirectory(src/restart/src/allocators)
add_subdirectory(src/restart/src/power)
add_subdirectory(src/restart/src/networks)


include_directories(${Python_INCLUDE_DIRS} ${pybind11_INCLUDE_DIR})

add_library(restart STATIC ${common_SRC} ${routers_SRC} ${arbiters_SRC} ${allocators_SRC} ${power_SRC} ${network_SRC}  ${common_INC} ${routers_INC} ${arbiters_INC} ${allocators_INC} ${power_INC} ${network_INC})
target_include_directories(restart PUBLIC src src/routers src/arbiters src/allocators src/power src/networks ${nlohmann_json_INCLUDE_DIR})


# Define the main executable
add_executable(run_restart src/restart/src/main.cpp)
# Define the pybind11 module
pybind11_add_module(nocsim src/restart/src/simulation_bindings.cpp)
# Link libraries
target_link_libraries(run_restart PRIVATE common routers arbiters allocators power networks)
target_link_libraries(nocsim PRIVATE common routers arbiters allocators power networks)
