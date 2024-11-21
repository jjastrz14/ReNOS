cmake_minimum_required(VERSION 3.5.0)
set(CMAKE_CXX_STANDARD 17)
project(restart VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

function(PREPEND invar)
    set(listVar "")
    foreach(f ${${invar}})
        list(APPEND listVar ${CMAKE_CURRENT_SOURCE_DIR}/${f})
    endforeach()
    set(${invar} ${listVar} PARENT_SCOPE)
endfunction()


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")

list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")

# reach the oter CMakeLists.txt files
add_subdirectory(src)
add_subdirectory(src/routers)
add_subdirectory(src/arbiters)
add_subdirectory(src/allocators)
add_subdirectory(src/power)
add_subdirectory(src/networks)



if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
# find the header file for json parsing
find_path(nlohmann_json_INCLUDE_DIR nlohmann/json.hpp)

add_library(restart STATIC ${network_SRC} ${routers_SRC} ${arbiters_SRC} ${allocators_SRC} ${power_SRC} ${topologies_SRC}   ${network_INC} ${routers_INC} ${arbiters_INC} ${allocators_INC} ${power_INC} ${topologies_INC})
target_include_directories(restart PUBLIC src src/routers src/arbiters src/allocators src/power src/networks ${nlohmann_json_INCLUDE_DIR}) 

add_executable(run_restart src/main.cpp)
target_link_libraries(run_restart PUBLIC restart)


