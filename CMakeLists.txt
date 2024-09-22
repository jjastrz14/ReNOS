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

add_subdirectory(network/src)

if(BUILD_TESTING)
    add_subdirectory(network/tests)
endif()

add_library(restart STATIC ${network_SRC} ${network_INC})

target_include_directories(restart PUBLIC network/src)