# Fast Analytical Model CMakeLists.txt
# Note: This is included from the main CMakeLists.txt

# Include directories for nlohmann json
include_directories(${CMAKE_SOURCE_DIR}/src/restart/include)

# Source files
set(FAST_MODEL_SOURCES
    ${CMAKE_SOURCE_DIR}/src/fast_analytical_model/fast_model.cpp
    ${CMAKE_SOURCE_DIR}/src/fast_analytical_model/fast_model.h
)

# Python bindings - only if pybind11 is available
if(pybind11_FOUND)
    pybind11_add_module(fast_nocsim
        ${CMAKE_SOURCE_DIR}/src/fast_analytical_model/fast_bindings.cpp
        ${FAST_MODEL_SOURCES}
    )

    # Set optimization flags for release builds
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(fast_nocsim PRIVATE -O3 -DNDEBUG)
    endif()

    # Installation
    install(TARGETS fast_nocsim DESTINATION ${CMAKE_SOURCE_DIR}/src/build)
endif()

# Create run_fast_analytical executable in bin/
add_executable(run_fast_analytical
    ${CMAKE_SOURCE_DIR}/src/fast_analytical_model/main.cpp
    ${FAST_MODEL_SOURCES}
)

# Set optimization flags for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(run_fast_analytical PRIVATE -O3 -DNDEBUG)
endif()

# Install binary to bin/
install(TARGETS run_fast_analytical DESTINATION ${CMAKE_SOURCE_DIR}/bin)