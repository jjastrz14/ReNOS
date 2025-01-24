

# set(PARSER_DIR "${CMAKE_CURRENT_BINARY_DIR}")
# # Find flex and bison
# find_package(BISON REQUIRED)
# find_package(FLEX REQUIRED)
# set(PARSER_OUT "${CMAKE_CURRENT_SOURCE_DIR}/parser.cpp")
# set(SCANNER_OUT "${CMAKE_CURRENT_SOURCE_DIR}/scanner.cpp")


# bison_target(MyParser config.y ${PARSER_OUT} DEFINES_FILE ${CMAKE_CURRENT_SOURCE_DIR}/parser.hpp)
# flex_target(MyScanner config.l  ${SCANNER_OUT} DEFINES_FILE ${CMAKE_CURRENT_SOURCE_DIR}/scanner.hpp)
# add_flex_bison_dependency(MyScanner MyParser)




# Source files list
set(network_SRC
    base.cpp
    booksim_config.cpp
    buffer_state.cpp
    buffer.cpp
    channel.cpp
    config.cpp
    misc_utils.cpp
    outset.cpp
    packet.cpp
    packet_reply_info.cpp
    random_utils.cpp
    rng_double_wrapper.cpp
    rng_wrapper.cpp
    routefunc.cpp
    stats.cpp
    vc.cpp
    trafficmanager.cpp
    traffic.cpp
    injection.cpp
    batchtrafficmanager.cpp
    booksim_config.cpp
    simulation_bindings.cpp
)

# Include files list
set(network_INC
    base.hpp
    booksim_config.hpp
    buffer_state.hpp
    buffer.hpp
    channel.hpp
    config.hpp
    globals.hpp
    misc_utils.hpp
    outset.hpp
    packet.hpp
    packet_reply_info.hpp
    params.hpp
    pipefifo.hpp
    random_utils.hpp
    routefunc.hpp
    stats.hpp
    vc.hpp
    trafficmanager.hpp
    traffic.hpp
    injection.hpp
    batchtrafficmanager.hpp
    booksim_config.hpp
    logger.hpp
    simulation_fun.hpp
)

PREPEND(network_SRC)
PREPEND(network_INC)

# pass the variable to parent scope
set(network_SRC ${network_SRC} PARENT_SCOPE)
set(network_INC ${network_INC} PARENT_SCOPE)
