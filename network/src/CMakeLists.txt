
# Source files list
set(network_SRC
    node.cpp
    utilities.cpp
    packet.cpp
)

# Include files list
set(network_INC
    node.hpp
    utilities.hpp
    packet.hpp
    params.hpp
)

PREPEND(network_SRC)
PREPEND(network_INC)

# pass the variable to parent scope
set(network_SRC ${network_SRC} PARENT_SCOPE)
set(network_INC ${network_INC} PARENT_SCOPE)