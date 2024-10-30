set(topologies_SRC
    anynet.cpp
    cmesh.cpp
    dragonfly.cpp
    fattree.cpp
    flatfly_onchip.cpp
    fly.cpp
    kncube.cpp
    network.cpp
    qtree.cpp
    tree4.cpp
)

set(topologies_INC
    anynet.hpp
    cmesh.hpp
    dragonfly.hpp
    fattree.hpp
    flatfly_onchip.hpp
    fly.hpp
    kncube.hpp
    network.hpp
    qtree.hpp
    tree4.hpp
)

PREPEND(topologies_SRC)
PREPEND(topologies_INC)

set(topologies_SRC ${topologies_SRC} PARENT_SCOPE)
set(topologies_INC ${topologies_INC} PARENT_SCOPE)