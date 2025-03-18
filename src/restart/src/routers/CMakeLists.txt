
set (SOURCES
    router.cpp
    iq_router.cpp
    event_router.cpp
    chaos_router.cpp
)

set (INCLUDES
    router.hpp
    iq_router.hpp
    event_router.hpp
    chaos_router.hpp
)


add_library(routers STATIC ${SOURCES} ${INCLUDES})

target_include_directories(routers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})