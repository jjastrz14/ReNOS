
cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)

# Source files list
set(tests_SRC
    test_base.cpp
    test_traffic.cpp
    test_injection.cpp
    test_channel.cpp
    test_outset.cpp
    test_config.cpp
)

# Header files
set(tests_INC
    ../src/base.hpp
    ../src/traffic.hpp 
    ../src/injection.hpp
    ../src/channel.hpp
    ../src/outset.hpp
    ../src/config.hpp
)

PREPEND(restartTests ${tests_SRC} ${tests_INC})
find_package(Catch2 3 REQUIRED)


add_executable(restartTests ${tests_SRC} ${tests_INC})
target_link_libraries(restartTests Catch2::Catch2WithMain restart)

# Enable testing via CTest
enable_testing()
add_test(NAME restartTests COMMAND restartTests)
