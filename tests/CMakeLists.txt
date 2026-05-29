cmake_minimum_required(VERSION 3.18)

# Find Google Test
find_package(GTest REQUIRED)

# Create test executable
add_executable(test_jetson
    test_jetson.cpp
    ../src/local_inference.cpp
    ../src/plato_jetson.cpp
    ../src/jetson_status.cpp
)

# Link against the library we're testing
target_link_libraries(test_jetson
    GTest::gtest
    GTest::gtest_main
    openconstruct-jetson
    pthread
)

# Enable mock mode for testing
target_compile_definitions(test_jetson PRIVATE MOCK_CUDA)

# Add test to CTest
add_test(NAME JetsonTests COMMAND test_jetson)

# Print test configuration
message(STATUS "Test configuration:")
message(STATUS "  Mock CUDA: enabled (for testing on non-Jetson systems)")