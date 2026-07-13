cmake_minimum_required(VERSION 3.18)

# Provide Google Test. Prefer a system-installed GTest if present; otherwise
# fetch it from source so the test suite is self-contained and reproducible on a
# clean CI host (no apt requirement).
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    # On Windows/MSVC we'd need gtest_force_shared_crt, but this project targets
    # Linux/macOS, so leave defaults.
    FetchContent_MakeAvailable(googletest)
endif()

# The test executable links ONLY against the library it is testing. Previously
# this target also re-compiled ../src/*.cpp directly into the executable, which
# (a) duplicated every symbol already present in libopenconstruct-jetson and
# (b) compiled jetson_status.cpp with MOCK_CUDA while the library compiled it
# differently, producing ODR/multiple-definition violations. Linking the real
# library instead means the tests exercise exactly the artifact we ship.
add_executable(test_jetson test_jetson.cpp)

target_link_libraries(test_jetson
    PRIVATE
        openconstruct-jetson
        GTest::gtest
        GTest::gtest_main
        pthread
)

target_compile_definitions(test_jetson PRIVATE MOCK_CUDA)

# Add test to CTest
add_test(NAME JetsonTests COMMAND test_jetson)

message(STATUS "Test configuration:")
message(STATUS "  Mock CUDA: enabled (for testing on non-Jetson systems)")
message(STATUS "  GTest source: ${GTest_FOUND}")
