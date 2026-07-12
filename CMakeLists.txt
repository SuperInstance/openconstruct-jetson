cmake_minimum_required(VERSION 3.18)
project(openconstruct-jetson VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_TESTS "Build unit tests" ON)
option(USE_CUDA "Enable real CUDA GPU code (requires the CUDA toolkit + nvcc). Disable for CPU/mock testing on x86_64." ON)
option(USE_TENSORRT "Enable TensorRT integration (only meaningful with USE_CUDA=ON)" OFF)
option(ENABLE_MOCK_CUDA "Force the CPU mock implementation of the CUDA entry points (implied when USE_CUDA=OFF)" OFF)

# CUDA is only enabled as a language when actually requested. This lets the
# project configure and build on plain x86_64 hosts (CI) in mock/CPU mode, which
# is exactly what the README documents with `-DUSE_CUDA=OFF`.
if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Jetson architectures (Maxwell 5.2 through Orin 8.7)
    set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;72;75;80;86;87")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

    find_package(CUDA REQUIRED)
endif()

# When CUDA is off (or mock mode is explicitly requested) we compile a plain C++
# mock of the CUDA entry points instead of the .cu kernels, so the library still
# links and the unit tests exercise the real C++ control flow.
if(NOT USE_CUDA OR ENABLE_MOCK_CUDA)
    set(OCJ_USING_MOCK ON)
endif()

if(USE_TENSORRT)
    if(NOT USE_CUDA)
        message(WARNING "USE_TENSORRT=ON requires USE_CUDA=ON; disabling TensorRT.")
        set(USE_TENSORRT OFF)
    else()
        find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            HINTS ${TENSORRT_ROOT}/include
            /usr/include/aarch64-linux-gnu
            /usr/local/include
        )
        find_library(TENSORRT_LIBRARY nvinfer
            HINTS ${TENSORRT_ROOT}/lib
            /usr/lib/aarch64-linux-gnu
            /usr/local/lib
        )
        if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
            set(USE_TENSORRT ON)
        else()
            message(WARNING "TensorRT not found, disabling TensorRT support")
            set(USE_TENSORRT OFF)
        endif()
    endif()
endif()

# C++ source files (CUDA-independent)
set(SOURCES
    src/local_inference.cpp
    src/plato_jetson.cpp
    src/jetson_status.cpp
)

# GPU source: real CUDA kernels, or the CPU mock implementation.
if(USE_CUDA AND NOT OCJ_USING_MOCK)
    list(APPEND SOURCES src/cuda_sense.cu)
else()
    list(APPEND SOURCES src/cuda_sense_mock.cpp)
endif()

# Create library
add_library(openconstruct-jetson SHARED ${SOURCES})

target_include_directories(openconstruct-jetson PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(USE_CUDA)
    target_include_directories(openconstruct-jetson PUBLIC ${CUDA_INCLUDE_DIRS})
endif()

if(USE_TENSORRT)
    target_compile_definitions(openconstruct-jetson PUBLIC USE_TENSORRT)
    target_include_directories(openconstruct-jetson PUBLIC ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(openconstruct-jetson PUBLIC ${TENSORRT_LIBRARY})
endif()

if(USE_CUDA)
    target_link_libraries(openconstruct-jetson PUBLIC ${CUDA_LIBRARIES} cudart)
endif()

target_link_libraries(openconstruct-jetson PUBLIC pthread)

# In mock/CPU mode the library expects the MOCK_CUDA preprocessor symbol so the
# C++ control flow takes the testing branches.
if(OCJ_USING_MOCK)
    target_compile_definitions(openconstruct-jetson PRIVATE MOCK_CUDA)
endif()

# Reasonable warning baseline for the C++ code.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(openconstruct-jetson PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install rules
install(TARGETS openconstruct-jetson
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "OpenConstruct Jetson Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Use CUDA (real GPU): ${USE_CUDA}")
message(STATUS "  Mock/CPU mode: ${OCJ_USING_MOCK}")
message(STATUS "  Use TensorRT: ${USE_TENSORRT}")
if(USE_CUDA)
    message(STATUS "  CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "")
