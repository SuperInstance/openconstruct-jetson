cmake_minimum_required(VERSION 3.18)
project(openconstruct-jetson VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;72;75;80;86;87") # Jetson architectures
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

# Options
option(BUILD_TESTS "Build unit tests" ON)
option(ENABLE_MOCK_CUDA "Mock CUDA calls for non-Jetson testing" OFF)
option(USE_TENSORRT "Enable TensorRT integration" ON)

# Find required packages
find_package(CUDA REQUIRED)

if(USE_TENSORRT)
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
        add_definitions(-DUSE_TENSORRT)
    else()
        message(WARNING "TensorRT not found, disabling TensorRT support")
        set(USE_TENSORRT OFF)
    endif()
endif()

# Source files
set(SOURCES
    src/local_inference.cpp
    src/plato_jetson.cpp
    src/jetson_status.cpp
)

set(CUDA_SOURCES
    src/cuda_sense.cu
)

# Create library
add_library(openconstruct-jetson SHARED
    ${SOURCES}
    ${CUDA_SOURCES}
)

target_include_directories(openconstruct-jetson PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CUDA_INCLUDE_DIRS}
)

if(USE_TENSORRT)
    target_include_directories(openconstruct-jetson PUBLIC ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(openconstruct-jetson PUBLIC ${TENSORRT_LIBRARY})
endif()

target_link_libraries(openconstruct-jetson PUBLIC
    ${CUDA_LIBRARIES}
    cudart
    pthread
)

if(ENABLE_MOCK_CUDA)
    target_compile_definitions(openconstruct-jetson PRIVATE MOCK_CUDA)
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
message(STATUS "  Mock CUDA: ${ENABLE_MOCK_CUDA}")
message(STATUS "  Use TensorRT: ${USE_TENSORRT}")
message(STATUS "  CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "")