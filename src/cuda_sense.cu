#include "openconstruct-jetson.hpp"
#include "cuda_bridge.h"
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>

namespace openconstruct {
namespace jetson {

// CUDA kernels for sensory processing

/**
 * @brief GPU-accelerated image processing kernel
 * Converts RGB image to grayscale and extracts features
 */
__global__ void image_to_grayscale_kernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        int out_idx = y * width + x;

        // Standard grayscale conversion: 0.299R + 0.587G + 0.114B
        output[out_idx] = static_cast<unsigned char>(
            0.299f * input[idx] +
            0.587f * input[idx + 1] +
            0.114f * input[idx + 2]
        );
    }
}

/**
 * @brief Audio FFT preprocessing kernel
 * Applies windowing function and prepares for FFT
 */
__global__ void audio_preprocess_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length) {
        // Hann window function. Guard the length==1 case to avoid dividing by
        // zero (the window would be degenerate anyway); use 1 as the divisor
        // so a single sample yields the neutral window value of 0.0.
        const int denom = (length > 1) ? (length - 1) : 1;
        float window = 0.5f * (1.0f - cosf(2.0f * M_PI * idx / denom));
        output[idx] = input[idx] * window;
    }
}

/**
 * @brief Feature extraction kernel for scene classification
 */
__global__ void feature_extraction_kernel(
    const unsigned char* __restrict__ grayscale,
    float* __restrict__ features,
    int width,
    int height,
    int feature_dim
) {
    int feature_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (feature_idx < feature_dim) {
        // Simple feature extraction: histogram-like computation
        // In production, this would use more sophisticated features
        int bin_size = (width * height) / feature_dim;
        int start = feature_idx * bin_size;
        int end = min(start + bin_size, width * height);

        float sum = 0.0f;
        for (int i = start; i < end; ++i) {
            sum += grayscale[i];
        }
        features[feature_idx] = sum / (end - start);
    }
}

// C wrapper functions called from C++ code

extern "C" {

void* cuda_create_stream() {
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream: " +
                                std::string(cudaGetErrorString(err)));
    }
    return stream;
}

void cuda_destroy_stream(void* stream) {
    cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}

void cuda_image_to_grayscale(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height,
    void* stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    image_to_grayscale_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        d_input, d_output, width, height
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

void cuda_audio_preprocess(
    const float* d_input,
    float* d_output,
    int length,
    void* stream
) {
    int threads = 256;
    int blocks = (length + threads - 1) / threads;

    audio_preprocess_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        d_input, d_output, length
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA audio preprocessing failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

void cuda_extract_features(
    const unsigned char* d_grayscale,
    float* d_features,
    int width,
    int height,
    int feature_dim,
    void* stream
) {
    int threads = 256;
    int blocks = (feature_dim + threads - 1) / threads;

    feature_extraction_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        d_grayscale, d_features, width, height, feature_dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA feature extraction failed: " +
                                std::string(cudaGetErrorString(err)));
    }
}

void cuda_synchronize_stream(void* stream) {
    cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
}

int cuda_get_device_count() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return count;
}

void cuda_get_device_properties(
    int device,
    char* name,
    int name_size,
    int* compute_major,
    int* compute_minor,
    size_t* total_mem
) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err == cudaSuccess) {
        strncpy(name, prop.name, name_size - 1);
        name[name_size - 1] = '\0';
        *compute_major = prop.major;
        *compute_minor = prop.minor;
        *total_mem = prop.totalGlobalMem;
    } else {
        name[0] = '\0';
        *compute_major = 0;
        *compute_minor = 0;
        *total_mem = 0;
    }
}

size_t cuda_get_free_memory(int device) {
    size_t free, total;
    cudaSetDevice(device);
    cudaMemGetInfo(&free, &total);
    return free;
}

} // extern "C"

} // namespace jetson
} // namespace openconstruct