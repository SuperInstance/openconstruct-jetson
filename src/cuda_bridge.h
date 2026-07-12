#pragma once

// Bridge declarations for the CUDA entry points.
//
// These are the only functions the C++ control flow (jetson_status.cpp and
// local_inference.cpp) is allowed to call to reach the GPU. They have two
// implementations, selected at build time:
//
//   * cuda_sense.cu      - real CUDA kernels  (USE_CUDA=ON)
//   * cuda_sense_mock.cpp - CPU mock           (USE_CUDA=OFF)
//
// Declaring them once here (rather than letting callers rely on a definition
// that happens to live in the same translation unit) keeps the two
// implementations honest: if a signature drifts, the build fails loudly instead
// of producing an ODR violation or an implicit-declaration call.

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

void*  cuda_create_stream(void);
void   cuda_destroy_stream(void* stream);

void   cuda_image_to_grayscale(const unsigned char* d_input,
                               unsigned char* d_output,
                               int width,
                               int height,
                               void* stream);

void   cuda_audio_preprocess(const float* d_input,
                             float* d_output,
                             int length,
                             void* stream);

void   cuda_extract_features(const unsigned char* d_grayscale,
                             float* d_features,
                             int width,
                             int height,
                             int feature_dim,
                             void* stream);

void   cuda_synchronize_stream(void* stream);

int    cuda_get_device_count(void);

void   cuda_get_device_properties(int device,
                                  char* name,
                                  int name_size,
                                  int* compute_major,
                                  int* compute_minor,
                                  size_t* total_mem);

size_t cuda_get_free_memory(int device);

#ifdef __cplusplus
} // extern "C"
#endif
