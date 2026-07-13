// CPU/mock implementation of the CUDA entry points declared in cuda_sense.cu.
//
// This translation unit is compiled instead of cuda_sense.cu when the project
// is built without a CUDA toolkit (USE_CUDA=OFF), so the library still links
// and the unit tests can exercise the real C++ control flow on plain x86_64
// hosts.
//
// The function signatures intentionally match the `extern "C"` block in
// cuda_sense.cu one-for-one.

#include "cuda_bridge.h"
#include <cstring>

extern "C" {

// A small, non-null sentinel so callers can distinguish "no stream" from a
// created stream. It must be non-null because OpenConstructJetson checks
// `cuda_stream_` for null to decide whether cleanup is needed.
static unsigned char g_mock_stream_storage = 1;

void *cuda_create_stream() { return &g_mock_stream_storage; }

void cuda_destroy_stream(void *stream) {
  (void)stream; // no-op in mock mode
}

void cuda_image_to_grayscale(const unsigned char * /*d_input*/,
                             unsigned char * /*d_output*/, int /*width*/,
                             int /*height*/, void * /*stream*/) {
  // No-op: in mock mode describe_scene() returns canned text and never feeds
  // real frames through this path.
}

void cuda_audio_preprocess(const float * /*d_input*/, float * /*d_output*/,
                           int /*length*/, void * /*stream*/) {
  // No-op (see above).
}

void cuda_extract_features(const unsigned char * /*d_grayscale*/,
                           float * /*d_features*/, int /*width*/,
                           int /*height*/, int /*feature_dim*/,
                           void * /*stream*/) {
  // No-op (see above).
}

void cuda_synchronize_stream(void *stream) {
  (void)stream; // no-op in mock mode
}

int cuda_get_device_count() {
  return 1; // report a single mock device so the control flow proceeds
}

void cuda_get_device_properties(int device, char *name, int name_size,
                                int *compute_major, int *compute_minor,
                                size_t *total_mem) {
  (void)device; // single mock device
  static const char mock_name[] = "Mock Jetson GPU";
  if (name && name_size > 0) {
    std::strncpy(name, mock_name, static_cast<size_t>(name_size) - 1);
    name[name_size - 1] = '\0';
  }
  if (compute_major)
    *compute_major = 7;
  if (compute_minor)
    *compute_minor = 2;
  if (total_mem)
    *total_mem = 8ULL * 1024 * 1024 * 1024; // 8 GB
}

size_t cuda_get_free_memory(int device) {
  (void)device;
  return 4ULL * 1024 * 1024 * 1024; // 4 GB free
}

} // extern "C"
