#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

namespace openconstruct {
namespace jetson {

/**
 * @brief Main class for OpenConstruct Jetson - GPU-accelerated edge node
 *
 * Provides CUDA-accelerated sensory processing, local inference,
 * and Plato shell capabilities for NVIDIA Jetson devices.
 */
class OpenConstructJetson {
public:
    struct Config {
        std::string model_path;
        int gpu_device_id = 0;
        int camera_width = 640;
        int camera_height = 480;
        int audio_sample_rate = 16000;
        bool enable_tensorrt = true;
        bool enable_mock_mode = false; // For testing on non-Jetson systems
    };

    struct Sensor {
        int device_id;
        std::string name;
        bool active = false;
    };

    OpenConstructJetson();
    ~OpenConstructJetson();

    /**
     * @brief Initialize the Jetson node with configuration
     */
    void init(const char* config_path);

    /**
     * @brief Register a camera device for scene perception
     */
    void register_camera(int device_id, const char* name);

    /**
     * @brief Register a microphone for audio perception
     */
    void register_microphone(int device_id, const char* name);

    /**
     * @brief GPU-accelerated scene description
     * @return Text description of the current scene
     */
    std::string describe_scene();

    /**
     * @brief GPU-accelerated audio description
     * @return Text description of current audio input
     */
    std::string describe_audio();

    /**
     * @brief Get system status including GPU/CPU/memory/thermal
     * @return Formatted status string
     */
    std::string system_status();

    /**
     * @brief Process a text command from Plato network
     */
    void process_command(const std::string& cmd);

    /**
     * @brief Main event loop - blocks until stopped
     */
    void run();

    /**
     * @brief Stop the main loop
     */
    void stop();

    /**
     * @brief Check if system is initialized
     */
    bool is_initialized() const { return initialized_; }

    /**
     * @brief Check if CUDA is available
     */
    bool cuda_available() const;

private:
    bool initialized_;
    bool running_;
    Config config_;
    std::map<int, Sensor> cameras_;
    std::map<int, Sensor> microphones_;
    int next_camera_id_;
    int next_microphone_id_;

    void* cuda_stream_;  // CUstream or cudaStream_t

    // Private implementation methods
    bool load_config(const char* path);
    void init_cuda();
    void cleanup_cuda();
    std::string format_gpu_status();
    std::string format_cpu_status();
    std::string format_memory_status();
    std::string format_thermal_status();
    std::string mock_scene_description();
    std::string mock_audio_description();
};

} // namespace jetson
} // namespace openconstruct