#include "openconstruct-jetson.hpp"
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>

#ifdef MOCK_CUDA
// Mock CUDA functions for testing on non-Jetson systems
extern "C" {
    void* cuda_create_stream() { return reinterpret_cast<void*>(0xDEADBEEF); }
    void cuda_destroy_stream(void* stream) { (void)stream; }
    void cuda_image_to_grayscale(const unsigned char*, unsigned char*, int, int, void*) {}
    void cuda_audio_preprocess(const float*, float*, int, void*) {}
    void cuda_extract_features(const unsigned char*, float*, int, int, int, void*) {}
    void cuda_synchronize_stream(void* stream) {}
    int cuda_get_device_count() { return 1; }
    void cuda_get_device_properties(int, char*, int, int*, int*, size_t*) {
        static const char* mock_name = "Mock Jetson GPU";
        strcpy(name, mock_name);
        *compute_major = 7;
        *compute_minor = 2;
        *total_mem = 8ULL * 1024 * 1024 * 1024; // 8GB
    }
    size_t cuda_get_free_memory(int) { return 4ULL * 1024 * 1024 * 1024; } // 4GB free
}
#endif

namespace openconstruct {
namespace jetson {

OpenConstructJetson::OpenConstructJetson()
    : initialized_(false)
    , running_(false)
    , next_camera_id_(0)
    , next_microphone_id_(0)
    , cuda_stream_(nullptr)
{
}

OpenConstructJetson::~OpenConstructJetson() {
    cleanup_cuda();
}

void OpenConstructJetson::init(const char* config_path) {
    if (initialized_) {
        std::cerr << "Warning: Already initialized" << std::endl;
        return;
    }

    // Load configuration
    if (config_path && std::string(config_path) != "") {
        if (!load_config(config_path)) {
            std::cerr << "Warning: Failed to load config from " << config_path
                      << ", using defaults" << std::endl;
        }
    }

    // Initialize CUDA
    init_cuda();

    initialized_ = true;
    std::cout << "OpenConstruct Jetson initialized successfully" << std::endl;
}

void OpenConstructJetson::register_camera(int device_id, const char* name) {
    if (!name) {
        std::cerr << "Error: Camera name cannot be null" << std::endl;
        return;
    }

    Sensor sensor;
    sensor.device_id = device_id;
    sensor.name = name;
    sensor.active = false;

    cameras_[next_camera_id_++] = sensor;
    std::cout << "Registered camera '" << name << "' with device ID " << device_id << std::endl;
}

void OpenConstructJetson::register_microphone(int device_id, const char* name) {
    if (!name) {
        std::cerr << "Error: Microphone name cannot be null" << std::endl;
        return;
    }

    Sensor sensor;
    sensor.device_id = device_id;
    sensor.name = name;
    sensor.active = false;

    microphones_[next_microphone_id_++] = sensor;
    std::cout << "Registered microphone '" << name << "' with device ID " << device_id << std::endl;
}

std::string OpenConstructJetson::describe_scene() {
    if (!initialized_) {
        return "ERROR: Not initialized. Call init() first.";
    }

    if (cameras_.empty()) {
        return "No cameras registered. Cannot describe scene.";
    }

    // In a real implementation, this would:
    // 1. Capture frame from camera
    // 2. Run CUDA preprocessing
    // 3. Run inference
    // 4. Convert output to text

    // For now, return a mock description
    return mock_scene_description();
}

std::string OpenConstructJetson::describe_audio() {
    if (!initialized_) {
        return "ERROR: Not initialized. Call init() first.";
    }

    if (microphones_.empty()) {
        return "No microphones registered. Cannot describe audio.";
    }

    // In a real implementation, this would:
    // 1. Capture audio samples
    // 2. Run CUDA FFT preprocessing
    // 3. Run inference
    // 4. Convert output to text

    // For now, return a mock description
    return mock_audio_description();
}

std::string OpenConstructJetson::system_status() {
    std::ostringstream oss;

    oss << "=== OpenConstruct Jetson Status ===\n\n";
    oss << format_gpu_status() << "\n";
    oss << format_cpu_status() << "\n";
    oss << format_memory_status() << "\n";
    oss << format_thermal_status() << "\n";
    oss << "\n=== Sensors ===\n";
    oss << "  Cameras: " << cameras_.size() << " registered\n";
    oss << "  Microphones: " << microphones_.size() << " registered\n";

    return oss.str();
}

void OpenConstructJetson::process_command(const std::string& cmd) {
    // Delegate to Plato shell
    class PlatoJetsonImpl {
    public:
        static void execute(OpenConstructJetson* parent, const std::string& cmd) {
            PlatoJetson shell(parent);
            std::string response = shell.process(cmd);
            std::cout << response << std::endl;
        }
    };

    PlatoJetsonImpl::execute(this, cmd);
}

void OpenConstructJetson::run() {
    if (!initialized_) {
        std::cerr << "Error: Not initialized. Call init() first." << std::endl;
        return;
    }

    running_ = true;
    std::cout << "Starting OpenConstruct Jetson main loop..." << std::endl;

    // Main event loop
    while (running_) {
        // In a real implementation, this would:
        // - Listen for network commands from Plato
        // - Periodically capture and process sensory data
        // - Handle system monitoring
        // - Send updates to the OpenConstruct network

        // For now, just sleep to simulate event processing
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Check for stopping condition (e.g., signal handler)
        // This would be set by a signal handler in production
        if (false) {  // Placeholder for actual stop condition
            running_ = false;
        }
    }

    std::cout << "OpenConstruct Jetson main loop stopped" << std::endl;
}

void OpenConstructJetson::stop() {
    running_ = false;
}

bool OpenConstructJetson::cuda_available() const {
#ifdef MOCK_CUDA
    return true;
#else
    int count = cuda_get_device_count();
    return count > 0;
#endif
}

// Private methods

bool OpenConstructJetson::load_config(const char* path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Simple key=value parsing
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);

            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            if (key == "model_path") {
                config_.model_path = value;
            } else if (key == "gpu_device_id") {
                config_.gpu_device_id = std::stoi(value);
            } else if (key == "camera_width") {
                config_.camera_width = std::stoi(value);
            } else if (key == "camera_height") {
                config_.camera_height = std::stoi(value);
            } else if (key == "audio_sample_rate") {
                config_.audio_sample_rate = std::stoi(value);
            } else if (key == "enable_tensorrt") {
                config_.enable_tensorrt = (value == "true" || value == "1");
            } else if (key == "enable_mock_mode") {
                config_.enable_mock_mode = (value == "true" || value == "1");
            }
        }
    }

    return true;
}

void OpenConstructJetson::init_cuda() {
    if (config_.enable_mock_mode) {
        std::cout << "Running in mock CUDA mode" << std::endl;
        cuda_stream_ = cuda_create_stream();
        return;
    }

    int device_count = cuda_get_device_count();
    if (device_count == 0) {
        std::cerr << "Warning: No CUDA devices found" << std::endl;
        return;
    }

    if (config_.gpu_device_id >= device_count) {
        std::cerr << "Warning: GPU device ID " << config_.gpu_device_id
                  << " not available, using device 0" << std::endl;
        config_.gpu_device_id = 0;
    }

    // Create CUDA stream for async operations
    cuda_stream_ = cuda_create_stream();

    std::cout << "CUDA initialized on device " << config_.gpu_device_id << std::endl;
}

void OpenConstructJetson::cleanup_cuda() {
    if (cuda_stream_) {
        cuda_destroy_stream(cuda_stream_);
        cuda_stream_ = nullptr;
    }
}

std::string OpenConstructJetson::format_gpu_status() {
    std::ostringstream oss;

    int device_count = cuda_get_device_count();
    oss << "=== GPU Status ===\n";
    oss << "  Available devices: " << device_count << "\n";

    if (device_count > 0) {
        char name[256];
        int major, minor;
        size_t total_mem;

        cuda_get_device_properties(config_.gpu_device_id, name, sizeof(name),
                                    &major, &minor, &total_mem);

        size_t free_mem = cuda_get_free_memory(config_.gpu_device_id);
        double used_mem_gb = (total_mem - free_mem) / (1024.0 * 1024.0 * 1024.0);
        double total_mem_gb = total_mem / (1024.0 * 1024.0 * 1024.0);
        double free_mem_gb = free_mem / (1024.0 * 1024.0 * 1024.0);
        double utilization = (1.0 - static_cast<double>(free_mem) / total_mem) * 100.0;

        oss << "  Active device: " << config_.gpu_device_id << "\n";
        oss << "  Name: " << name << "\n";
        oss << "  Compute capability: " << major << "." << minor << "\n";
        oss << "  Memory: " << free_mem_gb << " GB free / "
            << total_mem_gb << " GB total (" << static_cast<int>(utilization) << "% used)\n";
    }

    return oss.str();
}

std::string OpenConstructJetson::format_cpu_status() {
    std::ostringstream oss;

    oss << "=== CPU Status ===\n";
    // In a real implementation, read from /proc/stat
    oss << "  Cores: 6 (ARM Cortex-A57/A53)\n";
    oss << "  Load: 0.45, 0.38, 0.32 (1, 5, 15 min)\n";

    return oss.str();
}

std::string OpenConstructJetson::format_memory_status() {
    std::ostringstream oss;

    oss << "=== System Memory Status ===\n";
    // In a real implementation, read from /proc/meminfo
    oss << "  RAM: 3.2 GB free / 8.0 GB total\n";
    oss << "  Swap: 4.0 GB free / 4.0 GB total\n";

    return oss.str();
}

std::string OpenConstructJetson::format_thermal_status() {
    std::ostringstream oss;

    oss << "=== Thermal Status ===\n";
    // In a real implementation, read from /sys/class/thermal
    oss << "  CPU: 45°C\n";
    oss << "  GPU: 42°C\n";
    oss << "  SOC: 47°C\n";

    return oss.str();
}

std::string OpenConstructJetson::mock_scene_description() {
    // Mock scene descriptions for testing
    static const char* scenes[] = {
        "A well-lit indoor room with furniture and windows",
        "An outdoor scene with trees and sky visible",
        "A workspace with computer equipment on a desk",
        "A hallway with doors on both sides",
        "A kitchen area with appliances visible"
    };

    // Simple hash based on time to return consistent but varying results
    auto now = std::chrono::system_clock::now();
    auto timestamp = now.time_since_epoch().count();
    int idx = static_cast<int>(timestamp % 5);

    return scenes[idx];
}

std::string OpenConstructJetson::mock_audio_description() {
    // Mock audio descriptions for testing
    static const char* audio_descriptions[] = {
        "Quiet ambient noise, low background activity",
        "Human speech detected in conversation",
        "Mechanical sounds, possibly from equipment",
        "Nature sounds - birds or wind detected",
        "Music or rhythmic audio playing"
    };

    auto now = std::chrono::system_clock::now();
    auto timestamp = now.time_since_epoch().count();
    int idx = static_cast<int>((timestamp / 3) % 5);

    return audio_descriptions[idx];
}

} // namespace jetson
} // namespace openconstruct