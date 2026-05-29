# OpenConstruct Jetson

GPU-accelerated edge node for OpenConstruct - runs local inference, processes camera/sonar data, and serves as a Plato shell with real sensory capabilities on NVIDIA Jetson devices.

## Overview

OpenConstruct Jetson provides:

- **CudaSense**: GPU-accelerated sensory processing (image classification → text, audio FFT → text)
- **LocalInference**: Run TensorRT/ONNX models locally for zero-latency sense translation
- **PlatoJetson**: Plato shell that connects to the OpenConstruct network
- **JetsonStatus**: Report GPU utilization, memory, temperature, model status as text

## Supported Hardware

- NVIDIA Jetson Nano
- NVIDIA Jetson TX2
- NVIDIA Jetson Xavier NX
- NVIDIA Jetson AGX Xavier
- NVIDIA Jetson Orin (Nano/NX/AGX)

## Requirements

### On Jetson Device
- JetPack SDK 4.6+ or JetPack 5.0+
- CUDA 11.4+
- TensorRT 8.x (optional but recommended)
- CMake 3.18+
- C++17 compatible compiler

### For Development/Testing (x86_64)
- CUDA Toolkit 11.4+ (for testing CUDA code on desktop)
- CMake 3.18+
- C++17 compatible compiler
- Google Test (for running tests)

## Building

### On Jetson Device (Production)

```bash
# Clone the repository
git clone https://github.com/SuperInstance/openconstruct-jetson.git
cd openconstruct-jetson

# Create build directory
mkdir build && cd build

# Configure with TensorRT support
cmake .. -DUSE_TENSORRT=ON -DBUILD_TESTS=OFF

# Build
cmake --build . -j$(nproc)

# Install (optional)
sudo cmake --install .
```

### For Testing on x86_64 (Mock Mode)

```bash
# Clone the repository
git clone https://github.com/SuperInstance/openconstruct-jetson.git
cd openconstruct-jetson

# Create build directory
mkdir build && cd build

# Configure with mock CUDA (no real CUDA device required)
cmake .. -DENABLE_MOCK_CUDA=ON -DBUILD_TESTS=ON

# Build
cmake --build . -j$(nproc)

# Run tests
ctest --output-on-failure
```

## Configuration

Create a configuration file (e.g., `openconstruct-jetson.conf`):

```ini
# GPU Configuration
gpu_device_id=0

# Camera Settings
camera_width=640
camera_height=480

# Audio Settings
audio_sample_rate=16000

# Model Settings
model_path=/opt/openconstruct/models/scene_classifier.onnx
enable_tensorrt=true

# Testing/Development
enable_mock_mode=false
```

## Usage

### Basic Example

```cpp
#include "openconstruct-jetson.hpp"

int main() {
    openconstruct::jetson::OpenConstructJetson jetson;

    // Initialize with config
    jetson.init("openconstruct-jetson.conf");

    // Register sensors
    jetson.register_camera(0, "front_camera");
    jetson.register_microphone(0, "main_microphone");

    // Process commands (Plato shell interface)
    jetson.process_command("status");
    jetson.process_command("describe scene");
    jetson.process_command("describe audio");

    // Get system status
    std::string status = jetson.system_status();
    std::cout << status << std::endl;

    // Run main loop (blocking)
    jetson.run();

    return 0;
}
```

### Building Your Application

```bash
g++ -std=c++17 your_app.cpp -I/usr/local/include -L/usr/local/lib -lopenconstruct-jetson -lcudart -pthread -o your_app
```

## API Reference

### `OpenConstructJetson`

Main class for OpenConstruct Jetson functionality.

#### Methods

- `void init(const char* config_path)` - Initialize with configuration file
- `void register_camera(int device_id, const char* name)` - Register a camera
- `void register_microphone(int device_id, const char* name)` - Register a microphone
- `std::string describe_scene()` - Get GPU-accelerated scene description
- `std::string describe_audio()` - Get GPU-accelerated audio description
- `std::string system_status()` - Get formatted system status
- `void process_command(const std::string& cmd)` - Process Plato shell command
- `void run()` - Start main event loop (blocking)
- `void stop()` - Stop the main loop
- `bool is_initialized()` - Check if initialized
- `bool cuda_available()` - Check if CUDA is available

### Plato Shell Commands

Commands processed via `process_command()`:

- `status` - Show system status
- `describe [scene|audio]` - Describe current scene or audio
- `restart` - Restart the service
- `camera list` - List registered cameras
- `camera add <id> <name>` - Register a camera
- `microphone list` - List registered microphones
- `microphone add <id> <name>` - Register a microphone
- `ping` - Test connection (returns PONG)
- `echo <text>` - Echo text back
- `help` - Show available commands

## Model Integration

### ONNX Models

Place your ONNX models in the configured directory:

```bash
mkdir -p /opt/openconstruct/models/
cp your_model.onnx /opt/openconstruct/models/
```

Update config:
```ini
model_path=/opt/openconstruct/models/your_model.onnx
```

### TensorRT Engine Files

For maximum performance, convert ONNX to TensorRT engine:

```bash
# On Jetson device
trtexec --onnx=your_model.onnx \
        --saveEngine=your_model.trt \
        --fp16 \
        --workspace=1024
```

Update config:
```ini
model_path=/opt/openconstruct/models/your_model.trt
```

## Flashing and Deployment

### Prerequisites

- Host machine with Ubuntu 18.04/20.04
- SDK Manager from NVIDIA
- Jetson device with USB-C recovery mode enabled

### Flashing Jetson with JetPack

1. **Download SDK Manager**:
   ```bash
   wget https://developer.download.nvidia.com/embedded/L4T/r32_Release_v4.4/r32_Release_v4.4/sdkmanager_linux_5.x.x.x-xxxx_amd64.deb
   sudo apt install ./sdkmanager_linux_5.x.x.x-xxxx_amd64.deb
   ```

2. **Run SDK Manager**:
   ```bash
   sdkmanager
   ```

3. **Select your Jetson device** and choose:
   - JetPack SDK (latest)
   - CUDA Toolkit
   - cuDNN
   - TensorRT
   - OpenCV
   - VPI

4. **Follow the flashing process** (takes 30-60 minutes)

5. **Post-flash setup**:
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y

   # Set performance mode
   sudo nvpmodel -m 0  # Maximum performance
   sudo jetson_clocks

   # Install dependencies
   sudo apt install -y cmake git libopencv-dev

   # Clone and build OpenConstruct Jetson
   git clone https://github.com/SuperInstance/openconstruct-jetson.git
   cd openconstruct-jetson
   mkdir build && cd build
   cmake .. -DUSE_TENSORRT=ON
   cmake --build . -j$(nproc)
   sudo cmake --install .
   ```

### systemd Service (Auto-start)

Create `/etc/systemd/system/openconstruct-jetson.service`:

```ini
[Unit]
Description=OpenConstruct Jetson Edge Node
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/openconstruct
ExecStart=/usr/local/bin/openconstruct-jetson-daemon /opt/openconstruct/jetson.conf
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable openconstruct-jetson
sudo systemctl start openconstruct-jetson
```

## Testing

### Running Unit Tests

```bash
# Build with tests
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
cmake --build .

# Run all tests
ctest

# Run with verbose output
ctest --verbose

# Run specific test
./tests/test_jetson --gtest_filter="OpenConstructJetsonTest.SystemStatusReturnsText"
```

### Mock Mode for Non-Jetson Testing

When `ENABLE_MOCK_CUDA=ON`, CUDA calls are mocked, allowing you to:
- Develop and test on x86_64 machines
- Run tests in CI/CD pipelines without Jetson hardware
- Verify logic without GPU dependencies

## Troubleshooting

### CUDA Errors

```
Error: No CUDA devices found
```
**Solution**: Check JetPack installation and ensure GPU drivers are loaded:
```bash
nvidia-smi
jetson_release
```

### TensorRT Import Failures

```
Error: Could not find NvInfer.h
```
**Solution**: Install TensorRT via JetPack or set `TENSORRT_ROOT`:
```bash
export TENSORRT_ROOT=/usr/src/tensorrt
cmake .. -DTENSORRT_ROOT=$TENSORRT_ROOT
```

### Out of Memory

```
Error: CUDA out of memory
```
**Solution**: Reduce model input size or reduce batch size in config:
```ini
camera_width=320
camera_height=240
```

### Thermal Throttling

Check thermal status:
```bash
sudo tegrastats
```

If temperature exceeds 85°C, consider:
- Adding cooling (fans, heatsink)
- Reducing performance mode (`sudo nvpmodel -m 1`)
- Optimizing model (quantization, pruning)

## Performance Tuning

### Enable Maximum Performance

```bash
sudo nvpmodel -m 0  # Max performance mode
sudo jetson_clocks  # Max clock frequencies
```

### Use TensorRT with FP16

```ini
enable_tensorrt=true
model_path=/path/to/model.trt
```

### GPU Power Mode

```bash
sudo nvpmodel -q  # Show available modes
sudo nvpmodel -m 0  # Set to max power mode
```

## Development

### Project Structure

```
openconstruct-jetson/
├── include/
│   └── openconstruct-jetson.hpp   # Public API
├── src/
│   ├── cuda_sense.cu               # CUDA kernels
│   ├── local_inference.cpp         # TensorRT/ONNX wrapper
│   ├── plato_jetson.cpp            # Plato shell
│   └── jetson_status.cpp           # System monitoring
├── tests/
│   ├── test_jetson.cpp             # Unit tests
│   └── CMakeLists.txt              # Test build config
├── CMakeLists.txt                  # Main build config
└── README.md                       # This file
```

### Adding New Features

1. Add public API to `include/openconstruct-jetson.hpp`
2. Implement in appropriate source file
3. Add tests to `tests/test_jetson.cpp`
4. Update documentation

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

- Issues: https://github.com/SuperInstance/openconstruct-jetson/issues
- Discussions: https://github.com/SuperInstance/openconstruct-jetson/discussions

## Acknowledgments

- NVIDIA Jetson Platform
- TensorRT for high-performance inference
- OpenConstruct community