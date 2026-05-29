# OpenConstruct Jetson — GPU-Accelerated Edge Node

Run local inference, process camera and sonar data, and serve as a Plato shell with real sensory capabilities on NVIDIA Jetson devices.

**Part of [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct).**

## What This Gives You

- **CudaSense** — GPU-accelerated sensory processing (image → text, audio FFT → text)
- **LocalInference** — TensorRT/ONNX models for zero-latency sense translation
- **PlatoJetson** — Plato shell connecting to the OpenConstruct network
- **JetsonStatus** — GPU utilization, memory, temperature, model status as text
- **All Jetson variants** — Nano, TX2, Xavier NX, AGX Xavier, Orin (Nano/NX/AGX)

## Quick Start

```cpp
#include <openconstruct-jetson.hpp>

// Initialize CUDA sensor
auto sense = CudaSense::create();
auto text = sense.image_to_text(camera_frame);
// "Scene: kitchen table with laptop, coffee cup, and keyboard"

// Run local inference
auto inference = LocalInference::create("model.onnx");
auto result = inference.run(input_tensor);

// Plato shell
auto shell = PlatoJetson::create("jetson-hub-1");
shell.connect("ws://coordinator:9142");
shell.report_status();
```

## Building

```bash
mkdir build && cd build
cmake .. -DUSE_TENSORRT=ON
cmake --build . -j$(nproc)
```

### Prerequisites

- JetPack SDK 4.6+ or 5.0+
- CUDA 11.4+
- CMake 3.18+, C++17 compiler
- (Optional) TensorRT 8.x

### x86_64 Testing

```bash
cmake .. -DUSE_CUDA=OFF -DUSE_TENSORRT=OFF -DBUILD_TESTS=ON
cmake --build .
ctest --output-on-failure
```

## How It Fits

The Jetson is the hub in the fleet star topology — [ESP32s](https://github.com/SuperInstance/openconstruct-esp32) connect to it as spokes. [plato-fleet](https://github.com/SuperInstance/plato-fleet) manages discovery; [plato-vision](https://github.com/SuperInstance/plato-vision) and [plato-sonar-text](https://github.com/SuperInstance/plato-sonar-text) provide the sense modules.

## License

MIT
