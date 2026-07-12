# OpenConstruct Jetson — GPU-Accelerated Edge Node

A C++ edge node for NVIDIA Jetson devices: GPU-accelerated sensory preprocessing,
an optional TensorRT local-inference path, a Plato command shell, and a
system-status reporter.

> **Implementation status:** this repository is an early scaffold. Some
> capabilities are real, working C++; others are stubs or mocked for non-Jetson
> testing. Every advertised feature carries an explicit status marker below so
> you know exactly what runs today. See [Implementation status](#implementation-status).
>
> This honesty pass was added because an earlier README showed example code using
> symbols (`CudaSense::create()`, `LocalInference::create()`,
> `PlatoJetson::create()`, `.connect()`, `.report_status()`) that **do not exist
> anywhere in the codebase** and never compiled. Those examples have been replaced
> with the real, working API.

**Part of [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct).**

## Implementation status

| Feature | Status | Notes |
| --- | :---: | --- |
| **Sensory preprocessing** (CUDA kernels) | 🔧 Stub | Real CUDA kernels exist in `src/cuda_sense.cu` (RGB→grayscale, Hann-window audio preprocessing, histogram feature extraction), but they are **not yet wired** into `describe_scene()`/`describe_audio()`, which return canned mock text instead. No `CudaSense` class exists. |
| **Local inference** (TensorRT/ONNX) | 🔧 Stub | TensorRT engine + ONNX-parsing code exists in `src/local_inference.cpp` (built only with `-DUSE_TENSORRT=ON`), but its `load_inference_model()` / `run_inference()` entry points are not declared in the public header and are **not called from anywhere**. No `LocalInference::create()` class exists. |
| **Plato command shell** | 🟡 Partial | The in-process command dispatcher (`status`, `describe`, `camera`, `microphone`, `ping`, `echo`, `help`, `restart`) is implemented and unit-tested. There is **no network transport**: `connect()` / `report_status()` over WebSocket do not exist — commands are processed locally only. |
| **Jetson status** | 🟡 Partial | `system_status()` formats a status report. GPU memory figures come from the CUDA runtime API (real when CUDA is present); CPU load, RAM, swap, and thermal readings are **hardcoded placeholders** (not read from `/proc` or `/sys`). |
| **Device variants** | 🔧 Stub | CMake lists Jetson architectures (Nano … Orin AGX), but this project has only been built and tested on x86_64 in CPU-mock mode. |

Legend: 🔧 **Stub** = code/interface present but not functional · 🟡 **Partial** = core works, pieces missing · ✅ **Implemented** = works as described.

## Quick start (real, working API)

The public API is the `openconstruct::jetson::OpenConstructJetson` class:

```cpp
#include <openconstruct-jetson.hpp>

using namespace openconstruct::jetson;

OpenConstructJetson node;
node.init("config.txt");              // optional key=value config; nullptr for defaults

node.register_camera(0, "front");
node.register_microphone(0, "main");

// Mock description (see status table) — real inference is not yet wired in:
std::cout << node.describe_scene();

// GPU memory is real; CPU / RAM / thermal are placeholders:
std::cout << node.system_status();

// Plato shell, processed in-process (no network transport yet):
node.process_command("ping");                     // -> PONG
node.process_command("camera add 1 rear");         // -> OK: Camera 'rear' registered ...
node.process_command("status");

node.run();                           // event loop — blocks until node.stop()
```

`config.txt` is a simple `key=value` file with optional keys: `model_path`,
`gpu_device_id`, `camera_width`, `camera_height`, `audio_sample_rate`,
`enable_tensorrt`, `enable_mock_mode`.

## Building

### On a Jetson (real CUDA)

```bash
mkdir build && cd build
cmake .. -DUSE_CUDA=ON [-DUSE_TENSORRT=ON]
cmake --build . -j$(nproc)
```

### On x86_64 (CPU mock — what CI uses)

```bash
mkdir build && cd build
cmake .. -DUSE_CUDA=OFF -DUSE_TENSORRT=OFF -DBUILD_TESTS=ON
cmake --build . -j$(nproc)
ctest --output-on-failure
```

With `-DUSE_CUDA=OFF` the project does **not** require the CUDA toolkit or `nvcc`:
a CPU mock (`src/cuda_sense_mock.cpp`) is compiled instead of `src/cuda_sense.cu`
so the C++ control flow is fully buildable and testable on plain hosts.

### Prerequisites

- CMake 3.18+, C++17 compiler
- CUDA 11.4+ and `nvcc` — **only** for `-DUSE_CUDA=ON`
- JetPack SDK 4.6+/5.0+ — **only** on-device
- TensorRT 8.x — **only** for `-DUSE_TENSORRT=ON` (implies `-DUSE_CUDA=ON`)
- For tests: a network connection (GoogleTest is fetched via CMake
  `FetchContent`) **or** a system-installed `libgtest-dev`

## CMake options

| Option | Default | Meaning |
| --- | --- | --- |
| `USE_CUDA` | `ON` | Compile the real CUDA kernels (`src/cuda_sense.cu`); needs the CUDA toolkit. `OFF` → CPU mock. |
| `USE_TENSORRT` | `OFF` | Enable the TensorRT inference path (requires `USE_CUDA=ON`). |
| `BUILD_TESTS` | `ON` | Build the unit tests. |
| `ENABLE_MOCK_CUDA` | `OFF` | Force the CPU mock even when `USE_CUDA=ON`. |

## How it fits

The Jetson is intended as the hub in a fleet star topology —
[ESP32s](https://github.com/SuperInstance/openconstruct-esp32) connect to it as
spokes. [plato-fleet](https://github.com/SuperInstance/plato-fleet) manages
discovery; [plato-vision](https://github.com/SuperInstance/plato-vision) and
[plato-sonar-text](https://github.com/SuperInstance/plato-sonar-text) provide
the sense modules. Note: the **network integration with these siblings is not
implemented in this repository** (see the Plato shell status above).

## License

MIT
