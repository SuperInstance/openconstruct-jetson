#include "openconstruct-jetson.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

#ifdef USE_TENSORRT
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#endif

namespace openconstruct {
namespace jetson {

#ifdef USE_TENSORRT
class TensorRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

class LocalInferenceEngine {
public:
    LocalInferenceEngine() : engine_(nullptr), context_(nullptr), logger_() {}

    bool load_model(const std::string& model_path) {
#ifdef MOCK_CUDA
        std::cout << "[Mock] Loading TensorRT model from: " << model_path << std::endl;
        model_loaded_ = true;
        return true;
#endif

        // Check if it's ONNX or TensorRT engine
        if (model_path.find(".onnx") != std::string::npos) {
            return load_onnx_model(model_path);
        } else {
            return load_engine(model_path);
        }
    }

    bool load_onnx_model(const std::string& onnx_path) {
#ifdef MOCK_CUDA
        model_loaded_ = true;
        return true;
#endif

        std::ifstream file(onnx_path, std::ios::binary);
        if (!file.good()) {
            std::cerr << "Could not read ONNX model: " << onnx_path << std::endl;
            return false;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string model_data = buffer.str();

        auto builder = nvinfer1::createInferBuilder(logger_);
        if (!builder) return false;

        const auto explicitBatch = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(explicitBatch);
        if (!network) {
            builder->destroy();
            return false;
        }

        auto parser = nvonnxparser::createParser(*network, logger_);
        if (!parser) {
            network->destroy();
            builder->destroy();
            return false;
        }

        if (!parser->parse(model_data.data(), model_data.size())) {
            std::cerr << "Failed to parse ONNX model" << std::endl;
            for (int i = 0; i < parser->getNbErrors(); ++i) {
                std::cerr << parser->getError(i)->desc() << std::endl;
            }
            parser->destroy();
            network->destroy();
            builder->destroy();
            return false;
        }

        auto config = builder->createBuilderConfig();
        if (!config) {
            parser->destroy();
            network->destroy();
            builder->destroy();
            return false;
        }

        // Enable FP16 if supported
        if (builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        engine_ = builder->buildEngineWithConfig(*network, *config);
        if (!engine_) {
            config->destroy();
            parser->destroy();
            network->destroy();
            builder->destroy();
            return false;
        }

        context_ = engine_->createExecutionContext();

        config->destroy();
        parser->destroy();
        network->destroy();
        builder->destroy();

        model_loaded_ = true;
        return true;
    }

    bool load_engine(const std::string& engine_path) {
#ifdef MOCK_CUDA
        model_loaded_ = true;
        return true;
#endif

        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            std::cerr << "Could not read TensorRT engine: " << engine_path << std::endl;
            return false;
        }

        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);

        auto runtime = nvinfer1::createInferRuntime(logger_);
        if (!runtime) return false;

        engine_ = runtime->deserializeCudaEngine(engine_data.data(), size);
        if (!engine_) {
            runtime->destroy();
            return false;
        }

        context_ = engine_->createExecutionContext();

        runtime->destroy();
        model_loaded_ = true;
        return true;
    }

    bool infer(void* input_data, void* output_data) {
        if (!model_loaded_ || !context_) {
            return false;
        }

#ifdef MOCK_CUDA
        // Mock inference - return fake results
        float* output = static_cast<float*>(output_data);
        output[0] = 0.85f; // Confidence score
        return true;
#endif

        // Get input/output bindings
        void* buffers[2];
        int input_index = engine_->getBindingIndex("input");
        int output_index = engine_->getBindingIndex("output");

        // Allocate GPU memory (in production, cache these)
        size_t input_size = get_binding_size(input_index);
        size_t output_size = get_binding_size(output_index);

        void* d_input;
        void* d_output;
        cudaMalloc(&d_input, input_size);
        cudaMalloc(&d_output, output_size);

        // Copy input to GPU
        cudaMemcpy(d_input, input_data, input_size, cudaMemcpyHostToDevice);

        buffers[input_index] = d_input;
        buffers[output_index] = d_output;

        // Execute inference
        context_->executeV2(buffers);

        // Copy output to host
        cudaMemcpy(output_data, d_output, output_size, cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_input);
        cudaFree(d_output);

        return true;
    }

    bool is_loaded() const { return model_loaded_; }

    ~LocalInferenceEngine() {
        if (context_) context_->destroy();
        if (engine_) engine_->destroy();
    }

private:
    size_t get_binding_size(int binding_index) {
        nvinfer1::Dims dims = engine_->getBindingDimensions(binding_index);
        size_t size = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            size *= dims.d[i];
        }
        size *= sizeof(float); // Assuming float32
        return size;
    }

    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    TensorRTLogger logger_;
    bool model_loaded_ = false;
};
#endif

// Global inference engine instance
#ifdef USE_TENSORRT
static LocalInferenceEngine g_inference_engine;
#endif

bool load_inference_model(const std::string& model_path) {
#ifdef USE_TENSORRT
    return g_inference_engine.load_model(model_path);
#else
    std::cout << "[Info] TensorRT not enabled, using mock inference" << std::endl;
    return true;
#endif
}

bool run_inference(void* input_data, void* output_data) {
#ifdef USE_TENSORRT
    return g_inference_engine.infer(input_data, output_data);
#else
    // Mock inference
    float* output = static_cast<float*>(output_data);
    output[0] = 0.75f; // Fake confidence
    return true;
#endif
}

} // namespace jetson
} // namespace openconstruct