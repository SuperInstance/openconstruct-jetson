#include <gtest/gtest.h>

// Define MOCK_CUDA to enable mock functions for testing
#ifndef MOCK_CUDA
#define MOCK_CUDA 1
#endif

#include "openconstruct-jetson.hpp"
#include <fstream>
#include <sstream>

using namespace openconstruct::jetson;

class OpenConstructJetsonTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary config file
        config_path_ = "/tmp/openconstruct_test_config.txt";
        std::ofstream config(config_path_);
        config << "gpu_device_id=0\n";
        config << "camera_width=640\n";
        config << "camera_height=480\n";
        config << "audio_sample_rate=16000\n";
        config << "enable_tensorrt=false\n";
        config << "enable_mock_mode=true\n";
        config.close();

        // Initialize the Jetson instance
        jetson_ = std::make_unique<OpenConstructJetson>();
    }

    void TearDown() override {
        jetson_.reset();
        std::remove(config_path_.c_str());
    }

    std::unique_ptr<OpenConstructJetson> jetson_;
    std::string config_path_;
};

// Test initialization
TEST_F(OpenConstructJetsonTest, InitLoadsConfig) {
    jetson_->init(config_path_.c_str());

    EXPECT_TRUE(jetson_->is_initialized());
    EXPECT_TRUE(jetson_->cuda_available());
}

TEST_F(OpenConstructJetsonTest, InitWithNullPath) {
    jetson_->init(nullptr);

    // Should still initialize with defaults
    EXPECT_TRUE(jetson_->is_initialized());
}

TEST_F(OpenConstructJetsonTest, InitWithEmptyPath) {
    jetson_->init("");

    // Should still initialize with defaults
    EXPECT_TRUE(jetson_->is_initialized());
}

// Test sensor registration
TEST_F(OpenConstructJetsonTest, CameraRegistrationWorks) {
    jetson_->init(config_path_.c_str());

    testing::internal::CaptureStdout();
    jetson_->register_camera(0, "front_camera");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("Registered camera 'front_camera'") != std::string::npos);
}

TEST_F(OpenConstructJetsonTest, MicrophoneRegistrationWorks) {
    jetson_->init(config_path_.c_str());

    testing::internal::CaptureStdout();
    jetson_->register_microphone(0, "main_mic");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("Registered microphone 'main_mic'") != std::string::npos);
}

TEST_F(OpenConstructJetsonTest, CameraWithNullName) {
    jetson_->init(config_path_.c_str());

    testing::internal::CaptureStderr();
    jetson_->register_camera(0, nullptr);
    std::string output = testing::internal::GetCapturedStderr();

    EXPECT_TRUE(output.find("Camera name cannot be null") != std::string::npos);
}

// Test system status
TEST_F(OpenConstructJetsonTest, SystemStatusReturnsText) {
    jetson_->init(config_path_.c_str());

    std::string status = jetson_->system_status();

    EXPECT_FALSE(status.empty());
    EXPECT_TRUE(status.find("=== OpenConstruct Jetson Status ===") != std::string::npos);
    EXPECT_TRUE(status.find("=== GPU Status ===") != std::string::npos);
    EXPECT_TRUE(status.find("=== CPU Status ===") != std::string::npos);
    EXPECT_TRUE(status.find("=== Memory Status ===") != std::string::npos);
    EXPECT_TRUE(status.find("=== Thermal Status ===") != std::string::npos);
    EXPECT_TRUE(status.find("=== Sensors ===") != std::string::npos);
}

// Test scene description
TEST_F(OpenConstructJetsonTest, DescribeSceneReturnsText) {
    jetson_->init(config_path_.c_str());
    jetson_->register_camera(0, "test_camera");

    std::string scene = jetson_->describe_scene();

    EXPECT_FALSE(scene.empty());
    EXPECT_TRUE(scene.find("indoor") != std::string::npos ||
                scene.find("outdoor") != std::string::npos ||
                scene.find("workspace") != std::string::npos);
}

TEST_F(OpenConstructJetsonTest, DescribeSceneWithoutCameras) {
    jetson_->init(config_path_.c_str());

    std::string scene = jetson_->describe_scene();

    EXPECT_TRUE(scene.find("No cameras registered") != std::string::npos);
}

TEST_F(OpenConstructJetsonTest, DescribeSceneNotInitialized) {
    std::string scene = jetson_->describe_scene();

    EXPECT_TRUE(scene.find("ERROR: Not initialized") != std::string::npos);
}

// Test audio description
TEST_F(OpenConstructJetsonTest, DescribeAudioReturnsText) {
    jetson_->init(config_path_.c_str());
    jetson_->register_microphone(0, "test_mic");

    std::string audio = jetson_->describe_audio();

    EXPECT_FALSE(audio.empty());
    EXPECT_TRUE(audio.find("noise") != std::string::npos ||
                audio.find("speech") != std::string::npos ||
                audio.find("sounds") != std::string::npos);
}

TEST_F(OpenConstructJetsonTest, DescribeAudioWithoutMicrophones) {
    jetson_->init(config_path_.c_str());

    std::string audio = jetson_->describe_audio();

    EXPECT_TRUE(audio.find("No microphones registered") != std::string::npos);
}

TEST_F(OpenConstructJetsonTest, DescribeAudioNotInitialized) {
    std::string audio = jetson_->describe_audio();

    EXPECT_TRUE(audio.find("ERROR: Not initialized") != std::string::npos);
}

// Test command processing
TEST_F(OpenConstructJetsonTest, ProcessCommandStatus) {
    jetson_->init(config_path_.c_str());

    testing::internal::CaptureStdout();
    jetson_->process_command("status");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("=== OpenConstruct Jetson Status ===") != std::string::npos);
}

TEST_F(OpenConstructJetsonTest, ProcessCommandRestart) {
    jetson_->init(config_path_.c_str());

    testing::internal::CaptureStdout();
    jetson_->process_command("restart");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("RESTART") != std::string::npos);
    EXPECT_TRUE(output.find("Restart command received") != std::string::npos);
}

TEST_F(OpenConstructJetsonTest, ProcessCommandPing) {
    jetson_->init(config_path_.c_str());

    testing::internal::CaptureStdout();
    jetson_->process_command("ping");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("PONG") != std::string::npos);
}

TEST_F(OpenConstructJetsonTest, ProcessCommandUnknown) {
    jetson_->init(config_path_.c_str());

    testing::internal::CaptureStdout();
    jetson_->process_command("unknown_command");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("ERROR: Unknown command") != std::string::npos);
}

TEST_F(OpenConstructJetsonTest, ProcessCommandHelp) {
    jetson_->init(config_path_.c_str());

    testing::internal::CaptureStdout();
    jetson_->process_command("help");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("Available commands:") != std::string::npos);
    EXPECT_TRUE(output.find("status") != std::string::npos);
    EXPECT_TRUE(output.find("describe") != std::string::npos);
}

TEST_F(OpenConstructJetsonTest, ProcessCommandDescribe) {
    jetson_->init(config_path_.c_str());
    jetson_->register_camera(0, "cam");
    jetson_->register_microphone(0, "mic");

    testing::internal::CaptureStdout();
    jetson_->process_command("describe");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_FALSE(output.empty());
}

TEST_F(OpenConstructJetsonTest, ProcessCommandDescribeScene) {
    jetson_->init(config_path_.c_str());
    jetson_->register_camera(0, "cam");

    testing::internal::CaptureStdout();
    jetson_->process_command("describe scene");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_FALSE(output.empty());
}

TEST_F(OpenConstructJetsonTest, ProcessCommandDescribeAudio) {
    jetson_->init(config_path_.c_str());
    jetson_->register_microphone(0, "mic");

    testing::internal::CaptureStdout();
    jetson_->process_command("describe audio");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_FALSE(output.empty());
}

TEST_F(OpenConstructJetsonTest, ProcessCommandCameraAdd) {
    jetson_->init(config_path_.c_str());

    testing::internal::CaptureStdout();
    jetson_->process_command("camera add 0 my_camera");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("OK: Camera 'my_camera' registered") != std::string::npos);
}

TEST_F(OpenConstructJetsonTest, ProcessCommandMicrophoneAdd) {
    jetson_->init(config_path_.c_str());

    testing::internal::CaptureStdout();
    jetson_->process_command("microphone add 0 my_mic");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("OK: Microphone 'my_mic' registered") != std::string::npos);
}

TEST_F(OpenConstructJetsonTest, ProcessCommandEcho) {
    jetson_->init(config_path_.c_str());

    testing::internal::CaptureStdout();
    jetson_->process_command("echo hello world");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("hello world") != std::string::npos);
}

// Test multiple registrations
TEST_F(OpenConstructJetsonTest, MultipleCameraRegistrations) {
    jetson_->init(config_path_.c_str());

    testing::internal::CaptureStdout();
    jetson_->register_camera(0, "front");
    jetson_->register_camera(1, "back");
    jetson_->register_camera(2, "top");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("front") != std::string::npos);
    EXPECT_TRUE(output.find("back") != std::string::npos);
    EXPECT_TRUE(output.find("top") != std::string::npos);
}

TEST_F(OpenConstructJetsonTest, MultipleMicrophoneRegistrations) {
    jetson_->init(config_path_.c_str());

    testing::internal::CaptureStdout();
    jetson_->register_microphone(0, "left");
    jetson_->register_microphone(1, "right");
    jetson_->register_microphone(2, "center");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("left") != std::string::npos);
    EXPECT_TRUE(output.find("right") != std::string::npos);
    EXPECT_TRUE(output.find("center") != std::string::npos);
}

// Test stop functionality
TEST_F(OpenConstructJetsonTest, StopSetsRunningFalse) {
    jetson_->init(config_path_.c_str());

    jetson_->stop();
    // In a real test, we'd verify running_ is false
    // For now, just ensure it doesn't crash
}

// CUDA availability
TEST_F(OpenConstructJetsonTest, CudaAvailableInMockMode) {
    jetson_->init(config_path_.c_str());

    EXPECT_TRUE(jetson_->cuda_available());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}