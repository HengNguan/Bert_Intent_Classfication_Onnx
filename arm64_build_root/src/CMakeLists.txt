cmake_minimum_required(VERSION 3.13)
project(intent_classification)

# Path to your tokenizers-cpp root
add_subdirectory(../third_party/tokenizers-cpp tokenizers_build)

# Tell CMake where to find the headers
include_directories(../third_party/tokenizers-cpp/include)

# ONNX Runtime paths (update if needed)
include_directories(../third_party/onnxruntime-linux-aarch64-1.22.0/include)
link_directories(../third_party/onnxruntime-linux-aarch64-1.22.0/lib)

# Add pipeline executable (replace main.cpp with your pipeline source if needed)
add_executable(pipeline main.cpp)
target_link_libraries(pipeline PRIVATE tokenizers_cpp onnxruntime)
