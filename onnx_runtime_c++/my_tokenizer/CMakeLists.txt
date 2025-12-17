cmake_minimum_required(VERSION 3.13)
project(my_tokenizer_test)
set(CMAKE_CXX_STANDARD 17)

# Path to your tokenizers-cpp root
add_subdirectory(../tokenizers-cpp tokenizers_build)

# Tell CMake where to find the headers
include_directories(../tokenizers-cpp/include)

# ONNX Runtime paths (update if needed)
include_directories(../onnxruntime-linux-x64-1.20.0/include)
link_directories(../onnxruntime-linux-x64-1.20.0/lib)

#add_executable(my_tokenizer main.cpp)
#target_link_libraries(my_tokenizer PRIVATE tokenizers_cpp)

# Add pipeline executable (replace main.cpp with your pipeline source if needed)
add_executable(pipeline main.cpp)
target_link_libraries(pipeline PRIVATE tokenizers_cpp onnxruntime)
