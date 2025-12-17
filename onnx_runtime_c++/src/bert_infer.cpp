#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <map>
#include <numeric>  // for std::distance
#include <algorithm> // for std::max_element

// Simple .npy loader (only works for int64 numpy arrays)
std::vector<int64_t> load_npy(const std::string& filename, std::vector<int64_t>& shape) {
    std::ifstream f(filename, std::ios::binary);
    assert(f.is_open());

    // Read magic string
    char magic[6];
    f.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") {
        throw std::runtime_error("Not a NPY file: " + filename);
    }

    char v[2];
    f.read(v, 2); // version
    uint16_t header_len;
    f.read(reinterpret_cast<char*>(&header_len), 2);

    std::string header(header_len, ' ');
    f.read(&header[0], header_len);

    // Extract shape from header (quick + dirty parsing)
    auto loc1 = header.find("(");
    auto loc2 = header.find(")");
    std::string shape_str = header.substr(loc1 + 1, loc2 - loc1 - 1);

    shape.clear();
    size_t pos = 0;
    while ((pos = shape_str.find(",")) != std::string::npos) {
        shape.push_back(std::stoll(shape_str.substr(0, pos)));
        shape_str.erase(0, pos + 1);
    }
    if (!shape_str.empty()) {
        shape.push_back(std::stoll(shape_str));
    }

    // Load data
    std::vector<int64_t> data(shape[0] * shape[1]);
    f.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(int64_t));

    return data;
}

int main() {
    // ORT environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "/home/ubuntu/nlu/bert-nlu/model_distilbert/nlu-script/onnx_model/model.onnx", session_options);

    std::map<int, std::string> id2label = {
        {0, "lower_window"},
        {1, "raise_window"},
        {2, "set_temperature"},
        {3, "mute_media"},
        {4, "unmute_media"}
    };

    Ort::AllocatorWithDefaultOptions allocator;

    // Load input_ids + attention_mask
    std::vector<int64_t> shape_ids, shape_mask;
    auto input_ids = load_npy("input_ids.npy", shape_ids);
    auto attention_mask = load_npy("attention_mask.npy", shape_mask);

    std::vector<int64_t> input_shape = {1, (int64_t)shape_ids[1]}; // [batch, seq_len]

    // Create tensors
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor_ids = Ort::Value::CreateTensor<int64_t>(
        memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size());
    Ort::Value input_tensor_mask = Ort::Value::CreateTensor<int64_t>(
        memory_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size());

    const char* input_names[] = {"input_ids", "attention_mask"};
    auto output_names = std::array<const char*, 1>{"logits"};

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor_ids));
    ort_inputs.push_back(std::move(input_tensor_mask));

    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      input_names, ort_inputs.data(), ort_inputs.size(),
                                      output_names.data(), output_names.size());

    // Get output
    float* logits = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t num_classes = output_shape[1];

    std::cout << "Logits: ";
    for (int i = 0; i < num_classes; i++) {
        std::cout << logits[i] << " ";
    }
    std::cout << std::endl;

    int predicted_id = std::distance(
        logits, std::max_element(logits, logits + num_classes));

    std::cout << "Predicted ID: " << predicted_id << std::endl;
    std::cout << "Predicted Label: " << id2label[predicted_id] << std::endl;

    return 0;
}
