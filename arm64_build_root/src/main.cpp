#include "tokenizers_cpp.h"
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <map>
#include <numeric>
#include <algorithm>

using namespace tokenizers;

// Helper to load tokenizer.json as string
std::string LoadStringFromFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int main() {
    // Load tokenizer
    std::string json_blob = LoadStringFromFile("../models/tokenizer.json");
    auto tokenizer = Tokenizer::FromBlobJSON(json_blob);

    // ORT environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "pipeline");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "../models/model.onnx", session_options);

    std::map<int, std::string> id2label = {
        {0, "lower_window"},
        {1, "raise_window"},
        {2, "set_temperature"},
        {3, "mute_media"},
        {4, "unmute_media"}
    };

    std::string text;
    while (true) {
        std::cout << "Enter a sentence (empty to quit): ";
        std::getline(std::cin, text);
        if (text.empty()) break;

        // Encode text → returns vector of token IDs
        text = "[CLS] " + text + " [SEP]";
        std::vector<int> input_ids = tokenizer->Encode(text);
        std::vector<int> attention_mask(input_ids.size(), 1);

        // Print
        std::cout << "Input IDs: ";
        for (int id : input_ids) std::cout << id << " ";
        std::cout << "\n";
        std::cout << "Attention Mask: ";
        for (int m : attention_mask) std::cout << m << " ";
        std::cout << "\n";

        // Prepare input for ONNX
        std::vector<int64_t> ids64(input_ids.begin(), input_ids.end());
        std::vector<int64_t> mask64(attention_mask.begin(), attention_mask.end());
        std::vector<int64_t> input_shape = {1, (int64_t)ids64.size()}; // [batch, seq_len]

        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor_ids = Ort::Value::CreateTensor<int64_t>(
            memory_info, ids64.data(), ids64.size(), input_shape.data(), input_shape.size());
        Ort::Value input_tensor_mask = Ort::Value::CreateTensor<int64_t>(
            memory_info, mask64.data(), mask64.size(), input_shape.data(), input_shape.size());

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
    }

    return 0;
}

