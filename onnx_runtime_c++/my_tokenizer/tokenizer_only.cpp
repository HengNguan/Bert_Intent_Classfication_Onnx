#include "tokenizers_cpp.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

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
    std::string json_blob = LoadStringFromFile("tokenizer.json");
    auto tokenizer = Tokenizer::FromBlobJSON(json_blob);

    std::string text;
    while (true) {
        std::cout << "Enter a sentence (empty to quit): ";
        std::getline(std::cin, text);
        if (text.empty()) break;

        // Encode text → returns vector of token IDs
        text = "[CLS] " + text + " [SEP]";
        std::vector<int> input_ids = tokenizer->Encode(text);

        // Generate attention mask (1 for tokens)
        std::vector<int> attention_mask(input_ids.size(), 1);

        // Print
        std::cout << "Input IDs: ";
        for (int id : input_ids) std::cout << id << " ";
        std::cout << "\n";

        std::cout << "Attention Mask: ";
        for (int m : attention_mask) std::cout << m << " ";
        std::cout << "\n";
    }

    return 0;
}

