# DistilBERT NLU for Cockpit Intents

This project implements a Natural Language Understanding (NLU) module for a cockpit environment using a fine-tuned DistilBERT model. It includes the complete workflow from training in Python to high-performance inference using C++ and ONNX Runtime.

## Project Overview

The system classifies user commands into specific cockpit intents:
*   `lower_window`
*   `raise_window`
*   `set_temperature`
*   `mute_media`
*   `unmute_media`

The repository contains both the Python scripts for model development and a C++ implementation for efficient deployment.

## Directory Structure

*   **`onnx_model/`**: Contains the exported ONNX model (`model.onnx`).
*   **`onnx_runtime_c++/`**: C++ implementation using ONNX Runtime for inference.
    *   `my_tokenizer/`: Main C++ application source code.
    *   `tokenizers-cpp/`: Submodule/dependency for tokenization.
    *   `src/`: Additional C++ utility/inference files.
*   **`distilbert_cockpit_intents_2nd/`**: Saved artifacts for the fine-tuned Hugging Face model.
*   **Python Scripts**:
    *   `fine_tune.py`: Main script for training the model.
    *   `tokenize_data.py`: Pre-processing script to tokenize datasets.
    *   `inference_cockpit_intents.py`: Example inference using PyTorch/Transformers.
    *   `inference_onnx_intents.py`: Example inference using ONNX Runtime in Python.

## Python Workflow

### Prerequisites

Ensure you have the necessary Python libraries installed:

```bash
pip install transformers datasets evaluate torch numpy onnxruntime
```

### 1. Training

To fine-tune the DistilBERT model on your dataset:

```bash
python3 fine_tune.py
```

This script loads the data from CSV files (`training_data.csv`, etc.), tokenizes it, trains the model, and saves the results to `distilbert_cockpit_intents_2nd`.

### 2. Inference

**Using PyTorch:**

To test the trained model using standard Hugging Face components:

```bash
python3 inference_cockpit_intents.py
```

**Using ONNX (Python):**

To test the exported ONNX model via Python:

```bash
python3 inference_onnx_intents.py
```

## C++ ONNX Runtime Implementation

This section details how to build and run the C++ inference engine.

### Prerequisites

*   **CMake** (version 3.13 or higher)
*   **C++ Compiler** (supporting C++17)
*   **Dependencies**: The project expects `onnxruntime-linux-x64-1.20.0` and `tokenizers-cpp` to be present in the `onnx_runtime_c++` directory.

### Build Instructions

1.  Navigate to the project directory:

    ```bash
    cd onnx_runtime_c++/my_tokenizer
    ```

2.  Create a build directory and compile:

    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

### Usage

Run the compiled pipeline executable:

```bash
./pipeline
```

The program will launch an interactive CLI where you can type commands (e.g., "lower the temperature to 20") and see the predicted intent and confidence scores.

### Configuration

The build configuration is managed in `onnx_runtime_c++/my_tokenizer/CMakeLists.txt`. Ensure the paths to `onnxruntime` and `tokenizers-cpp` matches your directory layout if you move files around.
