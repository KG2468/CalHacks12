# CalHacks12

## Reproducing Results

In order to get the proper results, here are the steps to follow:

- **Reproducing quantization then pruning results**, run the programs in this order:
  - `python3 test_hardware_optimizater.py`
  - `python3 test_pruning.py`

- **Reproducing pruning then quantization results**, run programs in the following order:
  - `python3 test_hardware_optimizater.py`
  - `python3 test_pruning.py`

- For both Python commands, user has the option to include model of their choice to test pruning and quantization
  
- **Reproducing recursive inference logic**, run programs:
  - `recursive_inference.py`
    - This code using Qwen3-8B as its main model and our custom 70M parameter model as the smaller model
