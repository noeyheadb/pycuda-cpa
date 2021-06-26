# pycuda-CPA
`pycuda-CPA` is a CUDA implementation of CPA(Correlation Power Analysis) using [`pycuda`](https://github.com/inducer/pycuda). 
It is only works on CUDA-enabled systems.

## Environment setup
- GPUs with compute capability less than 2.x are not supported.
- Make `pycuda` available by installing the CUDA toolkit and setting environment variables, etc.
- Test code of `pycuda` is available [here](https://documen.tician.de/pycuda/).

## Requirements
- pycuda >= 2020.1
- numpy

## Usage
Demo script is available [here](https://github.com/noeyheadb/pycuda-CPA/blob/master/demo_first_order.py).

## Benchmark

#### Environment
- Intel(R) Core(TM) i5-10500 CPU (6 Cores, 12 Threads)
- DDR4 32GB 2666MHz RAM
- NVIDIA RTX2060 Super
  - GDDR6 8GB Memory
  - 2,176 CUDA cores (Streaming Processor)
  - 7.5 Compute capability
  - ([more](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability))

#### Experimental Setup
| Target algorithm | Detail                        | # of samples (per trace) |
| :--------------: | :---------------------------: | :---------------:        |
| Naïve AES        | 1st byte of 1-round SubBytes  | 24,400                   |

#### Results
| Language | Implementation detail   | # of traces | Time      |
| :------: | :---------------------: | :---------: | :-------: |
| Python   | **`pycuda-CPA`**        | 1,000       | **0.41s** |
| Python   | multiprocessing (12p.)  | 1,000       | 106s      |
| Python   | **`pycuda-CPA`**        | 5,000       | **1.54s** |
| Python   | multiprocessing (12p.)  | 5,000       | 418s      |
| Python   | **`pycuda-CPA`**        | 10,000      | **3.03s** |
| Python   | multiprocessing (12p.)  | 10,000      | 864s      |
