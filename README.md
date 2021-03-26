# pycuda-CPA
`pycuda-CPA` is a CUDA implementation of CPA(Correlation Power Analysis) using [`pycuda`](https://github.com/inducer/pycuda). 
It is only works on CUDA-enabled systems.

## Environment setup
- GPUs with compute capability less than 1.3 are not supported.
- Make `pycuda` available by installing the CUDA toolkit and setting environment variables, etc.
- Test code of `pycuda` is [here](https://documen.tician.de/pycuda/).

## Requirements
- pycuda >= 2020.1
- numpy

## Usage
Demo script is available [here](https://github.com/noeyheadb/pycuda-CPA/blob/master/demo.py).
