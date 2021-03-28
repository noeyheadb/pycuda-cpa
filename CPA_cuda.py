import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import os
from typing import Optional
from pathlib import Path
from pycuda.compiler import SourceModule


def cpa_cuda_8_bit(traces: np.ndarray,
                   estimated_power_consumption: np.ndarray,
                   auto_c_contiguous: bool = True
                   ) -> np.ndarray:
    """
    'cpa_cuda_8_bit' performs CPA(Correlation Power Analysis) using CUDA(pycuda) to find 1-byte key.

    :param traces: shape=(num_of_traces, num_of_samples)
    :param estimated_power_consumption: shape=(num_of_traces, 256)
    :param auto_c_contiguous: If true, the c_contiguous property does not have to be considered. (Copying is performed.)
    :return: correlation coefficient : shape=(256, num_of_samples)
    """
    num_of_samples = traces.shape[1]
    return cpa_cuda_core(traces, estimated_power_consumption, 256, "calc_corr_8_bit",
                         (256, 1, 1), (num_of_samples, 1, 1), auto_c_contiguous)


def cpa_cuda_16_bit(traces: np.ndarray,
                    estimated_power_consumption: np.ndarray,
                    auto_c_contiguous: bool = True
                    ) -> np.ndarray:
    """
    'cpa_cuda_16_bit' performs CPA(Correlation Power Analysis) using CUDA(pycuda) to find 2-byte keys.
    It can be used for higher-order attacks.

    :param traces: shape=(num_of_traces, num_of_samples)
    :param estimated_power_consumption: shape=(num_of_traces, 65536)
    :param auto_c_contiguous: If true, the c_contiguous property does not have to be considered. (Copying is performed.)
    :return: correlation coefficient : shape=(65536, num_of_samples)
    """
    num_of_samples = traces.shape[1]
    return cpa_cuda_core(traces, estimated_power_consumption, 0x10000, "calc_corr_16_bit",
                         (256, 1, 1), (256, num_of_samples, 1), auto_c_contiguous)


def cpa_cuda_core(traces: np.ndarray,
                  estimated_power_consumption: np.ndarray,
                  key_space: int,
                  kernel_func_id: str,
                  thread_shape: tuple,
                  block_shape: tuple,
                  auto_c_contiguous: bool = True
                  ) -> np.ndarray:
    if traces.flags['C_CONTIGUOUS'] is False or estimated_power_consumption.flags['C_CONTIGUOUS'] is False:
        if auto_c_contiguous:
            # copying arrays
            traces = np.ascontiguousarray(traces, dtype=np.float64)
            estimated_power_consumption = np.ascontiguousarray(estimated_power_consumption, dtype=np.float64)
        else:
            raise RuntimeError("'traces' or 'estimated_power_consumption' are not C_CONTIGUOUS. "
                               "Please use 'np.ascontiguousarray(...)' to make the array C_CONTIGUOUS.")
    if traces.dtype != np.float64:
        traces = traces.astype(np.float64, order='C')
    if estimated_power_consumption.dtype != np.float64:
        estimated_power_consumption = estimated_power_consumption.astype(np.float64, order='C')

    result_corr = np.empty(shape=(key_space, traces.shape[1]), order='C', dtype=np.float64)
    with open(str(Path(os.path.abspath(__file__)).parent) + "/cuda_kernels/cpa_kernel.cu", 'r') as kernel_fp:
        kernel_code = ''.join(kernel_fp.readlines())
    kernel_code = kernel_code.replace("#define traceNum -1", f"#define traceNum {traces.shape[0]}")
    kernel = SourceModule(kernel_code)
    calculate_cor = kernel.get_function(kernel_func_id)
    calculate_cor(drv.Out(result_corr), drv.In(traces), drv.In(estimated_power_consumption),
                  block=thread_shape, grid=block_shape)
    return np.nan_to_num(result_corr)


def power_modeling_for_cpa_16_using_cuda(text_in: Optional[np.ndarray],
                                         text_out: Optional[np.ndarray],
                                         target_byte_1: int,
                                         target_byte_2: int,
                                         block_size: int,
                                         auto_c_contiguous: bool = True
                                         ) -> np.ndarray:
    """
    'power_modeling_for_cpa_16_using_cuda' is used to generate the estimated power consumption values in parallel.
     It is only for 'cpa_cuda_16_bit', not 'cpa_cuda_8_bit'.
     Before using this function, 'power_modeling_kernel.cu' must be modified according to the target algorithm.

    :param text_in: shape=(num_of_traces, block_size), dtype=np.uint8  e.g., shape=(2000, 16)
    :param text_out: shape=(num_of_traces, block_size), dtype=np.uint8  e.g., shape=(2000, 16)
    :param target_byte_1: Target byte 1. range: 0 ~ block_size-1
    :param target_byte_2: Target byte 1. range: 0 ~ block_size-1
    :param block_size: Block size of target algorithm. (unit: byte)  e.g., AES-128 -> 16
    :param auto_c_contiguous: If true, the c_contiguous property does not have to be considered. (Copying is performed.)
    :return: 
    """
    if text_in is None:
        text_in = np.array([0], dtype=np.uint8, order='C')
    elif text_in.dtype != np.uint8:
        text_in = text_in.astype(dtype=np.uint8, order='C')

    if text_out is None:
        text_out = np.array([0], dtype=np.uint8, order='C')
    elif text_in.dtype != np.uint8:
        text_out = text_out.astype(dtype=np.uint8, order='C')

    if text_in.flags['C_CONTIGUOUS'] is False or text_out.flags['C_CONTIGUOUS'] is False:
        if auto_c_contiguous:
            # copying arrays
            text_in = np.ascontiguousarray(text_in, dtype=np.uint8)
            text_out = np.ascontiguousarray(text_out, dtype=np.uint8)
        else:
            raise RuntimeError("'traces' or 'estimated_power_consumption' are not C_CONTIGUOUS. "
                               "Please use 'np.ascontiguousarray(...)' to make the array C_CONTIGUOUS.")

    num_of_traces = len(text_in)
    estimated_power = np.empty(shape=(num_of_traces, 0x10000), order='C', dtype=np.float64)

    with open(str(Path(os.path.abspath(__file__)).parent) + "/cuda_kernels/power_modeling_kernel.cu", 'r') as kernel_fp:
        kernel_code = ''.join(kernel_fp.readlines())
    kernel_code = kernel_code.replace("#define blockSize -1", f"#define blockSize {block_size}")
    kernel_code = kernel_code.replace("#define targetByte1 -1", f"#define targetByte1 {target_byte_1}")
    kernel_code = kernel_code.replace("#define targetByte2 -1", f"#define targetByte2 {target_byte_2}")
    kernel = SourceModule(kernel_code)
    power_modeling = kernel.get_function("power_consumption_modeling")
    power_modeling(drv.Out(estimated_power), drv.In(text_in), drv.In(text_out),
                   block=(256, 1, 1), grid=(256, num_of_traces, 1))
    return estimated_power
