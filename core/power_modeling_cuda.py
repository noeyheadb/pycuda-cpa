import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import os
from typing import Optional
from pathlib import Path
from pycuda.compiler import SourceModule


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

    with open(str(Path(os.path.abspath(__file__)).parent.parent) + "/cuda_kernels/power_modeling_kernel.cu", 'r') \
            as kernel_fp:
        kernel_code = ''.join(kernel_fp.readlines())
    kernel_code = kernel_code.replace("#define blockSize -1", f"#define blockSize {block_size}")
    kernel_code = kernel_code.replace("#define targetByte1 -1", f"#define targetByte1 {target_byte_1}")
    kernel_code = kernel_code.replace("#define targetByte2 -1", f"#define targetByte2 {target_byte_2}")
    kernel = SourceModule(kernel_code)
    power_modeling = kernel.get_function("power_consumption_modeling")
    power_modeling(drv.Out(estimated_power), drv.In(text_in), drv.In(text_out),
                   block=(256, 1, 1), grid=(256, num_of_traces, 1))
    return estimated_power
