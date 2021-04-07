import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import os
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
    with open(str(Path(os.path.abspath(__file__)).parent.parent) + "/cuda_kernels/cpa_kernel.cu", 'r') as kernel_fp:
        kernel_code = ''.join(kernel_fp.readlines())
    kernel_code = kernel_code.replace("#define traceNum -1", f"#define traceNum {traces.shape[0]}")
    kernel = SourceModule(kernel_code)
    calculate_cor = kernel.get_function(kernel_func_id)
    calculate_cor(drv.Out(result_corr), drv.In(traces), drv.In(estimated_power_consumption),
                  block=thread_shape, grid=block_shape)
    return np.nan_to_num(result_corr)
