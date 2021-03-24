import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule


def cpa_cuda_256(traces: np.ndarray,
                 estimated_power_consumption: np.ndarray
                 ) -> np.ndarray:
    """
    CPA(Correlation Power Analysis) using CUDA(pycuda).

    :param traces: shape=(num_of_traces, num_of_samples)
    :param estimated_power_consumption: shape=(num_of_traces, 256)
    :return: correlation coefficient : shape=(256, num_of_samples)
    """

    if traces.flags['C_CONTIGUOUS'] is False or estimated_power_consumption.flags['C_CONTIGUOUS'] is False:
        raise RuntimeError("'traces' or 'estimated_power_consumption' are not contiguously allocated in memory."
                           "Please use 'np.ascontiguousarray(...)' to make the array contiguous.")

    result_corr = np.empty(shape=(256, traces.shape[1]))
    with open("./cpa_kernel.cu", 'r') as kernel_fp:
        kernel_code = ''.join(kernel_fp.readlines())
    kernel_code = kernel_code.replace("#define traceNum -1", f"#define traceNum {traces.shape[0]}")
    kernel = SourceModule(kernel_code)
    calculate_cor = kernel.get_function("calc_corr")
    num_of_samples = traces.shape[1]

    calculate_cor(drv.Out(result_corr), drv.In(traces), drv.In(estimated_power_consumption),
                  block=(256, 1, 1), grid=(num_of_samples, 1, 1))

    return np.nan_to_num(result_corr, 0)
