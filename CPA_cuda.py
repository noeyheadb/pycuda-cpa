import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import os
from pathlib import Path
from pycuda.compiler import SourceModule


def cpa_cuda_256(traces: np.ndarray,
                 estimated_power_consumption: np.ndarray,
                 auto_c_contiguous: bool = True
                 ) -> np.ndarray:
    """
    CPA(Correlation Power Analysis) using CUDA(pycuda).

    :param traces: shape=(num_of_traces, num_of_samples)
    :param estimated_power_consumption: shape=(num_of_traces, 256)
    :param auto_c_contiguous: If true, the c_contiguous property does not have to be considered. (Copying is performed.)
    :return: correlation coefficient : shape=(256, num_of_samples)
    """

    if traces.flags['C_CONTIGUOUS'] is False or estimated_power_consumption.flags['C_CONTIGUOUS'] is False:
        if auto_c_contiguous:
            # copying arrays
            traces = np.ascontiguousarray(traces)
            estimated_power_consumption = np.ascontiguousarray(estimated_power_consumption)
        else:
            raise RuntimeError("'traces' or 'estimated_power_consumption' are not C_CONTIGUOUS. "
                               "Please use 'np.ascontiguousarray(...)' to make the array C_CONTIGUOUS.")

    result_corr = np.empty(shape=(256, traces.shape[1]), order='C')
    with open(str(Path(os.path.abspath(__file__)).parent) + "/cpa_kernel.cu", 'r') as kernel_fp:
        kernel_code = ''.join(kernel_fp.readlines())
    kernel_code = kernel_code.replace("#define traceNum -1", f"#define traceNum {traces.shape[0]}")
    kernel = SourceModule(kernel_code)
    calculate_cor = kernel.get_function("calc_corr")
    num_of_samples = traces.shape[1]

    calculate_cor(drv.Out(result_corr), drv.In(traces), drv.In(estimated_power_consumption),
                  block=(256, 1, 1), grid=(num_of_samples, 1, 1))

    return np.nan_to_num(result_corr)
