import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import os
import time
from typing import Tuple
from pathlib import Path
from pycuda.compiler import SourceModule


def cpa_cuda_8_bit(traces: np.ndarray,
                   estimated_power_consumption: np.ndarray,
                   dtypes: Tuple[str, str, str] = ('double', 'double', 'double'),  # trace, PC, result
                   auto_c_contiguous: bool = True,
                   benchmark: bool = False
                   ) -> np.ndarray:
    """
    'cpa_cuda_8_bit' performs CPA(Correlation Power Analysis) using CUDA(pycuda) to find 1-byte key.

    :param traces: shape=(num_of_traces, num_of_samples)
    :param estimated_power_consumption: shape=(num_of_traces, 256)
    :param dtypes: data type of (trace, P.C, result) - (available : 'double' or 'float')
    :param auto_c_contiguous: If true, the c_contiguous property does not have to be considered. (Copying is performed.)
    :param benchmark: Benchmark the time and the memory. (output to stdout)
    :return: correlation coefficient : shape=(256, num_of_samples)
    """
    num_of_samples = traces.shape[1]
    return cpa_cuda_core(traces, estimated_power_consumption, 256, "calc_corr_8_bit",
                         (256, 1, 1), (num_of_samples, 1, 1), dtypes, auto_c_contiguous, benchmark)


def cpa_cuda_16_bit(traces: np.ndarray,
                    estimated_power_consumption: np.ndarray,
                    dtypes: Tuple[str, str, str] = ('double', 'double', 'double'),  # trace, PC, result
                    auto_c_contiguous: bool = True,
                    benchmark: bool = False
                    ) -> np.ndarray:
    """
    'cpa_cuda_16_bit' performs CPA(Correlation Power Analysis) using CUDA(pycuda) to find 2-byte keys.
    It can be used for higher-order attacks.

    :param traces: shape=(num_of_traces, num_of_samples)
    :param estimated_power_consumption: shape=(num_of_traces, 65536)
    :param dtypes: data type of (trace, P.C, result) - (available : 'double' or 'float')
    :param auto_c_contiguous: If true, the c_contiguous property does not have to be considered. (Copying is performed.)
    :param benchmark: Benchmark the time and the memory. (output to stdout)
    :return: correlation coefficient : shape=(65536, num_of_samples)
    """
    num_of_samples = traces.shape[1]
    return cpa_cuda_core(traces, estimated_power_consumption, 0x10000, "calc_corr_16_bit",
                         (256, 1, 1), (256, num_of_samples, 1), dtypes, auto_c_contiguous, benchmark)


def cpa_cuda_core(traces: np.ndarray,
                  estimated_power_consumption: np.ndarray,
                  key_space: int,
                  kernel_func_id: str,
                  thread_shape: tuple,
                  block_shape: tuple,
                  dtypes: Tuple[str, str, str] = ('double', 'double', 'double'),  # trace, PC, result
                  auto_c_contiguous: bool = True,
                  benchmark: bool = False
                  ) -> np.ndarray:
    np_dtypes = []
    for i in dtypes:
        assert i == 'double' or i == 'float'
        np_dtypes.append(np.float32 if i == 'float' else np.float64)

    if traces.flags['C_CONTIGUOUS'] is False:
        if auto_c_contiguous:
            traces = np.ascontiguousarray(traces, dtype=np_dtypes[0])
        else:
            raise RuntimeError("'traces' is not C_CONTIGUOUS. "
                               "Please use 'np.ascontiguousarray(...)' to make the array C_CONTIGUOUS.")
    if estimated_power_consumption.flags['C_CONTIGUOUS'] is False:
        if auto_c_contiguous:
            estimated_power_consumption = np.ascontiguousarray(estimated_power_consumption, dtype=np_dtypes[1])
        else:
            raise RuntimeError("'estimated_power_consumption' is not C_CONTIGUOUS. "
                               "Please use 'np.ascontiguousarray(...)' to make the array C_CONTIGUOUS.")

    if traces.dtype != np_dtypes[0]:
        traces = traces.astype(np_dtypes[0], order='C')
    if estimated_power_consumption.dtype != np_dtypes[1]:
        estimated_power_consumption = estimated_power_consumption.astype(np_dtypes[1], order='C')

    result_corr = np.empty(shape=(key_space, traces.shape[1]), order='C', dtype=np_dtypes[2])
    with open(str(Path(os.path.abspath(__file__)).parent.parent) + "/cuda_kernels/cpa_kernel.cu", 'r') as kernel_fp:
        kernel_code = ''.join(kernel_fp.readlines())
    kernel_code = kernel_code.replace("#define traceNum -1", f"#define traceNum {traces.shape[0]}")
    kernel_code = kernel_code.replace("typedef X T_trace;", f"typedef {dtypes[0]} T_trace;")
    kernel_code = kernel_code.replace("typedef X T_pc;", f"typedef {dtypes[1]} T_pc;")
    kernel_code = kernel_code.replace("typedef X T_result;", f"typedef {dtypes[2]} T_result;")
    kernel = SourceModule(kernel_code)
    calculate_cor = kernel.get_function(kernel_func_id)

    # Memory copy (Host -> Device)
    cp_1_start = time.time()
    cuda_traces = drv.mem_alloc(traces.nbytes)
    cuda_estimated_power_consumption = drv.mem_alloc(estimated_power_consumption.nbytes)
    cuda_result_corr = drv.mem_alloc(result_corr.nbytes)
    drv.memcpy_htod(cuda_traces, traces)
    drv.memcpy_htod(cuda_estimated_power_consumption, estimated_power_consumption)
    cp_1_end = time.time()

    # CUDA Calculation
    t = calculate_cor(cuda_result_corr, cuda_traces, cuda_estimated_power_consumption,
                      block=thread_shape, grid=block_shape, time_kernel=True)

    # Memory copy (Device -> Host)
    cp_2_start = time.time()
    drv.memcpy_dtoh(result_corr, cuda_result_corr)
    cp_2_end = time.time()

    if benchmark:
        d = 2**30
        print(f"\n============<<CUDA CPA>>============\n"
              f"[Allocation] Trace({traces.nbytes / d:.3f}GiB) + PC({estimated_power_consumption.nbytes/ d:.3f} GiB)"
              f" + Result({result_corr.nbytes / d:.3f} GiB)\n[Allocation] Total : "
              f"{(traces.nbytes + estimated_power_consumption.nbytes + result_corr.nbytes)/d:.3f} GiB\n"
              f"------------------------------------\n"
              f"[Time] Memory copy (H->D) : {cp_1_end - cp_1_start:.4f}s\n"
              f"[Time] CUDA calculation   : {t:.4f}s\n"
              f"[Time] Memory copy (D->H) : {cp_2_end - cp_2_start:.4f}s\n"
              f"------------------------------------\n"
              f"[Time]       Total        : {cp_1_end - cp_1_start + t + cp_2_end - cp_2_start:.4f}s\n"
              f"====================================\n")

    return np.nan_to_num(result_corr)
