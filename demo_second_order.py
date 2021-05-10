import numpy as np
from typing import Tuple
from core import cpa_cuda_16_bit, power_modeling_for_cpa_16_using_cuda


def combine_trace_for_so_attack_2_point(trace: np.ndarray,
                                        poi_1: Tuple[int, int],
                                        poi_2: Tuple[int, int]
                                        ) -> np.ndarray:
    def combine_function(t1: np.ndarray, t2: np.ndarray):
        return abs(t1 - t2)
    combined_trace_list = []
    for p1 in range(poi_1[0], poi_1[1]):
        target_trace_1 = trace[:, [p1]]
        target_trace_2 = trace[:, poi_2[0]:poi_2[1]]
        combined_trace_segment = combine_function(target_trace_1, target_trace_2)
        combined_trace_list.append(combined_trace_segment)
    combined_trace = np.hstack(combined_trace_list)
    return combined_trace


raw_trace = np.load(f"trace.npy")
# trace : shape=(num_of_traces, num_of_samples)
plain_text = np.load(f"plain.npy")
# plain_text : shape=(num_of_traces,)  dtype=np.str_
# e.g. ['00112233445566778899AABBCCDDEEFF', 'FFEEDDCCBBAA99887766554433221100', ...]

target_trace = combine_trace_for_so_attack_2_point(raw_trace, (100, 120), (300, 320))
target_plain = np.array([bytearray.fromhex(p) for p in plain_text], dtype=np.uint8, order='C')

estimated_power = power_modeling_for_cpa_16_using_cuda(target_plain, None, 0, 4, 16, True, benchmark=True)
print("* Power modeling finished.")

cor_mat = cpa_cuda_16_bit(target_trace, estimated_power, benchmark=True, dtypes=('double', 'double', 'float'))
print(f"* Guessed key is 0x{np.argmax(np.max(abs(cor_mat), axis=1)):04X}. "
      f"({np.max(np.max(abs(cor_mat), axis=1)):.04f})")
