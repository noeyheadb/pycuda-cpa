// ##### Do not modify below definition #####
#define traceNum -1
typedef X T_trace;
typedef X T_pc;
typedef X T_result;
// ##########################################

__device__ double calc_corr(T_trace *trace, T_pc *est_pc, int threshold, int i, int j, int i_step, int j_step)
{
    double sumXY = 0, sumX = 0, sumY = 0, sumX2 = 0, sumY2 = 0, x, y;

    while(i < threshold)
    {
        x = trace[i];
        y = est_pc[j];
        sumXY += x * y;
        sumX += x;
        sumY += y;
        sumX2 += x * x;
        sumY2 += y * y;
        i += i_step;
        j += j_step;
    }

    return (traceNum * sumXY - sumX * sumY) /
        sqrt((traceNum * sumX2 - sumX * sumX) * (traceNum * sumY2 - sumY * sumY));
}

__global__ void calc_corr_8_bit(T_result *result, T_trace *trace, T_pc *est_pc)
{
    // threadIdx.x : Key : 0 ~ 256
    // blockIdx.x  : PoI : 0 ~ #PoI
    // gridDim.x   :     : #PoI

    result[gridDim.x * threadIdx.x + blockIdx.x] =
        calc_corr(trace, est_pc, gridDim.x * traceNum, blockIdx.x, threadIdx.x, gridDim.x, 256);
}

__global__ void calc_corr_16_bit(T_result *result, T_trace *trace, T_pc *est_pc)
{
    // threadIdx.x : Key_L(8-bit) : 0 ~ 256
    // blockIdx.x  : Key_H(8-bit) : 0 ~ 256
    // blockIdx.y  : PoI          : 0 ~ #PoI
    // gridDim.y   :              : #PoI

    result[gridDim.y * ((blockIdx.x << 8) + threadIdx.x) + blockIdx.y] =
        calc_corr(trace, est_pc, gridDim.y * traceNum, blockIdx.y, (blockIdx.x << 8) + threadIdx.x, gridDim.y, 0x10000);
}
