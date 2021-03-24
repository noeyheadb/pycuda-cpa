// ##### Do not modify below definition #####
#define traceNum -1
// ##########################################

__global__ void calc_corr(double *result, double *trace, double *exp_pc)
{
    // threadIdx.x : 0 ~ 256
    // blockIdx.x  : 0 ~ #POI
    // gridDim.x   : #POI

    int threshold = gridDim.x * traceNum, i = blockIdx.x, j = threadIdx.x;
    double sumXY = 0, sumX = 0, sumY = 0, sumX2 = 0, sumY2 = 0, x, y;

    while( (i < threshold) && (j < threshold))
    {
        x = trace[i];
        y = exp_pc[j];
        sumXY += x * y;
        sumX += x;
        sumY += y;
        sumX2 += x * x;
        sumY2 += y * y;
        i += gridDim.x;
        j += 256;
    }

    x = (traceNum * sumXY - sumX * sumY);
    y = sqrt( (traceNum * sumX2 - sumX * sumX) * (traceNum * sumY2 - sumY * sumY) );
    result[gridDim.x * threadIdx.x + blockIdx.x] = x / y;
}
