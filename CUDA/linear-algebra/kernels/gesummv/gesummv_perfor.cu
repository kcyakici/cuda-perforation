/**
 * gesummv.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#define POLYBENCH_TIME 1

#include "gesummv.cuh"
#include "../../../utilities/polybench.h"
#include "../../../utilities/polybenchUtilFuncts.h"
#include "../../../utilities/gputimer.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

// define perforation rates
#define LOOP_PERFORATION_RATE 1.0
#define BLOCK_PERFORATION_RATE 0.95

#define GPU_DEVICE 0

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 43532.0f
#define BETA 12313.0f

#define RUN_ON_CPU

void gesummv(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_2D(B, N, N, n, n), DATA_TYPE POLYBENCH_1D(tmp, N, n),
             DATA_TYPE POLYBENCH_1D(x, N, n), DATA_TYPE POLYBENCH_1D(y, N, n))
{
    int i, j;

    for (i = 0; i < _PB_N; i++)
    {
        tmp[i] = 0;
        y[i] = 0;
        for (j = 0; j < _PB_N; j++)
        {
            tmp[i] = A[i][j] * x[j] + tmp[i];
            y[i] = B[i][j] * x[j] + y[i];
        }

        y[i] = alpha * tmp[i] + beta * y[i];
    }
}

void init(int n, DATA_TYPE *alpha, DATA_TYPE *beta, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_2D(B, N, N, n, n),
          DATA_TYPE POLYBENCH_1D(x, N, n))
{
    int i, j;

    *alpha = 43532;
    *beta = 12313;

    for (i = 0; i < n; i++)
    {
        x[i] = ((DATA_TYPE)i) / N;

        for (j = 0; j < n; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j) / N;
            B[i][j] = ((DATA_TYPE)i * j) / n;
        }
    }
}

void compareResults(int n, DATA_TYPE POLYBENCH_1D(y, N, n), DATA_TYPE POLYBENCH_1D(y_outputFromGpu, N, n))
{
    int i, fail, total;
    fail = 0;
    total = 0;

    for (i = 0; i < n; i++)
    {
        total++;
        if (percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
        {
            fail++;
        }
    }

    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

    printf("Total number of comparations: %d\n", total);
    printf("Loop perforation rate: %f\n", LOOP_PERFORATION_RATE);
    printf("Block perforation rate: %f\n", BLOCK_PERFORATION_RATE);
}

void GPU_argv_init()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
    cudaSetDevice(GPU_DEVICE);
}

__global__ void gesummv_kernel(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *tmp, DATA_TYPE *x, DATA_TYPE *y, int perforated_pb_n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < _PB_N)
    {
        int j;
        for (j = 0; j < perforated_pb_n; j++)
        {
            tmp[i] += A[i * N + j] * x[j];
            y[i] += B[i * N + j] * x[j];
        }
        y[i] = alpha * tmp[i] + beta * y[i];
    }
}

void gesummvCuda(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_2D(B, N, N, n, n),
                 DATA_TYPE POLYBENCH_1D(tmp, N, n), DATA_TYPE POLYBENCH_1D(x, N, n), DATA_TYPE POLYBENCH_1D(y, N, n),
                 DATA_TYPE POLYBENCH_1D(y_outputFromGpu, N, n))
{
    DATA_TYPE *A_gpu;
    DATA_TYPE *B_gpu;
    DATA_TYPE *x_gpu;
    DATA_TYPE *y_gpu;
    DATA_TYPE *tmp_gpu;

    float perforated_pb_n = LOOP_PERFORATION_RATE * _PB_N;

    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * N * N);
    cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * N * N);
    cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * N);
    cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * N);
    cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * N);

    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);

    dim3 block(ceil(DIM_THREAD_BLOCK_X * BLOCK_PERFORATION_RATE), ceil(DIM_THREAD_BLOCK_Y * BLOCK_PERFORATION_RATE));
    dim3 grid((unsigned int)ceil(((float)N) / ((float)block.x) * BLOCK_PERFORATION_RATE), 1);

    /* Start timer. */
    GpuTimer gpuTimer;
    gpuTimer.Start();

    gesummv_kernel<<<grid, block>>>(n, alpha, beta, A_gpu, B_gpu, tmp_gpu, x_gpu, y_gpu, perforated_pb_n);
    cudaDeviceSynchronize();

    /* Stop and print timer. */
    gpuTimer.Stop();
    float elapsed_time = gpuTimer.Elapsed() / 1000;
    printf("GPU Time in seconds:\n");
    printf("%f\n", elapsed_time);

    cudaMemcpy(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n,
                        DATA_TYPE POLYBENCH_1D(y, N, n))

{
    int i;

    for (i = 0; i < n; i++)
    {
        fprintf(stderr, DATA_PRINTF_MODIFIER, y[i]);
        if (i % 20 == 0)
            fprintf(stderr, "\n");
    }
}

int main(int argc, char *argv[])
{
    /* Retrieve problem size. */
    int n = N;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y_outputFromGpu, DATA_TYPE, N, n);

    init(n, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(x));

    GPU_argv_init();
    gesummvCuda(n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y),
                POLYBENCH_ARRAY(y_outputFromGpu));

#ifdef RUN_ON_CPU

    /* Start timer. */
    polybench_start_instruments;

    gesummv(n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y));

    /* Stop and print timer. */
    printf("CPU Time in seconds:\n");
    polybench_stop_instruments;
    polybench_print_instruments;

    compareResults(n, POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(y_outputFromGpu));

#else // prevent dead code elimination

    polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y_outputFromGpu)));

#endif // RUN_ON_CPU

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(tmp);
    POLYBENCH_FREE_ARRAY(x);
    POLYBENCH_FREE_ARRAY(y);
    POLYBENCH_FREE_ARRAY(y_outputFromGpu);

    return 0;
}

#include "../../../utilities/polybench.c"