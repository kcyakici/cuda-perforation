/**
 * lu.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#define POLYBENCH_TIME 1

#include "lu.cuh"
#include "../../../utilities/polybench.h"
#include "../../../utilities/polybenchUtilFuncts.h"
#include "../../../utilities/gputimer.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

// define perforation rates
#define LOOP_PERFORATION_RATE 0.95
#define KERNEL_LAUNCH_LOOP_RATE 1.0
#define GRID_PERFORATION_RATE 1.0
#define BLOCK_PERFORATION_RATE 1.0

#define GPU_DEVICE 0

#define RUN_ON_CPU

void lu(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
    for (int k = 0; k < _PB_N; k++)
    {
        for (int j = k + 1; j < _PB_N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }

        for (int i = k + 1; i < _PB_N; i++)
        {
            for (int j = k + 1; j < _PB_N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
        }
    }
}

void init_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n))
{
    int i, j;

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j + 1) / N;
        }
    }
}

void compareResults(int n, DATA_TYPE POLYBENCH_2D(A_cpu, N, N, n, n), DATA_TYPE POLYBENCH_2D(A_outputFromGpu, N, N, n, n))
{
    int i, j, fail, total;
    fail = 0;
    total = 0;

    // Compare a and b
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            total++;
            if (percentDiff(A_cpu[i][j], A_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
            {
                fail++;
            }
            else
            {
                printf("A: %lf, A Gpu: %lf\n", A_cpu[i][j], A_outputFromGpu[i][j]);
            }
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

__global__ void lu_kernel1(int n, DATA_TYPE *A, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((j > k) && (j < _PB_N))
    {
        A[k * N + j] = A[k * N + j] / A[k * N + k];
    }
}

__global__ void lu_kernel2(int n, DATA_TYPE *A, int k)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i > k) && (j > k) && (i < _PB_N) && (j < _PB_N))
    {
        A[i * N + j] = A[i * N + j] - A[i * N + k] * A[k * N + j];
    }
}

void luCuda(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_2D(A_outputFromGpu, N, N, n, n))
{
    DATA_TYPE *AGpu;

    cudaMalloc(&AGpu, N * N * sizeof(DATA_TYPE));
    cudaMemcpy(AGpu, A, N * N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);

    dim3 block1(ceil(DIM_THREAD_BLOCK_KERNEL_1_X * BLOCK_PERFORATION_RATE), ceil(DIM_THREAD_BLOCK_KERNEL_1_Y * BLOCK_PERFORATION_RATE));
    dim3 block2(ceil(DIM_THREAD_BLOCK_KERNEL_2_X * BLOCK_PERFORATION_RATE), ceil(DIM_THREAD_BLOCK_KERNEL_2_Y * BLOCK_PERFORATION_RATE));
    dim3 grid1(1, 1, 1);
    dim3 grid2(1, 1, 1);

    /* Start timer. */
    GpuTimer gpuTimer;
    gpuTimer.Start();

    for (int k = 0; k < N * KERNEL_LAUNCH_LOOP_RATE; k++)
    {
        grid1.x = (unsigned int)(ceil((float)(N - (k + 1)) / ((float)block1.x) * GRID_PERFORATION_RATE));
        lu_kernel1<<<grid1, block1>>>(n, AGpu, k);
        cudaDeviceSynchronize();

        grid2.x = (unsigned int)(ceil((float)(N - (k + 1)) / ((float)block2.x) * GRID_PERFORATION_RATE));
        grid2.y = (unsigned int)(ceil((float)(N - (k + 1)) / ((float)block2.y) * GRID_PERFORATION_RATE));
        lu_kernel2<<<grid2, block2>>>(n, AGpu, k);
        cudaDeviceSynchronize();
    }

    /* Stop and print timer. */
    gpuTimer.Stop();
    float elapsed_time = gpuTimer.Elapsed() / 1000;
    printf("GPU Time in seconds:\n");
    printf("%f\n", elapsed_time);

    cudaMemcpy(A_outputFromGpu, AGpu, N * N * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);
    cudaFree(AGpu);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n,
                        DATA_TYPE POLYBENCH_2D(A, N, N, n, n))

{
    int i, j;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
            if ((i * n + j) % 20 == 0)
                fprintf(stderr, "\n");
        }
    fprintf(stderr, "\n");
}

int main(int argc, char *argv[])
{
    int n = N;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_2D_ARRAY_DECL(A_outputFromGpu, DATA_TYPE, N, N, n, n);

    init_array(n, POLYBENCH_ARRAY(A));

    GPU_argv_init();
    luCuda(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_outputFromGpu));

#ifdef RUN_ON_CPU

    /* Start timer. */
    polybench_start_instruments;

    lu(n, POLYBENCH_ARRAY(A));

    /* Stop and print timer. */
    printf("CPU Time in seconds:\n");
    polybench_stop_instruments;
    polybench_print_instruments;

    compareResults(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_outputFromGpu));

#else // prevent dead code elimination

    polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A_outputFromGpu)));

#endif // RUN_ON_CPU

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(A_outputFromGpu);

    return 0;
}

#include "../../../utilities/polybench.c"