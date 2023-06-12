/**
 * 3DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "3DConvolution.cuh"
#include "../../utilities/polybench.h"
#include "../../utilities/polybenchUtilFuncts.h"
#include "../../utilities/gputimer.h"

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

// define perforation rates
#define LOOP_PERFORATION_RATE 1.0
#define KERNEL_LAUNCH_LOOP_RATE 0.85
#define GRID_PERFORATION_RATE 1.0
#define BLOCK_PERFORATION_RATE 1.0

#define GPU_DEVICE 0

#define RUN_ON_CPU

void conv3D(int ni, int nj, int nk, DATA_TYPE POLYBENCH_3D(A, NI, NJ, NK, ni, nj, nk), DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk))
{
    int i, j, k;
    DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

    c11 = +2;
    c21 = +5;
    c31 = -8;
    c12 = -3;
    c22 = +6;
    c32 = -9;
    c13 = +4;
    c23 = +7;
    c33 = +10;

    for (i = 1; i < _PB_NI - 1; ++i) // 0
    {
        for (j = 1; j < _PB_NJ - 1; ++j) // 1
        {
            for (k = 1; k < _PB_NK - 1; ++k) // 2
            {
                B[i][j][k] = c11 * A[(i - 1)][(j - 1)][(k - 1)] + c13 * A[(i + 1)][(j - 1)][(k - 1)] + c21 * A[(i - 1)][(j - 1)][(k - 1)] + c23 * A[(i + 1)][(j - 1)][(k - 1)] + c31 * A[(i - 1)][(j - 1)][(k - 1)] + c33 * A[(i + 1)][(j - 1)][(k - 1)] + c12 * A[(i + 0)][(j - 1)][(k + 0)] + c22 * A[(i + 0)][(j + 0)][(k + 0)] + c32 * A[(i + 0)][(j + 1)][(k + 0)] + c11 * A[(i - 1)][(j - 1)][(k + 1)] + c13 * A[(i + 1)][(j - 1)][(k + 1)] + c21 * A[(i - 1)][(j + 0)][(k + 1)] + c23 * A[(i + 1)][(j + 0)][(k + 1)] + c31 * A[(i - 1)][(j + 1)][(k + 1)] + c33 * A[(i + 1)][(j + 1)][(k + 1)];
            }
        }
    }
}

void init(int ni, int nj, int nk, DATA_TYPE POLYBENCH_3D(A, NI, NJ, NK, ni, nj, nk))
{
    int i, j, k;

    for (i = 0; i < ni; ++i)
    {
        for (j = 0; j < nj; ++j)
        {
            for (k = 0; k < nk; ++k)
            {
                A[i][j][k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
            }
        }
    }
}

void compareResults(int ni, int nj, int nk, DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk), DATA_TYPE POLYBENCH_3D(B_outputFromGpu, NI, NJ, NK, ni, nj, nk))
{
    int i, j, k, fail, total;
    fail = 0;
    total = 0;

    // Compare result from cpu and gpu
    for (i = 1; i < ni - 1; ++i) // 0
    {
        for (j = 1; j < nj - 1; ++j) // 1
        {
            for (k = 1; k < nk - 1; ++k) // 2
            {
                total++;
                if (percentDiff(B[i][j][k], B_outputFromGpu[i][j][k]) > PERCENT_DIFF_ERROR_THRESHOLD)
                {
                    fail++;
                }
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

__global__ void convolution3D_kernel(int ni, int nj, int nk, DATA_TYPE *A, DATA_TYPE *B, int i)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

    c11 = +2;
    c21 = +5;
    c31 = -8;
    c12 = -3;
    c22 = +6;
    c32 = -9;
    c13 = +4;
    c23 = +7;
    c33 = +10;

    if ((i < (_PB_NI - 1)) && (j < (_PB_NJ - 1)) && (k < (_PB_NK - 1)) && (i > 0) && (j > 0) && (k > 0))
    {
        B[i * (NK * NJ) + j * NK + k] = c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c21 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c23 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c31 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c33 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k - 1)] + c12 * A[(i + 0) * (NK * NJ) + (j - 1) * NK + (k + 0)] + c22 * A[(i + 0) * (NK * NJ) + (j + 0) * NK + (k + 0)] + c32 * A[(i + 0) * (NK * NJ) + (j + 1) * NK + (k + 0)] + c11 * A[(i - 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] + c13 * A[(i + 1) * (NK * NJ) + (j - 1) * NK + (k + 1)] + c21 * A[(i - 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] + c23 * A[(i + 1) * (NK * NJ) + (j + 0) * NK + (k + 1)] + c31 * A[(i - 1) * (NK * NJ) + (j + 1) * NK + (k + 1)] + c33 * A[(i + 1) * (NK * NJ) + (j + 1) * NK + (k + 1)];
    }
}

void convolution3DCuda(int ni, int nj, int nk, DATA_TYPE POLYBENCH_3D(A, NI, NJ, NK, ni, nj, nk), DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk), DATA_TYPE POLYBENCH_3D(B_outputFromGpu, NI, NJ, NK, ni, nj, nk))
{
    DATA_TYPE *A_gpu;
    DATA_TYPE *B_gpu;

    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NJ * NK);
    cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NI * NJ * NK);
    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ * NK, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NI * NJ * NK, cudaMemcpyHostToDevice);

    dim3 block(ceil(DIM_THREAD_BLOCK_X * BLOCK_PERFORATION_RATE), ceil(DIM_THREAD_BLOCK_Y * BLOCK_PERFORATION_RATE));
    dim3 grid((size_t)(ceil(((float)NK) / ((float)block.x) * GRID_PERFORATION_RATE)), (size_t)(ceil(((float)NJ) / ((float)block.y) * GRID_PERFORATION_RATE)));

    /* Start timer. */
    GpuTimer gpuTimer;
    gpuTimer.Start();

    int i;
    for (i = 1; i < (_PB_NI - 1) * KERNEL_LAUNCH_LOOP_RATE; ++i) // 0
    {
        convolution3D_kernel<<<grid, block>>>(ni, nj, nk, A_gpu, B_gpu, i);
    }
    cudaDeviceSynchronize();
    /* Stop and print timer. */
    gpuTimer.Stop();
    float elapsed_time = gpuTimer.Elapsed() / 1000;
    printf("GPU Time in seconds:\n");
    printf("%f\n", elapsed_time);

    cudaMemcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * NI * NJ * NK, cudaMemcpyDeviceToHost);

    cudaFree(A_gpu);
    cudaFree(B_gpu);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nj, int nk,
                        DATA_TYPE POLYBENCH_3D(B, NI, NJ, NK, ni, nj, nk))
{
    int i, j, k;

    for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++)
            for (k = 0; k < nk; k++)
            {
                fprintf(stderr, DATA_PRINTF_MODIFIER, B[i][j][k]);
                if ((i * (nj * nk) + j * nk + k) % 20 == 0)
                    fprintf(stderr, "\n");
            }
    fprintf(stderr, "\n");
}

int main(int argc, char *argv[])
{
    int ni = NI;
    int nj = NJ;
    int nk = NK;

    POLYBENCH_3D_ARRAY_DECL(A, DATA_TYPE, NI, NJ, NK, ni, nj, nk);
    POLYBENCH_3D_ARRAY_DECL(B, DATA_TYPE, NI, NJ, NK, ni, nj, nk);
    POLYBENCH_3D_ARRAY_DECL(B_outputFromGpu, DATA_TYPE, NI, NJ, NK, ni, nj, nk);

    init(ni, nj, nk, POLYBENCH_ARRAY(A));

    GPU_argv_init();

    convolution3DCuda(ni, nj, nk, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu));

#ifdef RUN_ON_CPU

    /* Start timer. */
    polybench_start_instruments;

    conv3D(ni, nj, nk, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

    printf("CPU Time in seconds:\n");
    polybench_stop_instruments;
    polybench_print_instruments;

    compareResults(ni, nj, nk, POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu));

#else // prevent dead code elimination

    polybench_prevent_dce(print_array(NI, NJ, NK, POLYBENCH_ARRAY(B_outputFromGpu)));

#endif // RUN_ON_CPU

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(B_outputFromGpu);

    return 0;
}

#include "../../utilities/polybench.c"