/**
 * bicg.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda.h>

#define POLYBENCH_TIME 1

#include "bicg.cuh"
#include "../../../utilities/polybench.h"
#include "../../../utilities/polybenchUtilFuncts.h"
#include "../../../utilities/gputimer.h"

// Error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

// define perforation rates
#define LOOP_PERFORATION_RATE 1.0
#define KERNEL_LAUNCH_LOOP_RATE 0.85
#define BLOCK_PERFORATION_RATE 1.0
#define GRID_PERFORATION_RATE 1.0

#define GPU_DEVICE 0

#ifndef M_PI
#define M_PI 3.14159
#endif

#define RUN_ON_CPU

void init_array(int nx, int ny, DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny), DATA_TYPE POLYBENCH_1D(p, NY, ny), DATA_TYPE POLYBENCH_1D(r, NX, nx))
{
    int i, j;

    for (i = 0; i < ny; i++)
    {
        p[i] = i * M_PI;
    }

    for (i = 0; i < nx; i++)
    {
        r[i] = i * M_PI;

        for (j = 0; j < ny; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j) / NX;
        }
    }
}

void compareResults(int nx, int ny, DATA_TYPE POLYBENCH_1D(s, NY, ny), DATA_TYPE POLYBENCH_1D(s_outputFromGpu, NY, ny),
                    DATA_TYPE POLYBENCH_1D(q, NX, nx), DATA_TYPE POLYBENCH_1D(q_outputFromGpu, NX, nx))
{
    int i, fail, total;
    fail = 0;
    total = 0;
    // Compare s with s_cuda
    for (i = 0; i < nx; i++)
    {
        total++;
        if (percentDiff(q[i], q_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
        {
            fail++;
        }
        else
        {
            printf("q: %f , q gpu: %f\n", q[i], q_outputFromGpu[i]);
        }
    }

    for (i = 0; i < ny; i++)
    {
        total++;
        if (percentDiff(s[i], s_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
        {
            fail++;
        }
        else
        {
            printf("s: %f , s gpu: %f\n", s[i], s_outputFromGpu[i]);
        }
    }

    // print results
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

// Distributed (split) from initial loop and permuted into reverse order to allow parallelism...
__global__ void bicg_kernel1(int nx, int ny, DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s, int perforated_pb_nx)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < _PB_NY)
    {
        s[j] = 0.0f;

        int i;
        for (i = 0; i < perforated_pb_nx; i++)
        {
            s[j] += r[i] * A[i * NY + j];
        }
    }
}

// Distributed (split) from initial loop to allow parallelism
__global__ void bicg_kernel2(int nx, int ny, DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *q, int perforated_pb_ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < _PB_NX)
    {
        q[i] = 0.0f;

        int j;
        for (j = 0; j < perforated_pb_ny; j++)
        {
            q[i] += A[i * NY + j] * p[j];
        }
    }
}

void bicg_cpu(int nx, int ny, DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny), DATA_TYPE POLYBENCH_1D(r, NX, nx), DATA_TYPE POLYBENCH_1D(s, NY, ny),
              DATA_TYPE POLYBENCH_1D(p, NY, ny), DATA_TYPE POLYBENCH_1D(q, NX, nx))
{
    int i, j;

    for (i = 0; i < _PB_NY; i++)
    {
        s[i] = 0.0;
    }

    for (i = 0; i < _PB_NX; i++)
    {
        q[i] = 0.0;
        for (j = 0; j < _PB_NY; j++)
        {
            s[j] = s[j] + r[i] * A[i][j];
            q[i] = q[i] + A[i][j] * p[j];
        }
    }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int nx, int ny,
                        DATA_TYPE POLYBENCH_1D(s, NY, ny),
                        DATA_TYPE POLYBENCH_1D(q, NX, nx))

{
    int i;

    for (i = 0; i < ny; i++)
    {
        fprintf(stderr, DATA_PRINTF_MODIFIER, s[i]);
        if (i % 20 == 0)
            fprintf(stderr, "\n");
    }
    for (i = 0; i < nx; i++)
    {
        fprintf(stderr, DATA_PRINTF_MODIFIER, q[i]);
        if (i % 20 == 0)
            fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}

void bicgCuda(int nx, int ny, DATA_TYPE POLYBENCH_2D(A, NX, NY, nx, ny), DATA_TYPE POLYBENCH_1D(r, NX, nx), DATA_TYPE POLYBENCH_1D(s, NY, ny),
              DATA_TYPE POLYBENCH_1D(p, NY, ny), DATA_TYPE POLYBENCH_1D(q, NX, nx), DATA_TYPE POLYBENCH_1D(s_outputFromGpu, NY, ny),
              DATA_TYPE POLYBENCH_1D(q_outputFromGpu, NX, nx))
{
    DATA_TYPE *A_gpu;
    DATA_TYPE *q_gpu;
    DATA_TYPE *p_gpu;
    DATA_TYPE *r_gpu;
    DATA_TYPE *s_gpu;

    float perforated_pb_ny = _PB_NY * LOOP_PERFORATION_RATE;
    float perforated_pb_nx = _PB_NX * LOOP_PERFORATION_RATE;

    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NX * NY);
    cudaMalloc((void **)&r_gpu, sizeof(DATA_TYPE) * NX);
    cudaMalloc((void **)&s_gpu, sizeof(DATA_TYPE) * NY);
    cudaMalloc((void **)&p_gpu, sizeof(DATA_TYPE) * NY);
    cudaMalloc((void **)&q_gpu, sizeof(DATA_TYPE) * NX);
    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice);
    cudaMemcpy(r_gpu, r, sizeof(DATA_TYPE) * NX, cudaMemcpyHostToDevice);
    cudaMemcpy(s_gpu, s, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
    cudaMemcpy(p_gpu, p, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice);
    cudaMemcpy(q_gpu, q, sizeof(DATA_TYPE) * NX, cudaMemcpyHostToDevice);

    dim3 block(ceil(DIM_THREAD_BLOCK_X * BLOCK_PERFORATION_RATE), ceil(DIM_THREAD_BLOCK_Y * BLOCK_PERFORATION_RATE));
    dim3 grid1((size_t)(ceil(((float)NY) / ((float)block.x) * GRID_PERFORATION_RATE)), 1);
    dim3 grid2((size_t)(ceil(((float)NX) / ((float)block.x) * GRID_PERFORATION_RATE)), 1);

    /* Start timer. */
    GpuTimer gpuTimer;
    gpuTimer.Start();

    bicg_kernel1<<<grid1, block>>>(nx, ny, A_gpu, r_gpu, s_gpu, perforated_pb_nx);
    cudaDeviceSynchronize();
    bicg_kernel2<<<grid2, block>>>(nx, ny, A_gpu, p_gpu, q_gpu, perforated_pb_ny);
    cudaDeviceSynchronize();

    /* Stop and print timer. */
    gpuTimer.Stop();
    float elapsed_time = gpuTimer.Elapsed() / 1000;
    printf("GPU Time in seconds:\n");
    printf("%f\n", elapsed_time);

    cudaMemcpy(s_outputFromGpu, s_gpu, sizeof(DATA_TYPE) * NY, cudaMemcpyDeviceToHost);
    cudaMemcpy(q_outputFromGpu, q_gpu, sizeof(DATA_TYPE) * NX, cudaMemcpyDeviceToHost);

    cudaFree(A_gpu);
    cudaFree(r_gpu);
    cudaFree(s_gpu);
    cudaFree(p_gpu);
    cudaFree(q_gpu);
}

int main(int argc, char **argv)
{
    int nx = NX;
    int ny = NY;

    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NX, NY, nx, ny);
    POLYBENCH_1D_ARRAY_DECL(s, DATA_TYPE, NY, ny);
    POLYBENCH_1D_ARRAY_DECL(q, DATA_TYPE, NX, nx);
    POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, NY, ny);
    POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, NX, nx);
    POLYBENCH_1D_ARRAY_DECL(s_outputFromGpu, DATA_TYPE, NY, ny);
    POLYBENCH_1D_ARRAY_DECL(q_outputFromGpu, DATA_TYPE, NX, nx);

    init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(r));

    GPU_argv_init();

    bicgCuda(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q),
             POLYBENCH_ARRAY(s_outputFromGpu), POLYBENCH_ARRAY(q_outputFromGpu));

#ifdef RUN_ON_CPU

    /* Start timer. */
    polybench_start_instruments;

    bicg_cpu(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));

    /* Stop and print timer. */
    printf("CPU Time in seconds:\n");
    polybench_stop_instruments;
    polybench_print_instruments;

    compareResults(nx, ny, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(s_outputFromGpu), POLYBENCH_ARRAY(q),
                   POLYBENCH_ARRAY(q_outputFromGpu));

#else // prevent dead code elimination

    polybench_prevent_dce(print_array(nx, ny, POLYBENCH_ARRAY(s_outputFromGpu), POLYBENCH_ARRAY(q_outputFromGpu)));

#endif // RUN_ON_CPU

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(r);
    POLYBENCH_FREE_ARRAY(s);
    POLYBENCH_FREE_ARRAY(p);
    POLYBENCH_FREE_ARRAY(q);
    POLYBENCH_FREE_ARRAY(s_outputFromGpu);
    POLYBENCH_FREE_ARRAY(q_outputFromGpu);

    return 0;
}

#include "../../../utilities/polybench.c"