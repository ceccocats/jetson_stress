#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <thread>
#include <fstream>
#include <cuda_runtime.h>

#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

bool gRun = true;


// Create a kernel to estimate pi
__global__ void count_samples_in_circles(curandState *state, uint64_t *d_countInBlocks, uint64_t num_blocks, uint64_t nsamples)
{

    __shared__ uint64_t shared_blocks[500];

    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = blockDim.x * num_blocks;

    // Iterates through
    uint64_t inCircle = 0;
    for (uint64_t i = index; i < nsamples; i += stride)
    {
        float xValue = curand_uniform_double(&state[0]);
        float yValue = curand_uniform_double(&state[0]);

        if (xValue * xValue + yValue * yValue <= 1.0f)
        {
            inCircle++;
        }
    }

    shared_blocks[threadIdx.x] = inCircle;

    __syncthreads();

    // Pick thread 0 for each block to collect all points from each Thread.
    if (threadIdx.x == 0)
    {
        uint64_t totalInCircleForABlock = 0;
        for (uint64_t j = 0; j < blockDim.x; j++)
        {
            totalInCircleForABlock += shared_blocks[j];
        }
        d_countInBlocks[blockIdx.x] = totalInCircleForABlock;
    }
}

__global__ void randomInitKernel(curandState *state, unsigned long long seed) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init((seed << 24) + i, 0, 0, &state[i]);
}


void gpuWork()
{
    uint64_t nsamples = 10000000000;

    std::ofstream of("gpu_times.txt");

    using namespace std;

	curandState *d_state;
	cudaMalloc((void**)&d_state, 1 * sizeof(curandState));

    // Launch kernel to count samples that fell inside unit circle
    uint64_t threadsPerBlock = 500;
    uint64_t num_blocks = nsamples / (1000 * threadsPerBlock);
    size_t countBlocks = num_blocks * sizeof(uint64_t);

    uint64_t *d_countInBlocks;
    cudaMalloc(&d_countInBlocks, countBlocks);

    while (gRun) {
        auto start = std::chrono::high_resolution_clock::now();

    	randomInitKernel <<<1, 1>>> (d_state, 42);
        cudaDeviceSynchronize();

        // CALL KERNEL
        count_samples_in_circles<<<num_blocks, threadsPerBlock>>>(d_state, d_countInBlocks, num_blocks, nsamples);
        if (cudaSuccess != cudaGetLastError())
            cout << "Error!\n";
        cudaDeviceSynchronize();

        // Return back the vector from device to host
        uint64_t *h_countInBlocks = new uint64_t[num_blocks];
        cudaMemcpy(h_countInBlocks, d_countInBlocks, countBlocks, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        uint64_t nsamples_in_circle = 0;
        for (uint64_t i = 0; i < num_blocks; i++)
        {
            // cout << "Value in block " + i << " is " << h_countInBlocks[i] << endl;
            nsamples_in_circle = nsamples_in_circle + h_countInBlocks[i];
        }

        // fraction that fell within (quarter) of unit circle
        float estimatedValue = 4.0 * float(nsamples_in_circle) / nsamples;

        auto stop = std::chrono::high_resolution_clock::now();
        double t_sec = std::chrono::duration<double, std::milli>(stop - start).count() / 1000.0;
        of << start.time_since_epoch().count() << " " << t_sec << " " << (fabs(estimatedValue) - 3.14 < 0.1f) << "\n"
           << std::flush;
    }
    cudaFree(d_countInBlocks);

    std::cout << "GPU finish\n";
}


void cpuWork()
{

    std::ofstream of("cpu_times.txt");

    while (gRun)
    {
        uint64_t nsamples = 1000000;

        auto start = std::chrono::high_resolution_clock::now();

        srand(42);
        uint64_t inCircle =0;
        #pragma omp parallel for reduction(+:inCircle)
        for (uint64_t i = 0; i < nsamples; i++) {

            float xValue = float(rand()) / RAND_MAX;
            float yValue = float(rand()) / RAND_MAX;

            if (xValue * xValue + yValue * yValue <= 1.0f) {
                inCircle++;
            }
        }

        float estimatedValue = 4.0 * float(inCircle) / nsamples;

        auto stop = std::chrono::high_resolution_clock::now();
        double t_sec = std::chrono::duration<double, std::milli>(stop - start).count() / 1000.0;
        of << start.time_since_epoch().count() << " " << t_sec << " " << (fabs(estimatedValue) - 3.14 < 0.1f) << "\n"
           << std::flush;
    }

    std::cout << "CPU finish\n";
}

void signal_callback_handler(int signum)
{
    printf("request stop\n");
    gRun = false;
}

/**
 * Host main routine
 */
int main(void)
{
    signal(SIGINT, signal_callback_handler);

    std::thread tGPU(gpuWork);
    std::thread tCPU(cpuWork);

    tGPU.join();
    tCPU.join();

    return 0;
}
