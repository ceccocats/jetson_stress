#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <thread>
#include <fstream>
#include <cuda_runtime.h>

__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i];
        for(int j=0; j<numElements; j++)
            C[i] += B[j];
    }
}

int numElements = 100000;
size_t size = numElements * sizeof(float);
float *h_A;
float *h_B;
float *h_C;

bool gRun = true;

void gpuWork() {

    std::ofstream of("gpu_times.txt");

    float *d_A = NULL;
    cudaMalloc((void **)&d_A, size);
    float *d_B = NULL;
    cudaMalloc((void **)&d_B, size);
    float *d_C = NULL;
    cudaMalloc((void **)&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    while(gRun) {
        auto start = std::chrono::high_resolution_clock::now();

        // Launch the Vector Add CUDA Kernel
        int threadsPerBlock = 256;
        int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        auto stop = std::chrono::high_resolution_clock::now();

        double t_sec = std::chrono::duration<double, std::milli>(stop - start).count()/1000.0;
        of << start.time_since_epoch().count() << " " << t_sec <<"\n" << std::flush;
    }

    std::cout<<"GPU finish\n";
}

void cpuWork() {

    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    cudaMemcpy(A, h_A, size, cudaMemcpyHostToHost);
    cudaMemcpy(B, h_B, size, cudaMemcpyHostToHost);

    std::ofstream of("cpu_times.txt");

    while(gRun) {
        auto start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for(int i=0; i<numElements; i++) {
            C[i] = A[i];
            for(int j=0; j<numElements; j++)
                C[i] += B[j];
        }

        cudaMemcpy(C, h_C, size, cudaMemcpyHostToHost);


        auto stop = std::chrono::high_resolution_clock::now();
        double t_sec = std::chrono::duration<double, std::milli>(stop - start).count()/1000.0;
        of << start.time_since_epoch().count() << " " << t_sec <<"\n" << std::flush;
    }

    std::cout<<"CPU finish\n";
}

void signal_callback_handler(int signum) {
    printf("request stop\n");
    gRun = false;
}

/**
 * Host main routine
 */
int
main(void)
{
    signal(SIGINT, signal_callback_handler);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    srand(42);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    std::thread tGPU(gpuWork);
    std::thread tCPU(cpuWork);

    tGPU.join();
    tCPU.join();

    return 0;
}

