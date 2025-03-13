#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#include <device_launch_parameters.h>

#define N 1024  // 行列サイズ

// CUDAカーネル（GPUで並列処理）
__global__ void matrixMulCUDA(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// CPU版の行列乗算
void matrixMulCPU(float* A, float* B, float* C, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                sum += A[row * n + i] * B[i * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

int main() {
    size_t bytes = N * N * sizeof(float);

    // ホストメモリ確保（CPU側）
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    float* h_C_GPU = (float*)malloc(bytes);

    // 行列の初期化（ランダム値）
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 100 / 100.0f;
        h_B[i] = rand() % 100 / 100.0f;
    }

    // **CPU実行**
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_A, h_B, h_C, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    float time_cpu = std::chrono::duration<float, std::milli>(end_cpu - start_cpu).count();
    printf("CPU Execution Time: %f ms\n", time_cpu);

    // **GPU実行**
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(N / threadsPerBlock.x, N / threadsPerBlock.y);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    matrixMulCUDA < < < blocksPerGrid, threadsPerBlock > > > (d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    float time_gpu = std::chrono::duration<float, std::milli>(end_gpu - start_gpu).count();

    printf("GPU Execution Time: %f ms\n", time_gpu);

    // 結果取得
    cudaMemcpy(h_C_GPU, d_C, bytes, cudaMemcpyDeviceToHost);

    // メモリ解放
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_GPU);

    return 0;
}

