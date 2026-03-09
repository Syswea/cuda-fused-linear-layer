#include <iostream>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// 定义全局矩阵维度
const int M = 4096;
const int N = 4096;
const int K = 4096;

// WMMA API 标准的 Tile 维度 (Tensor Core 硬件层面的分块大小)
const int WMMA_M = 32;
const int WMMA_N = 32;
const int WMMA_K = 32;

// 简单的 CUDA 错误检查宏
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 辅助 Kernel：在 GPU 端初始化矩阵，避免 CPU 处理 float16 的兼容性问题
__global__ void init_matrix_half(half *mat, int size, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        mat[idx] = __float2half(val);
    }
}

// 核心 Kernel：WMMA 矩阵乘法 C = A x B
__global__ void wmma_gemm(half *A, half *B, float *C, int M, int N, int K) {
    // 【逻辑简述】每个 Block 只有 1 个 Warp (32个线程)。
    // 它负责计算 C 矩阵中的一个 16x16 的输出块。
    
    // 1. 计算当前 Warp 负责的 C 矩阵元素的起始行 (warpM) 和列 (warpN)
    int warpM = blockIdx.y * WMMA_M;
    int warpN = blockIdx.x * WMMA_N;

    __shared__ half a[WMMA_M][WMMA_K];
    __shared__ half b[WMMA_K][WMMA_N];
    __shared__ float c[WMMA_M][WMMA_N];

    // 2. 声明 WMMA 的片段 (Fragments)
    // A矩阵片段: 大小 16x16x16, 数据类型 half, 行主序
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    // B矩阵片段: 大小 16x16x16, 数据类型 half, 行主序
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    // C矩阵累加器片段: 大小 16x16x16, 数据类型 float (FP32 累加精度更高)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    //清空C tile
    for (int i = threadIdx.x; i < WMMA_M * WMMA_K; i += 32) {
        int row = i / WMMA_K;
        int col = i % WMMA_K;
        c[row][col] = static_cast<float>(0.0f);
    }
    __syncthreads();

    // 4. 沿着 K 维度滑动窗口，每次取 WMMA_K (16) 长度进行内积累加
    for (size_t warpK = 0; warpK < K; warpK += WMMA_K) {
        //读取A tile
        for (int i = threadIdx.x; i < WMMA_M * WMMA_K; i += 32) {
            int row = i / WMMA_K;
            int col = i % WMMA_K;
            // 计算全局内存索引并赋值
            a[row][col] = A[(warpM + row) * K + (warpK + col)];
        }

        //读取B tile
        for (int i = threadIdx.x; i < WMMA_K * WMMA_N; i += 32) {
            int row = i / WMMA_N;
            int col = i % WMMA_N;
            b[row][col] = B[(warpK + row) * N + (warpN + col)];
        }

        __syncthreads();

        for (size_t i = 0; i < 0 + WMMA_M; i += 16) {
            for (size_t j = 0; j < 0 + WMMA_N; j += 16) {
                for (size_t k = 0; k < 0 + WMMA_K; k += 16) {
                    wmma::load_matrix_sync(a_frag, &a[i][k], WMMA_K);
                    wmma::load_matrix_sync(b_frag, &b[k][j], WMMA_N);
                    wmma::load_matrix_sync(c_frag, &c[i][j], WMMA_N, wmma::mem_row_major);

                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

                    wmma::store_matrix_sync(&c[i][j], c_frag, WMMA_N, wmma::mem_row_major);
                }
            }
        }

        __syncthreads();
    }
    
    for (int i = threadIdx.x; i < WMMA_K * WMMA_N; i += 32) {
        int row = i / WMMA_N;
        int col = i % WMMA_N;
        C[(warpM + row) * N + (warpN + col)] = c[row][col];
    }

    __syncthreads();
}

int main() {
    // 在 Host 端分配内存（仅分配 C 用于验证）
    float *h_C = (float*)malloc(M * N * sizeof(float));

    // 在 Device 端分配内存
    half *d_A, *d_B;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    // 初始化矩阵：A 全为 1.0, B 全为 2.0
    int threads = 256;
    CHECK_CUDA(cudaDeviceSynchronize());
    init_matrix_half<<<(M * K + threads - 1) / threads, threads>>>(d_A, M * K, 1.0f);
    init_matrix_half<<<(K * N + threads - 1) / threads, threads>>>(d_B, K * N, 2.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 配置 WMMA Kernel 的 Grid 和 Block
    // Block 大小: 32 个线程 (正好是一个 Warp 的大小)
    dim3 blockDim(32); 
    // Grid 大小: x 轴对应 N 的分块，y 轴对应 M 的分块
    dim3 gridDim(N / WMMA_N, M / WMMA_M);

    std::cout << "Starting WMMA Matrix Multiplication..." << std::endl;
    // 启动核心 Kernel
    wmma_gemm<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 将计算结果拷贝回 CPU 进行验证
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // 验证结果: 
    // 每个 C 的元素等于 K 次累加，每次累加 1.0 * 2.0 = 2.0
    // 理论正确结果 = 4096 * 2.0 = 8192.0
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (h_C[i] != 8192.0f) {
            std::cout << "Error at index " << i << ": expected 8192.0, got " << h_C[i] << std::endl;
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "Success! All elements in C are calculated correctly (8192.0)." << std::endl;
    }

    // --- 新增：打印前 20x20 的结果 ---
    std::cout << "\nFirst 20x20 elements of matrix C:" << std::endl;
    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 20; j++) {
            // C 是行主序存储的，索引为 i * N + j
            printf("%7.1f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // 释放资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C);

    return 0;
}