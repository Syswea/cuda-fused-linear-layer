#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

const int BLOCK_TILE_M = 64;
const int BLOCK_TILE_N = 64;
const int BLOCK_TILE_K = 64;

__device__ __forceinline__ half relu(half val) {
    return __hmax(val, __float2half(0.0f));
}

__global__ void fused_linear_kernel(
    const half* A,
    const half* B,
    const half* bias,
    half* C,
    int M, int N, int K) {

    __shared__ half sA[2][BLOCK_TILE_M][BLOCK_TILE_K];
    __shared__ half sB[2][BLOCK_TILE_K][BLOCK_TILE_N];

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_row = threadIdx.x / 32;
    int thread_col = threadIdx.x % 32;

    int warp_row = threadIdx.x / 32;
    int warp_col = threadIdx.x % 32;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[4];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[16];

    for (int i = 0; i < 16; i++) {
        wmma::fill_fragment(c_frag[i], 0.0f);
    }

    int start_col = threadIdx.x % 16;
    int step_col = 16;
    int start_row = threadIdx.x / 16;
    int step_row = 32;

    int read_idx = 0;
    int write_idx = 1;

    int k_tile = 0;
    int total_k_tiles = (K + BLOCK_TILE_K - 1) / BLOCK_TILE_K;

    if (k_tile < total_k_tiles) {
        for (int i = start_row; i < BLOCK_TILE_M; i += step_row) {
            for (int j = start_col; j < BLOCK_TILE_K; j += step_col) {
                int global_row = block_row * BLOCK_TILE_M + i;
                int global_col = k_tile * BLOCK_TILE_K + j;
                if (global_row < M && global_col < K) {
                    sA[read_idx][i][j] = A[global_row * K + global_col];
                } else {
                    sA[read_idx][i][j] = __float2half(0.0f);
                }
            }
        }
        for (int i = start_row; i < BLOCK_TILE_K; i += step_row) {
            for (int j = start_col; j < BLOCK_TILE_N; j += step_col) {
                int global_row = k_tile * BLOCK_TILE_K + i;
                int global_col = block_col * BLOCK_TILE_N + j;
                if (global_row < K && global_col < N) {
                    sB[read_idx][i][j] = B[global_row * N + global_col];
                } else {
                    sB[read_idx][i][j] = __float2half(0.0f);
                }
            }
        }
    }
    __syncthreads();

    for (k_tile = 0; k_tile < total_k_tiles; k_tile++) {
        int next_k_tile = k_tile + 1;

        if (next_k_tile < total_k_tiles) {
            for (int i = start_row; i < BLOCK_TILE_M; i += step_row) {
                for (int j = start_col; j < BLOCK_TILE_K; j += step_col) {
                    int global_row = block_row * BLOCK_TILE_M + i;
                    int global_col = next_k_tile * BLOCK_TILE_K + j;
                    if (global_row < M && global_col < K) {
                        sA[write_idx][i][j] = A[global_row * K + global_col];
                    } else {
                        sA[write_idx][i][j] = __float2half(0.0f);
                    }
                }
            }
            for (int i = start_row; i < BLOCK_TILE_K; i += step_row) {
                for (int j = start_col; j < BLOCK_TILE_N; j += step_col) {
                    int global_row = next_k_tile * BLOCK_TILE_K + i;
                    int global_col = block_col * BLOCK_TILE_N + j;
                    if (global_row < K && global_col < N) {
                        sB[write_idx][i][j] = B[global_row * N + global_col];
                    } else {
                        sB[write_idx][i][j] = __float2half(0.0f);
                    }
                }
            }
        }

        for (int m = 0; m < 4; m++) {
            for (int n = 0; n < 4; n++) {
                for (int k_step = 0; k_step < BLOCK_TILE_K / WMMA_K; k_step++) {
                    int a_row = m * WMMA_M;
                    int a_col = k_step * WMMA_K;
                    int b_row = k_step * WMMA_K;
                    int b_col = n * WMMA_N;
                    wmma::load_matrix_sync(a_frag[m], &sA[read_idx][a_row][a_col], BLOCK_TILE_K);
                    wmma::load_matrix_sync(b_frag[n], &sB[read_idx][b_row][b_col], BLOCK_TILE_N);
                    wmma::mma_sync(c_frag[m * 4 + n], a_frag[m], b_frag[n], c_frag[m * 4 + n]);
                }
            }
        }

        __syncthreads();

        int temp = read_idx;
        read_idx = write_idx;
        write_idx = temp;
    }

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> bias_frag;
    wmma::fill_fragment(bias_frag, 0.0f);

    for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 4; n++) {
            int idx = m * 4 + n;
            for (int i = 0; i < c_frag[idx].num_elements; i++) {
                int col = n * WMMA_N + (i % WMMA_N);
                if (col < N) {
                    c_frag[idx].x[i] += __half2float(bias[col]);
                }
            }
        }
    }

    for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 4; n++) {
            int idx = m * 4 + n;
            for (int i = 0; i < c_frag[idx].num_elements; i++) {
                c_frag[idx].x[i] = max(c_frag[idx].x[i], 0.0f);
            }
        }
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> out_frag;

    for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 4; n++) {
            int idx = m * 4 + n;
            int out_row = block_row * BLOCK_TILE_M + m * WMMA_M;
            int out_col = block_col * BLOCK_TILE_N + n * WMMA_N;
            if (out_row < M && out_col < N) {
                wmma::store_matrix_sync(
                    &C[out_row * N + out_col],
                    c_frag[idx],
                    N,
                    wmma::mem_row_major);
            }
        }
    }
}

void runFusedLinearBenchmark(int M, int N, int K) {
    std::cout << "Running Fused Linear Layer (WMMA) M=" << M << ", N=" << N << ", K=" << K << std::endl;

    half *d_A, *d_B, *d_bias, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_bias, N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));

    dim3 blockDim(512);
    dim3 gridDim((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    fused_linear_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_bias, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double operations = 2.0 * M * N * K;
    double teraflops = operations / (milliseconds * 1e-3) / 1e12;

    std::cout << "Kernel Execution Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << teraflops << " TFLOPS" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_bias);
    cudaFree(d_C);

    std::cout << "Fused Linear Layer Done." << std::endl;
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    runFusedLinearBenchmark(M, N, K);
    return 0;
}
