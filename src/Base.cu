#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_fp16.h>

int main() {
    const int M = 2048, N = 2048, K = 2048;
    std::cout << "Running Base (cuBLAS + cuDNN) M=N=K=2048 (FP16)..." << std::endl;

    cublasHandle_t cublas;
    cudnnHandle_t cudnn;
    cublasCreate(&cublas);
    cudnnCreate(&cudnn);

    __half *d_A, *d_B, *d_C, *d_bias;
    cudaMalloc(&d_A, M * K * sizeof(__half));
    cudaMalloc(&d_B, K * N * sizeof(__half));
    cudaMalloc(&d_C, M * N * sizeof(__half));
    cudaMalloc(&d_bias, N * sizeof(__half));

    // 1. cuBLAS GEMM: C = A * B
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, 
                 d_B, CUDA_R_16F, N, d_A, CUDA_R_16F, K, &beta, 
                 d_C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // 2. cuDNN Bias: C = C + bias
    cudnnTensorDescriptor_t c_desc, b_desc;
    cudnnCreateTensorDescriptor(&c_desc);
    cudnnCreateTensorDescriptor(&b_desc);
    cudnnSetTensor4dDescriptor(c_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, M, N, 1, 1);
    cudnnSetTensor4dDescriptor(b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, N, 1, 1);
    
    float a_f = 1.0f, b_f = 1.0f;
    cudnnAddTensor(cudnn, &alpha, b_desc, d_bias, &a_f, c_desc, d_C);

    // 3. cuDNN Activation (ReLU)
    cudnnActivationDescriptor_t act;
    cudnnCreateActivationDescriptor(&act);
    cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);
    cudnnActivationForward(cudnn, act, &a_f, c_desc, d_C, &beta, c_desc, d_C);

    cudaDeviceSynchronize();
    std::cout << "Done." << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_bias);
    cublasDestroy(cublas); cudnnDestroy(cudnn);
    
    return 0;
}
