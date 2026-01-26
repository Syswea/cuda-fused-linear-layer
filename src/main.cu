#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

int main() {
    std::cout << "=== CUDA & cuDNN & cuBLAS Setup Check ===" << std::endl;

    // 1. 检测 CUDA Runtime
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cout << "No CUDA devices found!" << std::endl;
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU Device: " << prop.name << std::endl;

    // 2. 检测 cuBLAS
    cublasHandle_t blas_handle;
    cublasStatus_t blas_status = cublasCreate(&blas_handle);
    if (blas_status == CUBLAS_STATUS_SUCCESS) {
        std::cout << "cuBLAS Initialization: SUCCESS" << std::endl;
        cublasDestroy(blas_handle);
    } else {
        std::cout << "cuBLAS Initialization: FAILED" << std::endl;
    }

    // 3. 检测 cuDNN
    size_t cudnn_ver = cudnnGetVersion();
    std::cout << "cuDNN Runtime Version: " << cudnn_ver << std::endl;

    cudnnHandle_t dnn_handle;
    cudnnStatus_t dnn_status = cudnnCreate(&dnn_handle);
    if (dnn_status == CUDNN_STATUS_SUCCESS) {
        std::cout << "cuDNN Initialization: SUCCESS" << std::endl;
        cudnnDestroy(dnn_handle);
    } else {
        std::cout << "cuDNN Initialization: FAILED (Check your library paths)" << std::endl;
    }

    std::cout << "==========================================" << std::endl;
    std::cout << "Test Finished!" << std::endl;

    return 0;
}