#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "==================================================" << std::endl;
        std::cout << "Device ID: " << i << " (" << prop.name << ")" << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "==================================================" << std::endl;

        // 1. Memory Hierarchy
        std::cout << "[Memory Hierarchy]" << std::endl;
        std::cout << "  Total Global Memory:       " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  L2 Cache Size:             " << prop.l2CacheSize / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared Memory per SM:      " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
        std::cout << "  Shared Memory per Block:   " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Total Constant Memory:     " << prop.totalConstMem / 1024 << " KB" << std::endl;
        
        // 2. Compute Resources
        std::cout << "\n[Compute Resources]" << std::endl;
        std::cout << "  Multiprocessors (SMs):     " << prop.multiProcessorCount << std::endl;
        std::cout << "  Warp Size:                 " << prop.warpSize << std::endl;
        std::cout << "  Max Threads per SM:        " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Max Threads per Block:     " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Registers per SM:      " << prop.regsPerMultiprocessor << std::endl;
        std::cout << "  Max Registers per Block:   " << prop.regsPerBlock << std::endl;

        // 3. Execution Limits
        std::cout << "\n[Execution Limits]" << std::endl;
        std::cout << "  Max Grid Size:             (" << prop.maxGridSize[0] << ", " 
                  << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Max Block Dimensions:      (" << prop.maxThreadsDim[0] << ", " 
                  << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;

        // 4. Specific for Ada (sm_89) / Performance
        std::cout << "\n[Ada/Performance Features]" << std::endl;
        std::cout << "  Memory Clock Rate:         " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory Bus Width:          " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  Asynchronous Engine Count: " << prop.asyncEngineCount << std::endl;
        std::cout << "  Unified Addressing (UVA):  " << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;
        std::cout << "==================================================\n" << std::endl;
    }

    return 0;
}
