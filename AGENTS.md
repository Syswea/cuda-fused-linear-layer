# Developer Agent Guide: Fused Linear Layer Project

This document provides essential instructions for AI agents and developers working on the `cuda-fused-linear-layer` repository. Adhere to these guidelines to ensure consistency, performance, and compatibility with the target Ada architecture (RTX 4070).

## 1. Build and Test Commands

### Prerequisites
- CUDA Toolkit 12.1+
- Conda environment with Clang and GCC toolchain
- CMake 3.18+

### Build Instructions
The project uses a specific toolchain configuration in CMake to bridge Conda and CUDA.

```bash
# Create build directory
mkdir -p build && cd build

# Configure with CMake
# Note: CMakeLists.txt has hardcoded paths for syswea's environment. 
# Update them if necessary or use:
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
make -j$(nproc)
```

### Running the Executable
```bash
./cudafused
```

### Performance Analysis (Crucial)
Use Nsight Compute for profiling the kernels as specified in the project goals:
```bash
# Basic profile
ncu ./cudafused

# Roofline analysis (as requested in README)
ncu --section SpeedOfLight ./cudafused
```

---

## 2. Code Style and Conventions

### Language Standards
- **C++:** C++14 (as enforced by `CMakeLists.txt`)
- **CUDA:** C++14

### File Structure
- Source files: `src/*.cu`, `src/*.cpp`
- Headers: `src/*.h`, `src/*.hpp`

### Naming Conventions
- **Variables:** `camelCase` (e.g., `deviceCount`, `blasHandle`)
- **Functions:** `camelCase` (e.g., `checkCuda`, `runBenchmark`)
- **CUDA Kernels:** `snake_case` with `_kernel` suffix (e.g., `fused_linear_kernel`)
- **Constants/Macros:** `UPPER_SNAKE_CASE` (e.g., `BLOCK_SIZE`)
- **Types/Classes:** `PascalCase` (e.g., `LinearLayerConfig`)

### Imports and Includes
1. Standard library headers (`<iostream>`, `<vector>`)
2. CUDA headers (`<cuda_runtime.h>`, `<cublas_v2.h>`)
3. Local project headers (`#include "my_header.h"`)

### Error Handling
Always check CUDA API calls using a helper function.
```cpp
void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}
// Usage:
checkCuda(cudaMalloc(&d_A, size));
```

For cuBLAS and cuDNN, use similar wrappers:
```cpp
#define CHECK_CUBLAS(status) \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    }
```

### Formatting
- **Indentation:** 4 spaces (no tabs).
- **Braces:** Open brace on the same line as the statement.
- **Line Length:** Aim for <100 characters.

---

## 3. CUDA Best Practices for Ada (sm_89)

### Tensor Core Usage
- Use `nvcuda::wmma` or `mma.sync` instructions to leverage Tensor Cores.
- Target data types: `FP16` or `BF16` for optimal throughput.

### Memory Optimization
- **Shared Memory:** Use for tiling to reduce global memory pressure.
- **Asynchronous Copy:** Utilize `cp.async` (SM 80+) to move data from Global to Shared memory.
- **L2 Cache:** Optimize for the 4070's large L2 cache by tuning block sizes.

### Kernel Fusion
- Combine GEMM, Bias addition, and ReLU activation into a single epilogue.
- Keep intermediate results in registers as much as possible.

---

## 4. Documentation
- Write high-level comments in **Chinese** (as per project preference in `README.md` and `main.cu`).
- Document complex kernel logic (tiling strategy, register usage) clearly in English or Chinese.
- Keep the `README.md` updated with benchmark results and Roofline analysis findings.

## 5. Agent Instructions
- When adding a new kernel, provide a corresponding host-side benchmark.
- Always verify changes by building the project (`make`) and running the basic setup check.
- If modifying `CMakeLists.txt`, ensure RPATH and toolchain settings remain intact to avoid library linking issues in the Conda environment.
