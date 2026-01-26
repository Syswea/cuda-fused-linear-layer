cmake_minimum_required(VERSION 3.18) # 建议 3.18+ 以更好地支持 CUDA

# 1. 环境路径定义 (根据你的 echo $CONDA_PREFIX 结果)
set(CONDA_ENV "/home/syswea/miniconda3/envs/cuda_fix")
set(CUDA_ROOT "/home/syswea/cuda-12.1")

# 2. 在 project() 之前强制指定编译器，防止 CMake 自动探测到系统编译器
set(CMAKE_C_COMPILER   "${CONDA_ENV}/bin/clang")
set(CMAKE_CXX_COMPILER "${CONDA_ENV}/bin/clang++")
set(CMAKE_CUDA_COMPILER "${CUDA_ROOT}/bin/nvcc")

# 3. 核心编译标志：建立 Conda 工具链桥梁
# --allow-unsupported-compiler: 绕过版本警告
# -ccbin: 指定 nvcc 调用的宿主编译器为 Conda 的 clang++
# -Xcompiler --gcc-toolchain: 极其关键！强制 Clang 使用 Conda 的 GCC 11 头文件，避开系统 GCC 13
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CONDA_ENV}/bin/clang++")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler --gcc-toolchain=${CONDA_ENV}")

# 4. 初始化项目
project(CudaFusedLinear LANGUAGES CXX CUDA)

# 5. 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 6. 配置包含路径
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

include_directories(SYSTEM ${CUDA_ROOT}/include)
include_directories(SYSTEM ${CONDA_ENV}/include)
include_directories(${SRC_DIR})

# 7. 查找并配置 CUDA 相关库
# 使用 NO_DEFAULT_PATH 确保只找我们指定的 CUDA 12.1 目录
find_library(CUBLAS_LIB cublas HINTS "${CUDA_ROOT}/lib64" NO_DEFAULT_PATH)
find_library(CUDNN_LIB  cudnn  HINTS "${CUDA_ROOT}/lib64" NO_DEFAULT_PATH)
find_library(CUDART_LIB cudart HINTS "${CUDA_ROOT}/lib64" NO_DEFAULT_PATH)

# 打印路径以便调试（可选）
message(STATUS "Using cublas: ${CUBLAS_LIB}")
message(STATUS "Using cudnn:  ${CUDNN_LIB}")

# 8. 定义可执行文件
set(SOURCES "${SRC_DIR}/main.cu")
add_executable(cudafused ${SOURCES})

# 9. 链接库
target_link_libraries(cudafused 
    PRIVATE
    ${CUBLAS_LIB} 
    ${CUDNN_LIB} 
    ${CUDART_LIB}
    # 链接 Conda 的动态库路径
    -L${CONDA_ENV}/lib
)

# 10. 设置 RPATH，确保运行时能找到动态链接库 (.so)
set_target_properties(cudafused PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${CUDA_ROOT}/lib64;${CONDA_ENV}/lib")