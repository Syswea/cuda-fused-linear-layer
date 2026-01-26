# Fused Linear Layer 性能挑战项目计划

## 项目概述
本项目计划在2-3周的时间内，通过一个高度可量化的 "Fused Linear Layer 性能挑战" 来展现算子开发能力。目标平台为RTX 4070 (Ada)，面试官期望看到如何利用Tensor Core压榨算力，并通过Roofline Model证明融合的意义。

## 📅 项目时间线与技术细分

### 第一阶段：基准建立与环境搭建 (第 1-4 天)
**核心目标：** 确定“对手”，拿到官方库的性能天花板数据。
- **技术点：** cuBLAS API 调用、nvcc 编译流程、std::chrono 或 cudaEvent_t 计时。
- **任务：**
    - 编写一个C++程序，使用cublasGemmEx (支持 FP16/BF16) 实现  $ C = A \times B $ 。
    - 接着调用一个简单的手写Bias + ReLU Kernel。
    - 记录不同规模矩阵（如  $ 1024^2, 2048^2, 4096^2 $ ）下的总端到端耗时。
- **技术支持：** 熟悉cublasHandle_t的管理以及CUDA Stream的异步概念。

### 第二阶段：核心Kernel开发 (第 5-10 天)
**核心目标：** 手写融合算子，体现对Ada架构Tensor Core的理解。
- **技术点：** wmma (Warp Level Matrix Multiply-Accumulate)、Shared Memory瓦片化、Pipeline异步拷贝。
- **任务：**
    - Naive Implementation：基于Shared Memory的Tiling实现FP16矩阵乘法。
    - Tensor Core Integration：使用nvcuda::wmma重构计算核心，适配4070的Tensor Core。
    - Epilogue Fusion：在wmma::store_matrix_sync之前，在寄存器层面完成Bias加法和ReLU激活。
- **Ada架构加分项：** 尝试理解cp.async指令（由内存直接拷贝到Shared Memory，不经过寄存器）。

### 第三阶段：深度性能分析 (第 11-15 天)
**核心目标：** 这一块是面试的“灵魂”，展示你如何发现瓶颈。
- **技术点：** Nsight Compute (ncu)、Roofline Analysis、Memory Throughput分析。
- **任务：**
    - Roofline分析：使用ncu --section SpeedOfLight导出图表。
    - 分析你的算子是在“屋檐下”（计算受限）还是“斜坡上”（访存受限）。
    - L1/L2 Cache分析：查看4070巨大的L2 Cache命中率。解释为什么增大矩阵后性能波动比旧架构小。
    - PTX/SASS检视：观察是否触发了MMA汇编指令。
- **技术支持：** 学习如何通过命令行使用ncu抓取特定Kernel。

## 🛠 技术栈清单
| 类别 | 工具/库 | 关键用途 |
| --- | --- | --- |
| 编程语言 | C++17, CUDA C++ | 算子实现 |
| 底层接口 | wmma 或 mma.sync | 驱动 Ada Tensor Core |
| 官方库 | cuBLAS, cuDNN | 作为 Baseline 性能对标 |
| 分析器 | Nsight Compute | 获取 Roofline, Sol, Bank Conflict 数据 |
| 编译器 | NVCC (CUDA 12.x+) | 针对 sm_89 (RTX 4070) 进行优化 |

## 💡 每一阶段的“分析笔记”
面试时需要展示的不是代码行数，而是这些洞察：
- **为什么融合？** 展示Nsight指标显示Baseline的Mem throughput频繁触顶，而你的Fused版Compute Throughput占比更高。
- **瓶颈在哪里？** 即使速度没超过cuBLAS，要能解释原因（例如：cuBLAS针对不同规模有几十种选择逻辑，而你只写了一种通用的Tiling策略）。
- **硬件特性利用：** “在RTX 4070上，我通过调整BlockSize，确保了最大化L2 Cache的驻留，减少了回写DRAM的频率。”