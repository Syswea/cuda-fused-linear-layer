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

# RTX 4070 上 4096×4096 FP16 矩阵乘法的高性能 WMMA Kernel 设计规范

本设计面向 NVIDIA RTX 4070 Desktop（Ada Lovelace / sm_89）显卡，目标是高效执行 C = A @ B，其中 A, B, C ∈ ℝ^{4096×4096}，数据类型为 FP16（输入）和 FP32（累加输出）。以下为完整的 tile 分层、资源分配与双缓冲策略。

---

## 一、硬件约束（RTX 4070）

资源 | 值 | 说明
---|---|---
SM 数量 | 48 | 并行处理单元
每 SM Shared Memory | 164 KB | 可配置，建议 ≤128 KB 用于 shared
每 SM 寄存器总量 | 65,536 × 32-bit | 即 256 KB
最大线程数/Block | 1024 | 实际受资源限制
Warp Size | 32 | 固定
Tensor Core | 第四代 | 支持 FP16 WMMA

---

## 二、Tile 分层设计

采用三层分块策略：

层级 | 尺寸 | 负责者 | 说明
---|---|---|---
Global Matrix | 4096×4096 | Grid | 整个问题规模
Block Tile (C) | 64×64 | Block | 每 block 负责一块输出
Warp Tile (C) | 16×16 | Warp | 每 warp 负责子块
K Tile (Shared) | 64 | Block | K 维分块大小
WMMA Operation | 16×16×16 | Warp | 单次 Tensor Core 指令

- Warp 数/Block：(64/16) × (64/16) = 16 warps
- 线程数/Block：16 × 32 = 512

---

## 三、Kernel 启动配置

gridDim 设置为 (64, 64)，对应 (4096/64, 4096/64)；blockDim 设置为 512，即每个 block 包含 16 个 warps。启动时调用 kernel<<<gridDim, blockDim>>>(A, B, C, 4096, 4096, 4096)。

---

## 四、Shared Memory 布局（双缓冲）

使用双缓冲机制隐藏全局内存访问延迟。声明两组 shared memory 缓冲区 sA[2][64][64] 和 sB[2][64][64]，分别用于缓存 A 和 B 的分块数据。每组 buffer 大小为 64×64×2 字节（8 KB），双缓冲总占用 32 KB，远低于每 SM 164 KB 的 shared memory 上限。

---

## 五、双缓冲主循环逻辑

主循环按 K_TILE=64 步进遍历 K 维。在每次迭代中，首先异步预取下一个 K-tile 到备用 buffer（若未达边界），然后使用当前 buffer 执行 K_TILE/16 = 4 次 WMMA 操作（每次处理 K=16）。每次 WMMA 由 warp 内 32 线程协作完成，加载对应子块并累加到 accumulator fragment。循环末尾插入 __syncthreads() 确保预取完成后再复用 buffer。

---

## 六、资源占用分析

资源 | 用量 | 是否满足
---|---|---
Shared Memory / Block | 32 KB | 是
Registers / Thread | 约 60 | 是（occupancy 约 50%）
Threads / Block | 512 | 是
Blocks / Grid | 64×64 = 4096 | 是（可被调度器有效管理）

---

## 七、关键优化点

1. 向量化加载：使用 half4 提升从 global memory 到 shared memory 的带宽效率。
2. 协作加载：block 内所有 512 个线程并行协作填充 shared memory，最大化内存吞吐。
3. K_TILE=64：在 shared memory 容量、数据重用率和计算强度之间取得平衡。
4. WMMA Shape：固定采用 16×16×16，这是 Ada 架构 FP16 下最通用且高效的配置。
5. 输出累加：accumulator 使用 float 类型，避免 FP16 累加导致的精度损失。

---

## 八、性能预期

理论 FP16 算力约为 40 TFLOPS；预期实际性能可达 25–35 TFLOPS；完整 GEMM 运行时间估计为 5–8 毫秒（受内存带宽限制）；引入双缓冲相比无缓冲方案可提升性能 15%–25%。

---

## 九、推荐实现路径

首选使用 CUTLASS 库的 Gemm 模块配合其内置的双缓冲 pipeline，以获得接近 cuBLAS 的性能；次选手写 kernel 仅适用于教学目的或需要自定义融合操作（如 fused ReLU）的场景；若无需自定义 epilogue，应优先考虑 cuBLAS。编译时建议使用 nvcc -arch=sm_89 -O3 -use_fast_math 选项。