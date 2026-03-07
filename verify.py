import numpy as np

def verify_gemm(m, n, k, val_a, val_b):
    print(f"Config: M={m}, N={n}, K={k}")
    
    # 1. 模拟 GPU 端的初始化
    # A 矩阵全为 val_a (1.0), B 矩阵全为 val_b (2.0)
    # 注意：CUDA 代码中 A, B 使用的是 float16 (half)
    A = np.full((m, k), val_a, dtype=np.float16)
    B = np.full((k, n), val_b, dtype=np.float16)
    
    print("Computing reference result using NumPy...")
    
    # 2. 计算参考结果
    # 虽然 A, B 是 float16，但累加过程在 CUDA 中是 float32
    # 我们用 float32 进行矩阵乘法以匹配 CUDA 的 accumulator 类型
    C_ref = np.matmul(A.astype(np.float32), B.astype(np.float32))
    
    # 3. 验证特定位置的值
    expected_val = k * val_a * val_b
    actual_val = C_ref[0, 0]
    
    print("-" * 30)
    print(f"Expected value at each element: {expected_val}")
    print(f"NumPy result at [0,0]:          {actual_val}")
    
    # 4. 打印前 5x5 示例用于手动对齐
    print("\nFirst 5x5 of NumPy Reference:")
    print(C_ref[:5, :5])
    
    if np.allclose(actual_val, expected_val):
        print("\nVerification Passed: NumPy matches CUDA logic.")
    else:
        print("\nVerification Failed!")

if __name__ == "__main__":
    # 参数需与 C++ 代码一致
    M, N, K = 4096, 4096, 4096
    VAL_A, VAL_B = 1.0, 2.0
    verify_gemm(M, N, K, VAL_A, VAL_B)