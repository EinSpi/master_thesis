import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed

# 定义已知数据
M = 5  # S 矩阵的数量
n = 2  # S 矩阵的维度（n x n）
S_array = np.random.rand(M, n, n)  # 生成 M 个随机的 S 矩阵，形状为 (M, n, n)
a = np.array([1, 2])  # 已知的 a 向量
ones = np.ones(n)  # 全 1 向量

# 并行化计算
results = Parallel(n_jobs=-1)(
    delayed(lambda S: minimize(
        lambda x: np.sum((S @ x[:n] + x[n] * ones - a)**2),  # 目标函数
        np.append(np.ones(n), 0),  # 初始猜测
        constraints={'type': 'eq', 'fun': lambda x: np.linalg.norm(x[:n]) - 1}  # 约束条件
    ).x)(S_array[i]) for i in range(M)
)

# 提取结果
w_matrix = np.array([result[:n] for result in results])  # w 的结果矩阵
b_vector = np.array([result[n] for result in results])   # b 的结果向量

print("w_matrix (each row is a w vector):")
print(w_matrix)
print("b_vector:")
print(b_vector)