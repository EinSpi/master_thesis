import numpy as np

def generic_evaluation_vectorized(x, P, func):
    """
    使用广播机制对每个样本 x 的每个元素 x_i 使用参数 p_i 进行运算。

    参数:
    x: ndarray, 形状为 (B, M)
    P: ndarray, 形状为 (M, D)
    func: 自定义函数，接受两个参数 (x_i, p_i)，返回一个标量值。
          需要支持广播操作。

    返回:
    result: ndarray, 形状为 (B, M)
    """
    # 将 x 扩展为 (B, M, 1)，P 扩展为 (1, M, D)
    x_expanded = x[:, :, np.newaxis]  # 形状: (B, M, 1)
    P_expanded = P[np.newaxis, :, :]  # 形状: (1, M, D)

    # 调用用户自定义函数，支持广播
    result = func(x_expanded, P_expanded)  # 形状: (B, M)

    return result

# 示例：自定义函数（例如，线性组合）
def linear_combination(x_i, p_i):
    square_term=(x_i-np.arange(3))**2
    return np.sum(p_i*square_term, axis=-1)  # 沿最后一个轴求和


# 示例数据
B = 2
M = 4
D = 3
x = np.array([[1,3,5,7],[2,4,6,8]])
P = np.array([[1,0,0],[1,1,0],[2,1,0],[2,3,1]])

# 执行运算
result = generic_evaluation_vectorized(x, P, linear_combination)
print("输入 x:\n", x)
print("参数矩阵 P:\n", P)
print("结果 result:\n", result)