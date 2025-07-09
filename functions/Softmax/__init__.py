# functions/softmax/__init__.py
"""
Softmax 激活函数

理论说明: theory.md
适用场景: usage.md
可视化图像: softmax.png
"""

import torch
from torch import nn

class Softmax(nn.Module):
    r"""
    Softmax 激活函数:
    :math:`\text{Softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^{C} \exp(x_j)}`
    
    其中 :math:`x` 是输入向量，:math:`C` 是类别数量。
    Softmax 将多维输入映射到 (0, 1) 区间，且所有输出值之和为 1，常用于分类问题的概率分布。
    
    数学特性:
    - 输出范围：每个元素 ∈ (0, 1)，且总和为 1
    - 对输入的微小变化敏感，具有"max-like"性质
    - 指数运算放大了输入值之间的差异
    
    优点:
    - 将原始分数转换为可解释的概率分布
    - 与交叉熵损失函数配合良好
    
    缺点:
    - 计算开销较大（涉及指数运算）
    - 对异常值敏感，可能导致梯度不稳定
    - 在高维度下可能出现数值不稳定问题（如溢出或下溢）
    
    参考文档: [theory.md](theory.md)
    """
    def __init__(self, dim: int = -1):
        """
        初始化 Softmax
        
        参数:
            dim: 计算 Softmax 的维度
        """
        super().__init__()
        self.dim = dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算 Softmax
        
        参数:
            x: 输入张量
            
        返回:
            Softmax 变换后的张量
        """
        return torch.softmax(x, dim=self.dim)

def softmax_grad(input: torch.Tensor, output: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    计算 Softmax 的梯度（基于输入值和输出值）
    
    参数:
        input: Softmax 函数的输入值
        output: Softmax 函数的输出值
        dim: 计算 Softmax 的维度
        
    返回:
        Softmax 相对于输入的雅可比矩阵
    """
    # 创建一个与输入相同形状的张量，用于存储雅可比矩阵
    batch_size, num_classes = output.shape
    grad = torch.zeros(batch_size, num_classes, num_classes, device=input.device)
    
    # 计算雅可比矩阵：d(softmax_i)/d(x_j) = softmax_i * (δ_ij - softmax_j)
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                grad[:, i, j] = output[:, i] * (1 - output[:, i])
            else:
                grad[:, i, j] = -output[:, i] * output[:, j]
    
    return grad