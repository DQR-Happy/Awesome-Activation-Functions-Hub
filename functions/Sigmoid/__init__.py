# functions/sigmoid/__init__.py
"""
Sigmoid 激活函数

理论说明: theory.md
适用场景: usage.md
可视化图像: sigmoid.png
"""

import torch
from torch import nn

class Sigmoid(nn.Module):
    r"""
    Sigmoid 激活函数:
    :math:`\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}`
    
    将输入值压缩到 (0, 1) 范围内，常用于二分类问题的输出层。
    
    数学特性:
    - 当 x → +∞ 时，Sigmoid(x) → 1
    - 当 x → -∞ 时，Sigmoid(x) → 0
    - 导数为: Sigmoid(x) * (1 - Sigmoid(x))
    
    缺点:
    - 存在梯度消失问题（当输入值过大或过小时，梯度接近0）
    - 计算开销较大（涉及指数运算）
    
    参考文档: [theory.md](theory.md)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算 Sigmoid
        
        参数:
            x: 输入张量
            
        返回:
            Sigmoid 变换后的张量
        """
        return torch.sigmoid(x)

def sigmoid_grad(output: torch.Tensor) -> torch.Tensor:
    """
    计算 Sigmoid 的梯度（基于输出值）
    
    参数:
        output: Sigmoid 函数的输出值
        
    返回:
        对应输入的梯度值
    """
    return output * (1 - output)