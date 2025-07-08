# functions/tanh/__init__.py
"""
Tanh 激活函数

理论说明: theory.md
适用场景: usage.md
可视化图像: tanh.png
"""

import torch
from torch import nn

class Tanh(nn.Module):
    r"""
    Tanh 激活函数:
    :math:`\text{Tanh}(x) = \frac{\sinh(x)}{\cosh(x)} = \frac{e^x - e^{-x}}{e^x + e^{-x}}`
    
    将输入值压缩到 (-1, 1) 范围内，输出关于原点对称。
    
    数学特性:
    - 当 x → +∞ 时，Tanh(x) → 1
    - 当 x → -∞ 时，Tanh(x) → -1
    - 导数为: 1 - Tanh(x)^2
    
    与Sigmoid的关系:
    :math:`\text{Tanh}(x) = 2 \cdot \text{Sigmoid}(2x) - 1`
    
    优点:
    - 输出关于原点对称，更适合输入分布在负值区域的数据
    - 比Sigmoid收敛速度更快
    
    缺点:
    - 仍然存在梯度消失问题（当输入值过大或过小时，梯度接近0）
    
    参考文档: [theory.md](theory.md)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算 Tanh
        
        参数:
            x: 输入张量
            
        返回:
            Tanh 变换后的张量
        """
        return torch.tanh(x)

def tanh_grad(output: torch.Tensor) -> torch.Tensor:
    """
    计算 Tanh 的梯度（基于输出值）
    
    参数:
        output: Tanh 函数的输出值
        
    返回:
        对应输入的梯度值
    """
    return 1 - output ** 2