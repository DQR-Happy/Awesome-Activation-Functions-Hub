# functions/leaky_relu/__init__.py
"""
LeakyReLU 激活函数

理论说明: theory.md
适用场景: usage.md
可视化图像: leaky_relu.png
"""

import torch
from torch import nn

class LeakyReLU(nn.Module):
    r"""
    LeakyReLU 激活函数:
    :math:`\text{LeakyReLU}(x) = \max(\alpha x, x)`
    
    其中 :math:`\alpha` 是一个小的正斜率（通常为 0.01），用于解决 ReLU 的 "神经元死亡" 问题。
    
    数学特性:
    - 当 x > 0 时，导数为 1
    - 当 x < 0 时，导数为 α
    - 全域可导，包括 x = 0 处（导数为 α）
    
    优点:
    - 解决了 ReLU 的 "神经元死亡" 问题
    - 保持了计算高效性和稀疏激活特性
    
    缺点:
    - α 值需要手动设定，难以自适应数据特性
    
    参考文档: [theory.md](theory.md)
    """
    def __init__(self, alpha: float = 0.01):
        """
        初始化 LeakyReLU
        
        参数:
            alpha: 负斜率系数，默认值 0.01
        """
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算 LeakyReLU
        
        参数:
            x: 输入张量
            
        返回:
            LeakyReLU 变换后的张量
        """
        return torch.nn.functional.leaky_relu(x, self.alpha)

def leaky_relu_grad(input: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """
    计算 LeakyReLU 的梯度（基于输入值）
    
    参数:
        input: LeakyReLU 函数的输入值
        alpha: 负斜率系数
        
    返回:
        对应输入的梯度值
    """
    grad = torch.ones_like(input)
    grad[input < 0] = alpha
    return grad