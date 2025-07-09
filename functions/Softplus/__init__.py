# functions/softplus/__init__.py
"""
Softplus 激活函数

理论说明: theory.md
适用场景: usage.md
可视化图像: softplus.png
"""

import torch
from torch import nn

class Softplus(nn.Module):
    r"""
    Softplus 激活函数:
    :math:`\text{Softplus}(x) = \frac{1}{\beta} \log(1 + \exp(\beta x))`
    
    其中 :math:`\beta` 是控制平滑度的参数。当 :math:`\beta \to \infty` 时，Softplus 趋近于 ReLU；
    当 :math:`\beta \to 0` 时，Softplus 趋近于线性函数。
    
    数学特性:
    - 平滑近似 ReLU 函数（"smooth ReLU"）
    - 全域连续可导，导数为 :math:`\text{Softplus}'(x) = \frac{1}{1 + \exp(-\beta x)}`（即 Sigmoid 函数）
    - 输出范围：:math:`(0, +\infty)`
    - 输入为负数时，输出平滑趋近于 0；输入为正数时，输出近似线性增长
    
    优点:
    - 平滑特性避免了 ReLU 的 "神经元死亡" 问题
    - 可导性使得优化过程更稳定
    - 输出始终为正，适合需要非负激活的场景（如变分自编码器中的方差估计）
    
    缺点:
    - 计算开销比 ReLU 高（涉及指数运算）
    - 梯度在输入绝对值很大时趋近于 0 或 1，可能导致学习缓慢
    
    参考文档: [theory.md](theory.md)
    """
    def __init__(self, beta: int = 1, threshold: int = 20):
        """
        初始化 Softplus
        
        参数:
            beta: 控制平滑度的参数，默认值 1
            threshold: 数值稳定性阈值，默认值 20
        """
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算 Softplus
        
        参数:
            x: 输入张量
            
        返回:
            Softplus 变换后的张量
        """
        return torch.nn.functional.softplus(x, beta=self.beta, threshold=self.threshold)

def softplus_grad(input: torch.Tensor, beta: int = 1) -> torch.Tensor:
    """
    计算 Softplus 的梯度（基于输入值）
    
    参数:
        input: Softplus 函数的输入值
        beta: 控制平滑度的参数
        
    返回:
        对应输入的梯度值
    """
    return torch.sigmoid(beta * input)