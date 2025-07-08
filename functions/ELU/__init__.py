# functions/elu/__init__.py
"""
ELU (Exponential Linear Unit) 激活函数

理论说明: theory.md
适用场景: usage.md
可视化图像: elu.png
"""

import torch
from torch import nn

class ELU(nn.Module):
    r"""
    ELU (Exponential Linear Unit) 激活函数:
    :math:`\text{ELU}(x) = \begin{cases} 
    x, & \text{if } x > 0 \\
    \alpha \cdot (\exp(x) - 1), & \text{if } x \leq 0 
    \end{cases}`
    
    其中 :math:`\alpha` 是一个正的超参数，控制负数区域的饱和值。
    
    数学特性:
    - 当 x > 0 时，输出线性增长
    - 当 x ≤ 0 时，输出趋近于 -α（指数饱和）
    - 导数为:
      :math:`\text{ELU}'(x) = \begin{cases} 
      1, & \text{if } x > 0 \\
      \text{ELU}(x) + \alpha, & \text{if } x \leq 0 
      \end{cases}`
    
    优点:
    - 输出均值接近 0，加速学习
    - 指数项缓解了梯度消失问题
    - 负值区域提供了更强的正则化效果
    
    缺点:
    - 指数运算计算开销较大
    - α 值需要手动调整
    
    参考文档: [theory.md](theory.md)
    """
    def __init__(self, alpha: float = 1.0):
        """
        初始化 ELU
        
        参数:
            alpha: 负数区域的饱和参数，默认值 1.0
        """
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算 ELU
        
        参数:
            x: 输入张量
            
        返回:
            ELU 变换后的张量
        """
        return torch.nn.functional.elu(x, self.alpha)

def elu_grad(input: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    计算 ELU 的梯度（基于输入值）
    
    参数:
        input: ELU 函数的输入值
        alpha: 负数区域的饱和参数
        
    返回:
        对应输入的梯度值
    """
    grad = torch.ones_like(input)
    mask = input <= 0
    grad[mask] = alpha * torch.exp(input[mask])
    return grad