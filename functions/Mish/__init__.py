# functions/mish/__init__.py
"""
Mish 激活函数

理论说明: theory.md
适用场景: usage.md
可视化图像: mish.png
"""

import torch
from torch import nn

class Mish(nn.Module):
    r"""
    Mish 激活函数:
    :math:`\text{Mish}(x) = x \cdot \tanh(\text{Softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))`
    
    数学特性:
    - 全域连续可导，导数为:
      :math:`\text{Mish}'(x) = \tanh(\ln(1 + e^x)) + \frac{x}{(1 + e^{-x})(1 + \cosh(2\ln(1 + e^x)))}`
    - 输出范围: :math:`(-\infty, +\infty)`
    - 负值区域平滑，渐近于 0；正值区域无界增长
    
    优点:
    - 自门控机制允许更灵活的信息流动
    - 平滑特性缓解了梯度消失问题
    - 在 ImageNet、CIFAR-10/100 等基准测试中表现优于 ReLU 和 Swish
    - 无需参数调优，计算开销仅略高于 ReLU
    
    缺点:
    - 计算复杂度高于 ReLU（涉及多次指数和双曲函数运算）
    - 在小型模型中可能不如 ReLU 高效
    
    参考文档: [theory.md](theory.md)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算 Mish
        
        参数:
            x: 输入张量
            
        返回:
            Mish 变换后的张量
        """
        return x * torch.tanh(torch.nn.functional.softplus(x))

def mish_grad(input: torch.Tensor) -> torch.Tensor:
    """
    计算 Mish 的梯度（基于输入值）
    
    参数:
        input: Mish 函数的输入值
        
    返回:
        对应输入的梯度值
    """
    softplus = torch.nn.functional.softplus(input)
    tanh_softplus = torch.tanh(softplus)
    
    # 导数公式: tanh(softplus(x)) + x * (1 - tanh²(softplus(x))) * sigmoid(x)
    return tanh_softplus + input * (1 - tanh_softplus**2) * torch.sigmoid(input)