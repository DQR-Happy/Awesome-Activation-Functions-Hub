# functions/gelu/__init__.py
"""
GELU (Gaussian Error Linear Unit) 激活函数

理论说明: theory.md
适用场景: usage.md
可视化图像: gelu.png
"""

import torch
from torch import nn

class GELU(nn.Module):
    r"""
    GELU (Gaussian Error Linear Unit) 激活函数:
    :math:`\text{GELU}(x) = x \cdot \Phi(x)`
    
    其中 :math:`\Phi(x)` 是标准高斯分布的累积分布函数。
    常用的近似实现为:
    :math:`\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)`
    
    数学特性:
    - 全域连续可导
    - 非线性程度比 ReLU 更高
    - 输出范围为 (-∞, +∞)，负值区域输出较小但不为零
    
    优点:
    - 在 Transformer、BERT 等模型中表现优于 ReLU
    - 平滑的曲线有助于梯度传播
    - 自动对输入进行加权（大输入得到更大权重）
    
    缺点:
    - 计算复杂度较高（涉及多次乘法和 tanh 运算）
    - 缺乏直观的物理解释
    
    参考文档: [theory.md](theory.md)
    """
    def __init__(self, approximate: str = "tanh"):
        """
        初始化 GELU
        
        参数:
            approximate: 近似方法，可选 "none" 或 "tanh"
        """
        super().__init__()
        self.approximate = approximate
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算 GELU
        
        参数:
            x: 输入张量
            
        返回:
            GELU 变换后的张量
        """
        return torch.nn.functional.gelu(x, approximate=self.approximate)

def gelu_grad(input: torch.Tensor, approximate: str = "tanh") -> torch.Tensor:
    """
    计算 GELU 的梯度（基于输入值）
    
    参数:
        input: GELU 函数的输入值
        approximate: 近似方法
        
    返回:
        对应输入的梯度值
    """
    if approximate == "none":
        cdf = 0.5 * (1.0 + torch.erf(input / torch.sqrt(torch.tensor(2.0))))
        pdf = (1.0 / torch.sqrt(torch.tensor(2.0 * torch.pi))) * torch.exp(-0.5 * input**2)
        return cdf + input * pdf
    else:  # tanh 近似
        tanh_out = torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (input + 0.044715 * input**3))
        return 0.5 * (1 + tanh_out) + 0.5 * input * (1 - tanh_out**2) * torch.sqrt(torch.tensor(2.0 / torch.pi)) * (1 + 0.134145 * input**2)