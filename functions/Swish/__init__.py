# functions/swish/__init__.py
"""
Swish (SiLU) 激活函数

理论说明: theory.md
适用场景: usage.md
可视化图像: swish.png
"""

import torch
from torch import nn

class Swish(nn.Module):
    r"""
    Swish (Self-Gated Linear Unit, SiLU) 激活函数:
    :math:`\text{Swish}(x) = x \cdot \sigma(x)`
    
    其中 :math:`\sigma(x)` 是 Sigmoid 函数。
    
    数学特性:
    - 全域连续可导
    - 兼具线性和非线性特性
    - 渐近性质:
      - 当 x → +∞ 时，Swish(x) → x
      - 当 x → -∞ 时，Swish(x) → 0
    
    优点:
    - 在深层网络中表现优于 ReLU（如 ImageNet 分类）
    - 平滑的曲线缓解了梯度消失问题
    - 自门控机制允许更灵活的信息流动
    
    缺点:
    - 计算开销比 ReLU 略高（涉及 Sigmoid 运算）
    - 在小模型或浅层网络中可能不如 ReLU 高效
    
    参考文档: [theory.md](theory.md)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算 Swish
        
        参数:
            x: 输入张量
            
        返回:
            Swish 变换后的张量
        """
        return x * torch.sigmoid(x)

def swish_grad(input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """
    计算 Swish 的梯度（基于输入值和输出值）
    
    参数:
        input: Swish 函数的输入值
        output: Swish 函数的输出值
        
    返回:
        对应输入的梯度值
    """
    sigmoid_x = torch.sigmoid(input)
    return sigmoid_x * (1 + input * (1 - sigmoid_x))