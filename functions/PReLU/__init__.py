# functions/prelu/__init__.py
"""
PReLU (Parametric ReLU) 激活函数

理论说明: theory.md
适用场景: usage.md
可视化图像: prelu.png
"""

import torch
from torch import nn

class PReLU(nn.Module):
    r"""
    PReLU (Parametric ReLU) 激活函数:
    :math:`\text{PReLU}(x) = \max(\alpha x, x)`
    
    与 LeakyReLU 类似，但 :math:`\alpha` 是一个可学习的参数，而非固定值。
    
    数学特性:
    - 当 x > 0 时，导数为 1
    - 当 x < 0 时，导数为 α（可学习）
    
    优点:
    - 自适应数据特性，避免手动调参
    - 缓解了 ReLU 的 "神经元死亡" 问题
    - 在 ImageNet 等大型数据集上表现优于 ReLU 和 LeakyReLU
    
    缺点:
    - 增加了模型参数数量
    - 小数据集上可能过拟合
    
    参考文档: [theory.md](theory.md)
    """
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        """
        初始化 PReLU
        
        参数:
            num_parameters: 需要学习的 α 参数数量
            init: α 的初始值，默认 0.25（He 等人推荐值）
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init).expand(num_parameters))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算 PReLU
        
        参数:
            x: 输入张量
            
        返回:
            PReLU 变换后的张量
        """
        return torch.nn.functional.prelu(x, self.alpha)

def prelu_grad(input: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    计算 PReLU 的梯度（基于输入值和当前 α）
    
    参数:
        input: PReLU 函数的输入值
        alpha: 当前的 α 参数
        
    返回:
        对应输入的梯度值
    """
    grad = torch.ones_like(input)
    grad[input < 0] = alpha
    return grad