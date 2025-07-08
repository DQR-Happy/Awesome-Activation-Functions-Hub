# functions/selu/__init__.py
"""
SELU (Scaled Exponential Linear Unit) 激活函数

理论说明: theory.md
适用场景: usage.md
可视化图像: selu.png
"""

import torch
from torch import nn

class SELU(nn.Module):
    r"""
    SELU (Scaled Exponential Linear Unit) 激活函数:
    :math:`\text{SELU}(x) = \lambda \cdot \begin{cases} 
    x, & \text{if } x > 0 \\
    \alpha \cdot (\exp(x) - 1), & \text{if } x \leq 0 
    \end{cases}`
    
    其中:
    - :math:`\alpha \approx 1.673263242354377284817042991671`
    - :math:`\lambda \approx 1.050700987355480493419334985294`
    
    这些值使得网络具有自归一化特性（Self-Normalizing），即：
    如果输入标准化为均值0和方差1，那么每一层的输出也将保持均值0和方差1。
    
    数学特性:
    - 全域连续可导
    - 自归一化保证了深度网络不会出现梯度爆炸或消失
    - 负值区域指数饱和，输出范围约为 [-αλ, +∞)
    
    优点:
    - 无需 BatchNorm 即可训练极深的网络
    - 对权重初始化和学习率不敏感
    - 在 ImageNet 和 MNIST 等任务上表现优于 ReLU
    
    缺点:
    - 严格要求输入标准化和 LeCun 初始化
    - 计算开销较大（涉及指数运算）
    
    参考文档: [theory.md](theory.md)
    """
    def __init__(self):
        """初始化 SELU（固定参数，无需外部配置）"""
        super().__init__()
        self.alpha = 1.673263242354377284817042991671
        self.lambda_ = 1.050700987355480493419334985294
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算 SELU
        
        参数:
            x: 输入张量
            
        返回:
            SELU 变换后的张量
        """
        return self.lambda_ * torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))

def selu_grad(input: torch.Tensor) -> torch.Tensor:
    """
    计算 SELU 的梯度（基于输入值）
    
    参数:
        input: SELU 函数的输入值
        
    返回:
        对应输入的梯度值
    """
    alpha = 1.673263242354377284817042991671
    lambda_ = 1.050700987355480493419334985294
    
    grad = torch.ones_like(input)
    mask = input <= 0
    grad[mask] = lambda_ * alpha * torch.exp(input[mask])
    return grad