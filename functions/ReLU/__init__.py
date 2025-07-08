# functions/relu/__init__.py
"""
ReLU 激活函数

理论说明: theory.md
适用场景: usage.md
可视化图像: relu.png
"""

import torch
from torch import nn

class ReLU(nn.Module):
    r"""
    ReLU (Rectified Linear Unit) 激活函数:
    :math:`\text{ReLU}(x) = \max(0, x)`
    
    将所有负值归零，保留正值不变，是深度学习中最常用的激活函数之一。
    
    数学特性:
    - 当 x > 0 时，导数为 1
    - 当 x < 0 时，导数为 0
    - 在 x = 0 处不可导（实际实现中通常取左导数或右导数）
    
    优点:
    - 计算效率极高（只需进行阈值比较）
    - 有效缓解梯度消失问题（导数恒为 1 或 0）
    - 使网络具有稀疏性（部分神经元输出为 0）
    
    缺点:
    - 存在 "神经元死亡" 问题（当输入长期为负时，神经元不再更新）
    - 输出均值不为 0，可能导致梯度偏移
    
    参考文档: [theory.md](theory.md)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算 ReLU
        
        参数:
            x: 输入张量
            
        返回:
            ReLU 变换后的张量
        """
        return torch.nn.functional.relu(x)

def relu_grad(input: torch.Tensor) -> torch.Tensor:
    """
    计算 ReLU 的梯度（基于输入值）
    
    参数:
        input: ReLU 函数的输入值
        
    返回:
        对应输入的梯度值
    """
    return (input > 0).float()