# utils/visualize.py
import sys
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# 将项目根目录添加到 sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

def plot_activation_function(func_name, func, x_range=(-10, 10), num_points=10000):
    """生成激活函数的图像并保存"""
    x = torch.linspace(x_range[0], x_range[1], num_points)
    y = func(x)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x.numpy(), y.numpy())
    plt.title(f"{func_name.capitalize()} Activation Function")
    plt.grid(True)
    plt.xlabel("Input")
    plt.ylabel("Output")
    
    # 保存到对应函数目录
    output_dir = Path(f"functions/{func_name.lower()}")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"{func_name.lower()}.png")
    plt.close()
    print(f"已保存 {func_name} 激活函数图像")

def generate_all_figures():
    """为所有激活函数生成图像"""
    from functions import get_all_activation_functions
    
    for name, func_cls in get_all_activation_functions().items():
        # 先实例化类，再调用实例
        func = func_cls()  # 实例化激活函数
        plot_activation_function(name, func)

if __name__ == "__main__":
    generate_all_figures()