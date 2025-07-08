# utils/benchmark.py
import time
import torch

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from functions import get_all_activation_functions

def benchmark_activation_functions(input_size=(1000, 1000), device="cpu", num_runs=100):
    """测试所有激活函数的计算速度"""
    x = torch.randn(input_size, device=device)
    results = {}
    
    for name, func in get_all_activation_functions().items():
        func = func().to(device)
        torch.cuda.synchronize() if device == "cuda" else None
        
        start_time = time.time()
        for _ in range(num_runs):
            y = func(x)
        torch.cuda.synchronize() if device == "cuda" else None
        
        elapsed = (time.time() - start_time) / num_runs
        results[name] = elapsed
    
    # 按速度排序
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    print("性能测试结果（秒/次）:")
    for name, time in sorted_results:
        print(f"{name}: {time:.6f}")
    
    return sorted_results

if __name__ == "__main__":
    benchmark_activation_functions(device="cuda" if torch.cuda.is_available() else "cpu")