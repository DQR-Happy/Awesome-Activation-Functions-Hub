# utils/doc_generator.py
from pathlib import Path
import sys

def create_function_template(func_name):
    """创建新激活函数的目录和文件模板"""
    func_dir = Path(f"functions/{func_name.lower()}")
    func_dir.mkdir(exist_ok=True)
    
    # 创建代码文件
    (func_dir / "__init__.py").write_text(f"""\"\"\"
{func_name} 激活函数

理论说明: theory.md
适用场景: usage.md
可视化图像: {func_name.lower()}.png
\"\"\"

import torch
from torch import nn

class {func_name}(nn.Module):
    def forward(self, x):
        # 实现逻辑
        pass
""")
    
    # 创建理论文档
    (func_dir / "theory.md").write_text(f"""# {func_name} 激活函数

## 1. 公式
$$
\\text{{{func_name}}}(x) = ...
$$

## 2. 原理
...

## 3. 图像
![{func_name}]({func_name.lower()}.png)

## 4. 优缺点
- **优点**：...
- **缺点**：...
""")
    
    # 创建使用文档
    (func_dir / "usage.md").write_text(f"""# {func_name} 使用指南

## 1. 适用场景
- ...

## 2. 推荐配置
- ...

## 3. 注意事项
- ...
""")
sys.path.append(str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        print("用法: python doc_generator.py <FunctionName>")
        sys.exit(1)
    
        create_function_template(sys.argv[1])
    else:
        create_function_template('Sigmoid')