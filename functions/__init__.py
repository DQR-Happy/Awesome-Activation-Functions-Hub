# functions/__init__.py
import pkgutil
import importlib
from pathlib import Path
from typing import Dict, Type
from torch import nn

# 自动导入所有子模块中的激活函数类
def get_all_activation_functions() -> Dict[str, Type[nn.Module]]:
    """获取所有激活函数类"""
    functions = {}
    
    # 遍历 functions 目录下的所有子目录
    for module_info in pkgutil.iter_modules([str(Path(__file__).parent)]):
        if module_info.ispkg:  # 只处理子包（如 relu, sigmoid 等）
            module_name = module_info.name
            try:
                # 导入子包
                module = importlib.import_module(f"functions.{module_name}")
                # 查找所有 nn.Module 子类
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, nn.Module) and attr != nn.Module:
                        functions[attr_name] = attr
            except ImportError:
                continue
    
    return functions