# src/data_loader.py
from typing import Union 
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import torch
import warnings


def load_processed_data(file_path: Path, sheet_name: Union[str, int] = 0) -> Union[pd.DataFrame, None]:
    """
    从指定的 Excel 文件加载数据。

    Args:
        file_path (Path): Excel 文件的完整路径。
        sheet_name (str or int, optional): 要加载的工作表名称或索引。默认为 0 (第一个工作表)。

    Returns:
        pd.DataFrame | None: 加载的 DataFrame，如果文件不存在或加载失败则返回 None。
    """
    if not file_path.exists():
        print(f"错误: 处理后的数据文件不存在于: {file_path}")
        return None
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name) # <<< 传入 sheet_name <<<
        print(f"成功从 {file_path} (Sheet: {sheet_name}) 加载数据。形状: {df.shape}")
        return df
    except Exception as e:
        print(f"加载处理后的数据文件 {file_path} (Sheet: {sheet_name}) 失败: {e}")
        return None


def load_npy_data(file_path: Path) -> Union[np.ndarray, None]:
    """
    从指定的 .npy 文件加载数据。

    Args:
        file_path (Path): .npy 文件的完整路径。

    Returns:
        np.ndarray | None: 加载的 NumPy 数组，如果文件不存在或加载失败则返回 None。
    """
    if not file_path.exists():
        print(f"错误: .npy 数据文件不存在于: {file_path}")
        return None
    try:
        data = np.load(file_path)
        print(f"成功从 {file_path} 加载 .npy 数据。形状: {data.shape}")
        return data
    except Exception as e:
        print(f"加载 .npy 数据文件 {file_path} 失败: {e}")
        return None


def save_performance_data(df: pd.DataFrame, file_path: Path):
    """
    保存性能评估结果到 CSV 文件。

    Args:
        df (pd.DataFrame): 包含性能指标的 DataFrame。
        file_path (Path): 保存 CSV 文件的完整路径。
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"性能评估结果已保存到: {file_path}")
    except Exception as e:
        print(f"保存性能评估结果到 {file_path} 失败: {e}")


def save_intermediate_data(obj, file_path: Path):
    """
    保存中间数据(例如DataFrame, Series, list, numpy array, torch tensor)。
    根据对象类型选择合适的保存方法。
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(obj, pd.DataFrame):
            # 对于DataFrame，可以使用parquet或feather，比pickle更高效和跨语言
            obj.to_parquet(file_path.with_suffix('.parquet')) # 推荐：更高效
            print(f"中间数据 (DataFrame) 已保存到: {file_path.with_suffix('.parquet')}")
        elif isinstance(obj, torch.Tensor):
            torch.save(obj, file_path.with_suffix('.pt')) # PyTorch tensor
            print(f"中间数据 (Tensor) 已保存到: {file_path.with_suffix('.pt')}")
        else: # 默认为 .pkl
            joblib.dump(obj, file_path) # 其他通用对象
            print(f"中间数据已保存到: {file_path}")
    except Exception as e:
        print(f"保存中间数据失败到 {file_path}, 错误: {e}")


def load_intermediate_data(file_path: Path):
    """
    加载中间数据。根据文件后缀判断加载方法。
    """
    # 尝试各种可能的后缀
    suffixes = ['.pkl', '.parquet', '.pt']
    actual_file_path = None
    for suffix in suffixes:
        potential_path = file_path.with_suffix(suffix)
        if potential_path.exists():
            actual_file_path = potential_path
            break
            
    if not actual_file_path:
        # print(f"警告: 未找到中间数据文件：{file_path} (.pkl, .parquet, .pt 均未找到)。") # 不再打印警告，因为这可能是预期行为 (第一次运行)
        return None

    try:
        if actual_file_path.suffix == '.parquet':
            obj = pd.read_parquet(actual_file_path)
            print(f"中间数据 (DataFrame) 已从 {actual_file_path} 加载。")
        elif actual_file_path.suffix == '.pt':
            obj = torch.load(actual_file_path)
            print(f"中间数据 (Tensor) 已从 {actual_file_path} 加载。")
        else: # 默认为 .pkl
            obj = joblib.load(actual_file_path)
            print(f"中间数据已从 {actual_file_path} 加载。")
        return obj
    except Exception as e:
        print(f"加载中间数据失败从 {actual_file_path}, 错误: {e}")
        return None


def save_model(model, path):
    """
    保存训练好的模型。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True) # 确保目录存在
        joblib.dump(model, path)
        print(f"模型已保存到: {path}")
    except Exception as e:
        print(f"保存模型失败: {e}")

# 新增一个用于保存性能数据的函数，如果你在main.py中用到它
def save_performance_data(df, path):
    """
    保存性能数据到 CSV 文件。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True) # 确保目录存在
        df.to_csv(path, index=False , encoding='utf-8-sig')
        print(f"性能数据已保存到: {path}")
    except Exception as e:
        print(f"保存性能数据失败: {e}")
