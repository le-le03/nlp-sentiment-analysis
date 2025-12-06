"""
工具函数模块
包含各种辅助功能
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from datetime import datetime


def save_model(model, filepath: str):
    """保存模型"""
    model.save(filepath)
    print(f"✅ 模型已保存: {filepath}")


def load_model(filepath: str):
    """加载模型"""
    import tensorflow as tf
    model = tf.keras.models.load_model(filepath)
    print(f"✅ 模型已加载: {filepath}")
    return model


def save_tokenizer(tokenizer, filepath: str):
    """保存tokenizer"""
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"✅ Tokenizer已保存: {filepath}")


def load_tokenizer(filepath: str):
    """加载tokenizer"""
    with open(filepath, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"✅ Tokenizer已加载: {filepath}")
    return tokenizer


def save_scaler(scaler, filepath: str):
    """保存特征缩放器"""
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ 特征缩放器已保存: {filepath}")


def load_scaler(filepath: str):
    """加载特征缩放器"""
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✅ 特征缩放器已加载: {filepath}")
    return scaler


def save_results(results: Dict, filepath: str):
    """保存结果"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ 结果已保存: {filepath}")


def load_results(filepath: str) -> Dict:
    """加载结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f"✅ 结果已加载: {filepath}")
    return results


def print_progress_bar(iteration: int, total: int, prefix: str = '',
                       suffix: str = '', length: int = 50, fill: str = '█'):
    """打印进度条"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"


def check_gpu():
    """检查GPU可用性"""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        for gpu in gpus:
            print(f"✅ 发现GPU: {gpu}")
        return True
    else:
        print("⚠️  未发现GPU，将使用CPU")
        return False


def set_random_seed(seed: int = 42):
    """设置随机种子"""
    import random
    import numpy as np
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"✅ 随机种子已设置为: {seed}")


def create_experiment_dir(experiment_name: str = None):
    """创建实验目录"""
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"

    exp_dir = os.path.join('experiments', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 创建子目录
    for subdir in ['models', 'logs', 'results', 'checkpoints']:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)

    print(f"📁 实验目录已创建: {exp_dir}")
    return exp_dir


def analyze_dataset_distribution(y_train, y_val, y_test):
    """分析数据集分布"""
    import matplotlib.pyplot as plt

    train_counts = np.bincount(y_train)
    val_counts = np.bincount(y_val)
    test_counts = np.bincount(y_test)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    datasets = [('训练集', train_counts), ('验证集', val_counts), ('测试集', test_counts)]
    colors = ['skyblue', 'lightgreen', 'lightcoral']

    for idx, (title, counts) in enumerate(datasets):
        ax = axes[idx]
        labels = ['负面', '正面']

        ax.bar(labels, counts, color=colors[idx])
        ax.set_title(title)
        ax.set_ylabel('样本数量')

        # 添加数值标签
        for i, count in enumerate(counts):
            ax.text(i, count + 0.1, str(count), ha='center')

    plt.tight_layout()
    plt.savefig('results/dataset_distribution.png', dpi=300)
    plt.show()

    return {
        'train_distribution': train_counts.tolist(),
        'val_distribution': val_counts.tolist(),
        'test_distribution': test_counts.tolist()
    }