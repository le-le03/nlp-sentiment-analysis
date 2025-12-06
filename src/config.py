"""
配置文件
"""

import os
from pathlib import Path

# 项目路径
PROJECT_ROOT = Path(__file__).parent

# 数据配置
DATA_CONFIG = {
    'use_full_dataset': True,
    'max_vocab_size': 10000,
    'max_sequence_length': 200,
    'embedding_dim': 128,
    'test_size': 0.2,
    'val_size': 0.2,
    'random_state': 42
}

# 模型配置
MODEL_CONFIG = {
    'model_types': ['cnn', 'lstm', 'hybrid'],
    'ensemble_method': 'weighted',
    'use_text_features': True,
    'use_advanced_features': True,
    'dropout_rate': 0.5,
    'l2_reg': 0.001
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 64,
    'epochs': 20,
    'learning_rate': 0.001,
    'early_stopping_patience': 5,
    'reduce_lr_patience': 3,
    'reduce_lr_factor': 0.5,
    'min_learning_rate': 1e-6
}

# 路径配置
PATHS = {
    'models': PROJECT_ROOT / 'models',
    'results': PROJECT_ROOT / 'results',
    'logs': PROJECT_ROOT / 'logs',
    'data': PROJECT_ROOT / 'data',
    'experiments': PROJECT_ROOT / 'experiments'
}


def setup_environment():
    """设置环境"""
    # 创建目录
    for path_name, path in PATHS.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"📁 创建目录: {path_name} -> {path}")

    # 设置TensorFlow日志级别
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 设置GPU内存增长
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU内存增长已启用")
        except RuntimeError as e:
            print(f"⚠️  GPU设置错误: {e}")