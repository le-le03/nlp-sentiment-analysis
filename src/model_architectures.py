"""
简化的模型架构定义
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def create_cnn_model(vocab_size, max_length=200, num_features=0, use_features=False):
    """创建CNN模型"""
    if use_features and num_features > 0:
        # 多输入模型
        text_input = layers.Input(shape=(max_length,))
        features_input = layers.Input(shape=(num_features,))

        # 文本处理分支
        x = layers.Embedding(vocab_size, 128, input_length=max_length)(text_input)
        x = layers.Conv1D(128, 5, activation='relu')(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(0.5)(x)

        # 特征处理分支
        y = layers.Dense(64, activation='relu')(features_input)
        y = layers.Dropout(0.5)(y)

        # 合并
        combined = layers.concatenate([x, y])
        z = layers.Dense(64, activation='relu')(combined)
        z = layers.Dropout(0.5)(z)
        output = layers.Dense(1, activation='sigmoid')(z)

        model = models.Model(inputs=[text_input, features_input], outputs=output)
    else:
        # 单输入模型
        model = models.Sequential([
            layers.Embedding(vocab_size, 128, input_length=max_length),
            layers.Conv1D(128, 5, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_lstm_model(vocab_size, max_length=200, num_features=0, use_features=False):
    """创建LSTM模型"""
    if use_features and num_features > 0:
        text_input = layers.Input(shape=(max_length,))
        features_input = layers.Input(shape=(num_features,))

        x = layers.Embedding(vocab_size, 128, input_length=max_length)(text_input)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LSTM(32)(x)
        x = layers.Dropout(0.5)(x)

        y = layers.Dense(32, activation='relu')(features_input)
        y = layers.Dropout(0.5)(y)

        combined = layers.concatenate([x, y])
        z = layers.Dense(32, activation='relu')(combined)
        z = layers.Dropout(0.5)(z)
        output = layers.Dense(1, activation='sigmoid')(z)

        model = models.Model(inputs=[text_input, features_input], outputs=output)
    else:
        model = models.Sequential([
            layers.Embedding(vocab_size, 128, input_length=max_length),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(32),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_hybrid_model(vocab_size, max_length=200, num_features=0, use_features=False):
    """创建混合模型"""
    if use_features and num_features > 0:
        text_input = layers.Input(shape=(max_length,))
        features_input = layers.Input(shape=(num_features,))

        x = layers.Embedding(vocab_size, 128, input_length=max_length)(text_input)
        x = layers.Conv1D(64, 5, activation='relu')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.LSTM(64)(x)
        x = layers.Dropout(0.5)(x)

        y = layers.Dense(32, activation='relu')(features_input)
        y = layers.Dropout(0.5)(y)

        combined = layers.concatenate([x, y])
        z = layers.Dense(64, activation='relu')(combined)
        z = layers.Dropout(0.5)(z)
        output = layers.Dense(1, activation='sigmoid')(z)

        model = models.Model(inputs=[text_input, features_input], outputs=output)
    else:
        model = models.Sequential([
            layers.Embedding(vocab_size, 128, input_length=max_length),
            layers.Conv1D(64, 5, activation='relu'),
            layers.MaxPooling1D(2),
            layers.LSTM(64),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_simple_model(vocab_size, max_length=200):
    """创建简单模型"""
    model = models.Sequential([
        layers.Embedding(vocab_size, 128, input_length=max_length),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_model_config(model_type):
    """获取模型配置"""
    configs = {
        'cnn': {'name': 'CNN', 'description': '卷积神经网络'},
        'lstm': {'name': 'LSTM', 'description': '长短期记忆网络'},
        'hybrid': {'name': '混合模型', 'description': 'CNN-LSTM混合'},
        'simple': {'name': '简单模型', 'description': '基线模型'}
    }
    return configs.get(model_type, configs['simple'])


# 模型配置常量
MODEL_CONFIGS = {
    'cnn': get_model_config('cnn'),
    'lstm': get_model_config('lstm'),
    'hybrid': get_model_config('hybrid'),
    'simple': get_model_config('simple')
}