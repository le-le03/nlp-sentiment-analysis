from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2


def create_cnn_model_with_features(vocab_size=10000, sequence_length=200,
                                   embedding_dim=128, feature_dim=10,
                                   dropout_rate=0.5, l2_reg=0.001):
    """
    CNN模型 + 特征融合

    Args:
        vocab_size: 词汇表大小
        sequence_length: 序列长度
        embedding_dim: 嵌入维度
        feature_dim: 特征维度
        dropout_rate: dropout比率
        l2_reg: L2正则化参数

    Returns:
        model: Keras模型
    """
    # 文本输入
    text_input = layers.Input(shape=(sequence_length,), name='text_input')

    # 额外特征输入
    feature_input = layers.Input(shape=(feature_dim,), name='feature_input')

    # 文本处理分支
    embedding = layers.Embedding(
        vocab_size, embedding_dim,
        name='embedding'
    )(text_input)

    # 多尺度卷积
    conv1 = layers.Conv1D(
        64, 3, activation='relu', padding='same',
        kernel_regularizer=l2(l2_reg),
        name='conv1d_3gram'
    )(embedding)
    pool1 = layers.GlobalMaxPooling1D(name='global_pool_3gram')(conv1)

    conv2 = layers.Conv1D(
        64, 5, activation='relu', padding='same',
        kernel_regularizer=l2(l2_reg),
        name='conv1d_5gram'
    )(embedding)
    pool2 = layers.GlobalMaxPooling1D(name='global_pool_5gram')(conv2)

    # 文本特征合并
    text_features = layers.Concatenate(name='text_feature_concat')([pool1, pool2])
    text_features = layers.Dense(
        64, activation='relu',
        kernel_regularizer=l2(l2_reg),
        name='text_feature_dense'
    )(text_features)

    # 特征处理分支
    feature_processed = layers.Dense(
        32, activation='relu',
        kernel_regularizer=l2(l2_reg),
        name='feature_dense_1'
    )(feature_input)
    feature_processed = layers.Dropout(
        dropout_rate * 0.5,
        name='feature_dropout_1'
    )(feature_processed)

    feature_processed = layers.Dense(
        16, activation='relu',
        kernel_regularizer=l2(l2_reg),
        name='feature_dense_2'
    )(feature_processed)

    # 特征融合
    combined = layers.Concatenate(name='feature_fusion')([text_features, feature_processed])

    # 分类器
    x = layers.Dense(
        64, activation='relu',
        kernel_regularizer=l2(l2_reg),
        name='combined_dense_1'
    )(combined)
    x = layers.Dropout(dropout_rate, name='combined_dropout_1')(x)

    x = layers.Dense(
        32, activation='relu',
        kernel_regularizer=l2(l2_reg),
        name='combined_dense_2'
    )(x)
    x = layers.Dropout(dropout_rate * 0.6, name='combined_dropout_2')(x)

    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = models.Model(inputs=[text_input, feature_input], outputs=outputs)

    return model


def create_hybrid_model_with_features(vocab_size=10000, sequence_length=200,
                                      embedding_dim=128, feature_dim=10,
                                      dropout_rate=0.5, l2_reg=0.001):
    """
    混合模型 + 特征融合 (CNN + LSTM)

    Args:
        vocab_size: 词汇表大小
        sequence_length: 序列长度
        embedding_dim: 嵌入维度
        feature_dim: 特征维度
        dropout_rate: dropout比率
        l2_reg: L2正则化参数

    Returns:
        model: Keras模型
    """
    # 文本输入
    text_input = layers.Input(shape=(sequence_length,), name='text_input')

    # 额外特征输入
    feature_input = layers.Input(shape=(feature_dim,), name='feature_input')

    # 嵌入层
    embedding = layers.Embedding(
        vocab_size, embedding_dim,
        name='embedding'
    )(text_input)

    # CNN分支
    conv = layers.Conv1D(
        64, 3, activation='relu', padding='same',
        kernel_regularizer=l2(l2_reg),
        name='cnn_conv'
    )(embedding)
    conv_pool = layers.GlobalMaxPooling1D(name='cnn_pool')(conv)

    # LSTM分支
    lstm = layers.LSTM(
        64, return_sequences=True,
        dropout=dropout_rate * 0.6,
        kernel_regularizer=l2(l2_reg),
        name='lstm_1'
    )(embedding)
    lstm_pool = layers.GlobalMaxPooling1D(name='lstm_pool')(lstm)

    # 文本特征合并
    text_features = layers.Concatenate(name='text_feature_concat')([conv_pool, lstm_pool])
    text_features = layers.Dense(
        64, activation='relu',
        kernel_regularizer=l2(l2_reg),
        name='text_feature_dense'
    )(text_features)

    # 特征处理分支
    feature_processed = layers.Dense(
        32, activation='relu',
        kernel_regularizer=l2(l2_reg),
        name='feature_dense_1'
    )(feature_input)
    feature_processed = layers.Dropout(
        dropout_rate * 0.5,
        name='feature_dropout_1'
    )(feature_processed)

    feature_processed = layers.Dense(
        16, activation='relu',
        kernel_regularizer=l2(l2_reg),
        name='feature_dense_2'
    )(feature_processed)

    # 特征融合
    combined = layers.Concatenate(name='feature_fusion')([text_features, feature_processed])

    # 分类器
    x = layers.Dense(
        64, activation='relu',
        kernel_regularizer=l2(l2_reg),
        name='combined_dense_1'
    )(combined)
    x = layers.Dropout(dropout_rate, name='combined_dropout_1')(x)

    x = layers.Dense(
        32, activation='relu',
        kernel_regularizer=l2(l2_reg),
        name='combined_dense_2'
    )(x)
    x = layers.Dropout(dropout_rate * 0.6, name='combined_dropout_2')(x)

    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = models.Model(inputs=[text_input, feature_input], outputs=outputs)

    return model


def create_attention_fusion_model(vocab_size=10000, sequence_length=200,
                                  embedding_dim=128, feature_dim=10):
    """
    注意力机制特征融合模型（高级）

    Args:
        vocab_size: 词汇表大小
        sequence_length: 序列长度
        embedding_dim: 嵌入维度
        feature_dim: 特征维度

    Returns:
        model: Keras模型
    """
    # 文本输入
    text_input = layers.Input(shape=(sequence_length,), name='text_input')

    # 特征输入
    feature_input = layers.Input(shape=(feature_dim,), name='feature_input')

    # 文本处理（使用预训练BERT或类似架构的位置）
    embedding = layers.Embedding(vocab_size, embedding_dim)(text_input)

    # 双向LSTM获取上下文信息
    lstm_out = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True)
    )(embedding)

    # 注意力机制
    attention = layers.Attention()([lstm_out, lstm_out])
    text_pool = layers.GlobalAveragePooling1D()(attention)

    # 文本特征进一步处理
    text_features = layers.Dense(64, activation='relu')(text_pool)

    # 特征处理
    feature_processed = layers.Dense(32, activation='relu')(feature_input)
    feature_processed = layers.Dense(16, activation='relu')(feature_processed)

    # 注意力特征融合
    combined = layers.Concatenate()([text_features, feature_processed])

    # 分类器
    x = layers.Dense(64, activation='relu')(combined)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=[text_input, feature_input], outputs=outputs)

    return model


# 特征融合模型配置
FEATURE_FUSION_MODELS = {
    'cnn_with_features': {
        'creator': create_cnn_model_with_features,
        'params': {'dropout_rate': 0.5, 'l2_reg': 0.001}
    },
    'hybrid_with_features': {
        'creator': create_hybrid_model_with_features,
        'params': {'dropout_rate': 0.5, 'l2_reg': 0.001}
    },
    'attention_fusion': {
        'creator': create_attention_fusion_model,
        'params': {}
    }
}


def get_feature_fusion_model_config(model_name):

    if model_name not in FEATURE_FUSION_MODELS:
        raise ValueError(f"未知特征融合模型: {model_name}。可用模型: {list(FEATURE_FUSION_MODELS.keys())}")

    config = FEATURE_FUSION_MODELS[model_name]
    return config['creator'], config['params']


# 使用示例
if __name__ == "__main__":
    # 测试特征融合模型
    models_to_test = ['cnn_with_features', 'hybrid_with_features']

    for model_name in models_to_test:
        creator, params = get_feature_fusion_model_config(model_name)
        model = creator(
            vocab_size=1000,
            sequence_length=100,
            embedding_dim=50,
            feature_dim=15,
            **params
        )

        print(f"✅ {model_name} 模型创建成功")
        print(f"   参数数量: {model.count_params():,}")
        print(f"   输入层: {[input.shape for input in model.inputs]}")
        print(f"   输出层: {model.output.shape}")
        print()