"""
神经网络模型定义
包含CNN、LSTM和混合模型，添加正则化防止过拟合
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from typing import Tuple


class SentimentModel:
    """情感分析模型基类"""

    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 max_length: int = 200, dropout_rate: float = 0.5,
                 l2_reg: float = 0.001):
        """
        初始化模型

        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            max_length: 序列最大长度
            dropout_rate: Dropout比率
            l2_reg: L2正则化系数
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

    def create_embedding_layer(self):
        """创建词嵌入层"""
        return layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_length,
            embeddings_regularizer=regularizers.l2(self.l2_reg)
        )

    def create_text_features_branch(self, input_features, num_features: int):
        """
        创建文本特征处理分支

        Args:
            input_features: 输入特征张量
            num_features: 特征数量

        Returns:
            处理后的特征
        """
        # 密集层处理文本特征
        x = layers.Dense(64, activation='relu',
                         kernel_regularizer=regularizers.l2(self.l2_reg))(input_features)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(32, activation='relu',
                         kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = layers.Dropout(self.dropout_rate)(x)
        return x

    def compile_model(self, model, learning_rate: float = 0.001):
        """编译模型"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model


class CNNModel(SentimentModel):
    """CNN情感分析模型"""

    def build(self, num_features: int = 0, use_features: bool = False):
        """
        构建CNN模型

        Args:
            num_features: 文本特征数量
            use_features: 是否使用文本特征

        Returns:
            编译后的模型
        """
        # 文本输入
        text_input = layers.Input(shape=(self.max_length,))

        # 词嵌入层
        embedding = self.create_embedding_layer()
        x = embedding(text_input)

        # 添加SpatialDropout1D防止过拟合
        x = layers.SpatialDropout1D(0.3)(x)

        # 卷积层1
        x = layers.Conv1D(128, 5, activation='relu',
                          kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.3)(x)

        # 卷积层2
        x = layers.Conv1D(64, 5, activation='relu',
                          kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # 如果有文本特征，添加特征分支
        if use_features and num_features > 0:
            features_input = layers.Input(shape=(num_features,))
            features_branch = self.create_text_features_branch(features_input, num_features)

            # 合并文本特征和CNN特征
            merged = layers.concatenate([x, features_branch])

            # 全连接层
            x = layers.Dense(64, activation='relu',
                             kernel_regularizer=regularizers.l2(self.l2_reg))(merged)
            x = layers.Dropout(self.dropout_rate)(x)

            # 创建多输入模型
            model = models.Model(inputs=[text_input, features_input], outputs=x)
        else:
            # 全连接层
            x = layers.Dense(64, activation='relu',
                             kernel_regularizer=regularizers.l2(self.l2_reg))(x)
            x = layers.Dropout(self.dropout_rate)(x)
            model = models.Model(inputs=text_input, outputs=x)

        # 输出层
        output = layers.Dense(1, activation='sigmoid')(x)

        # 创建最终模型
        if use_features and num_features > 0:
            final_model = models.Model(inputs=[text_input, features_input], outputs=output)
        else:
            final_model = models.Model(inputs=text_input, outputs=output)

        # 编译模型
        return self.compile_model(final_model)


class LSTMModel(SentimentModel):
    """LSTM情感分析模型"""

    def build(self, num_features: int = 0, use_features: bool = False):
        """
        构建LSTM模型

        Args:
            num_features: 文本特征数量
            use_features: 是否使用文本特征

        Returns:
            编译后的模型
        """
        # 文本输入
        text_input = layers.Input(shape=(self.max_length,))

        # 词嵌入层
        embedding = self.create_embedding_layer()
        x = embedding(text_input)

        # 添加SpatialDropout1D
        x = layers.SpatialDropout1D(0.3)(x)

        # 双向LSTM层
        x = layers.Bidirectional(
            layers.LSTM(64,
                        dropout=0.3,
                        recurrent_dropout=0.3,
                        kernel_regularizer=regularizers.l2(self.l2_reg),
                        recurrent_regularizer=regularizers.l2(self.l2_reg),
                        return_sequences=True)
        )(x)

        # 第二层LSTM
        x = layers.Bidirectional(
            layers.LSTM(32,
                        dropout=0.3,
                        recurrent_dropout=0.3,
                        kernel_regularizer=regularizers.l2(self.l2_reg),
                        recurrent_regularizer=regularizers.l2(self.l2_reg))
        )(x)

        x = layers.Dropout(self.dropout_rate)(x)

        # 如果有文本特征，添加特征分支
        if use_features and num_features > 0:
            features_input = layers.Input(shape=(num_features,))
            features_branch = self.create_text_features_branch(features_input, num_features)

            # 合并特征
            merged = layers.concatenate([x, features_branch])

            # 全连接层
            x = layers.Dense(64, activation='relu',
                             kernel_regularizer=regularizers.l2(self.l2_reg))(merged)
            x = layers.Dropout(self.dropout_rate)(x)

            # 创建多输入模型
            model = models.Model(inputs=[text_input, features_input], outputs=x)
        else:
            # 全连接层
            x = layers.Dense(64, activation='relu',
                             kernel_regularizer=regularizers.l2(self.l2_reg))(x)
            x = layers.Dropout(self.dropout_rate)(x)
            model = models.Model(inputs=text_input, outputs=x)

        # 输出层
        output = layers.Dense(1, activation='sigmoid')(x)

        # 创建最终模型
        if use_features and num_features > 0:
            final_model = models.Model(inputs=[text_input, features_input], outputs=output)
        else:
            final_model = models.Model(inputs=text_input, outputs=output)

        # 编译模型
        return self.compile_model(final_model)


class HybridModel(SentimentModel):
    """CNN-LSTM混合模型"""

    def build(self, num_features: int = 0, use_features: bool = False):
        """
        构建混合模型

        Args:
            num_features: 文本特征数量
            use_features: 是否使用文本特征

        Returns:
            编译后的模型
        """
        # 文本输入
        text_input = layers.Input(shape=(self.max_length,))

        # 词嵌入层
        embedding = self.create_embedding_layer()
        x = embedding(text_input)

        # 添加SpatialDropout1D
        x = layers.SpatialDropout1D(0.3)(x)

        # CNN部分
        conv1 = layers.Conv1D(128, 5, activation='relu',
                              kernel_regularizer=regularizers.l2(self.l2_reg))(x)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.MaxPooling1D(2)(conv1)
        conv1 = layers.Dropout(0.3)(conv1)

        conv2 = layers.Conv1D(64, 5, activation='relu',
                              kernel_regularizer=regularizers.l2(self.l2_reg))(conv1)
        conv2 = layers.BatchNormalization()(conv2)

        # LSTM部分
        lstm = layers.Bidirectional(
            layers.LSTM(64,
                        dropout=0.3,
                        recurrent_dropout=0.3,
                        kernel_regularizer=regularizers.l2(self.l2_reg),
                        recurrent_regularizer=regularizers.l2(self.l2_reg),
                        return_sequences=True)
        )(conv2)

        lstm = layers.GlobalMaxPooling1D()(lstm)
        lstm = layers.Dropout(self.dropout_rate)(lstm)

        # 如果有文本特征，添加特征分支
        if use_features and num_features > 0:
            features_input = layers.Input(shape=(num_features,))
            features_branch = self.create_text_features_branch(features_input, num_features)

            # 合并特征
            merged = layers.concatenate([lstm, features_branch])

            # 全连接层
            x = layers.Dense(128, activation='relu',
                             kernel_regularizer=regularizers.l2(self.l2_reg))(merged)
            x = layers.Dropout(self.dropout_rate)(x)

            # 创建多输入模型
            model = models.Model(inputs=[text_input, features_input], outputs=x)
        else:
            # 全连接层
            x = layers.Dense(128, activation='relu',
                             kernel_regularizer=regularizers.l2(self.l2_reg))(lstm)
            x = layers.Dropout(self.dropout_rate)(x)
            model = models.Model(inputs=text_input, outputs=x)

        # 输出层
        output = layers.Dense(1, activation='sigmoid')(x)

        # 创建最终模型
        if use_features and num_features > 0:
            final_model = models.Model(inputs=[text_input, features_input], outputs=output)
        else:
            final_model = models.Model(inputs=text_input, outputs=output)

        # 编译模型
        return self.compile_model(final_model)


class ModelFactory:
    """模型工厂类"""

    @staticmethod
    def create_model(model_type: str, vocab_size: int, embedding_dim: int = 128,
                     max_length: int = 200, num_features: int = 0,
                     use_features: bool = False, **kwargs):
        """
        创建模型

        Args:
            model_type: 模型类型 ('cnn', 'lstm', 'hybrid')
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            max_length: 序列最大长度
            num_features: 文本特征数量
            use_features: 是否使用文本特征
            **kwargs: 其他参数

        Returns:
            创建的模型
        """
        if model_type == 'cnn':
            model_class = CNNModel
        elif model_type == 'lstm':
            model_class = LSTMModel
        elif model_type == 'hybrid':
            model_class = HybridModel
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 创建模型实例
        model_instance = model_class(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_length=max_length,
            dropout_rate=kwargs.get('dropout_rate', 0.5),
            l2_reg=kwargs.get('l2_reg', 0.001)
        )

        # 构建模型
        model = model_instance.build(
            num_features=num_features,
            use_features=use_features
        )

        return model