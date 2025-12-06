"""
集成学习训练器
包含模型训练、集成和评估功能
"""

import os
import time  # 添加缺失的time模块导入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings

warnings.filterwarnings('ignore')

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 导入自定义模块
try:
    from .model_architectures import (
        create_cnn_model, create_lstm_model, create_hybrid_model,
        create_simple_model
    )

    MODEL_ARCHITECTURES_AVAILABLE = True
except ImportError:
    MODEL_ARCHITECTURES_AVAILABLE = False
    print("⚠️  model_architectures module not available, using built-in models")


class EnsembleTrainer:
    """集成学习训练器"""

    def __init__(self, model_types: List[str] = None,
                 ensemble_method: str = 'weighted',
                 use_text_features: bool = True,
                 use_advanced_features: bool = True):
        """
        初始化训练器

        Args:
            model_types: 要训练的模型类型列表
            ensemble_method: 集成方法 ('weighted', 'average', 'voting')
            use_text_features: 是否使用文本特征
            use_advanced_features: 是否使用高级特征
        """
        self.model_types = model_types or ['cnn', 'lstm', 'hybrid']
        self.ensemble_method = ensemble_method
        self.use_text_features = use_text_features
        self.use_advanced_features = use_advanced_features

        # 存储模型和结果
        self.models = {}
        self.histories = {}
        self.predictions = {}
        self.metrics = {}
        self.ensemble_model = None
        self.ensemble_weights = {}

        # 训练配置
        self.batch_size = 64
        self.epochs = 15  # 减少epochs，因为有早停机制
        self.learning_rate = 0.001

        # 创建保存目录
        self._create_directories()

    def _create_directories(self):
        """创建保存目录"""
        directories = ['models', 'results']
        for dir_name in directories:
            os.makedirs(dir_name, exist_ok=True)

    def _create_simple_cnn_model(self, vocab_size: int, max_length: int,
                                 num_features: int = 0) -> tf.keras.Model:
        """创建简单的CNN模型（如果model_architectures不可用）"""
        if self.use_text_features and num_features > 0:
            # 多输入模型
            text_input = tf.keras.layers.Input(shape=(max_length,))
            features_input = tf.keras.layers.Input(shape=(num_features,))

            # 文本处理分支
            x = tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length)(text_input)
            x = tf.keras.layers.Conv1D(128, 5, activation='relu')(x)
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
            x = tf.keras.layers.Dropout(0.5)(x)

            # 特征处理分支
            y = tf.keras.layers.Dense(64, activation='relu')(features_input)
            y = tf.keras.layers.Dropout(0.5)(y)

            # 合并
            combined = tf.keras.layers.concatenate([x, y])
            z = tf.keras.layers.Dense(64, activation='relu')(combined)
            z = tf.keras.layers.Dropout(0.5)(z)
            output = tf.keras.layers.Dense(1, activation='sigmoid')(z)

            model = tf.keras.Model(inputs=[text_input, features_input], outputs=output)
        else:
            # 单输入模型
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length),
                tf.keras.layers.Conv1D(128, 5, activation='relu'),
                tf.keras.layers.GlobalMaxPooling1D(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _create_simple_lstm_model(self, vocab_size: int, max_length: int,
                                  num_features: int = 0) -> tf.keras.Model:
        """创建简单的LSTM模型"""
        if self.use_text_features and num_features > 0:
            text_input = tf.keras.layers.Input(shape=(max_length,))
            features_input = tf.keras.layers.Input(shape=(num_features,))

            x = tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length)(text_input)
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
            x = tf.keras.layers.Dropout(0.5)(x)

            y = tf.keras.layers.Dense(32, activation='relu')(features_input)
            y = tf.keras.layers.Dropout(0.5)(y)

            combined = tf.keras.layers.concatenate([x, y])
            z = tf.keras.layers.Dense(32, activation='relu')(combined)
            z = tf.keras.layers.Dropout(0.5)(z)
            output = tf.keras.layers.Dense(1, activation='sigmoid')(z)

            model = tf.keras.Model(inputs=[text_input, features_input], outputs=output)
        else:
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _create_simple_hybrid_model(self, vocab_size: int, max_length: int,
                                    num_features: int = 0) -> tf.keras.Model:
        """创建简单的混合模型"""
        if self.use_text_features and num_features > 0:
            text_input = tf.keras.layers.Input(shape=(max_length,))
            features_input = tf.keras.layers.Input(shape=(num_features,))

            x = tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length)(text_input)
            x = tf.keras.layers.Conv1D(64, 5, activation='relu')(x)
            x = tf.keras.layers.MaxPooling1D(2)(x)
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
            x = tf.keras.layers.Dropout(0.5)(x)

            y = tf.keras.layers.Dense(32, activation='relu')(features_input)
            y = tf.keras.layers.Dropout(0.5)(y)

            combined = tf.keras.layers.concatenate([x, y])
            z = tf.keras.layers.Dense(64, activation='relu')(combined)
            z = tf.keras.layers.Dropout(0.5)(z)
            output = tf.keras.layers.Dense(1, activation='sigmoid')(z)

            model = tf.keras.Model(inputs=[text_input, features_input], outputs=output)
        else:
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size, 128, input_length=max_length),
                tf.keras.layers.Conv1D(64, 5, activation='relu'),
                tf.keras.layers.MaxPooling1D(2),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_single_model(self, model_type: str,
                           X_train_text: np.ndarray,
                           X_train_features: np.ndarray,
                           y_train: np.ndarray,
                           X_val_text: np.ndarray,
                           X_val_features: np.ndarray,
                           y_val: np.ndarray,
                           vocab_size: int,
                           max_length: int,
                           num_features: int) -> Tuple[tf.keras.Model, Dict]:
        """
        训练单个模型

        Args:
            model_type: 模型类型
            X_train_text: 训练文本数据
            X_train_features: 训练文本特征
            y_train: 训练标签
            X_val_text: 验证文本数据
            X_val_features: 验证文本特征
            y_val: 验证标签
            vocab_size: 词汇表大小
            max_length: 序列最大长度
            num_features: 特征数量

        Returns:
            训练好的模型和训练历史
        """
        print(f"\n🚀 开始训练 {model_type}_model...")

        # 创建模型
        if MODEL_ARCHITECTURES_AVAILABLE:
            if model_type == 'cnn':
                model = create_cnn_model(
                    vocab_size=vocab_size,
                    max_length=max_length,
                    num_features=num_features if self.use_text_features else 0,
                    use_features=self.use_text_features
                )
            elif model_type == 'lstm':
                model = create_lstm_model(
                    vocab_size=vocab_size,
                    max_length=max_length,
                    num_features=num_features if self.use_text_features else 0,
                    use_features=self.use_text_features
                )
            elif model_type == 'hybrid':
                model = create_hybrid_model(
                    vocab_size=vocab_size,
                    max_length=max_length,
                    num_features=num_features if self.use_text_features else 0,
                    use_features=self.use_text_features
                )
            else:
                model = self._create_simple_cnn_model(vocab_size, max_length, num_features)
        else:
            # 使用内置的简单模型
            if model_type == 'cnn':
                model = self._create_simple_cnn_model(vocab_size, max_length, num_features)
            elif model_type == 'lstm':
                model = self._create_simple_lstm_model(vocab_size, max_length, num_features)
            elif model_type == 'hybrid':
                model = self._create_simple_hybrid_model(vocab_size, max_length, num_features)
            else:
                model = self._create_simple_cnn_model(vocab_size, max_length, num_features)

        # 设置回调函数
        model_name = f"{model_type}_model"
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,  # 减少耐心值，加快训练
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                f'models/{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]

        # 准备输入数据
        if self.use_text_features and num_features > 0:
            train_inputs = [X_train_text, X_train_features]
            val_inputs = [X_val_text, X_val_features]
        else:
            train_inputs = X_train_text
            val_inputs = X_val_text

        # 训练模型
        print(f"开始训练，共 {self.epochs} 个周期...")
        history = model.fit(
            train_inputs, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(val_inputs, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # 保存模型
        model.save(f'models/{model_name}.h5')
        print(f"💾 模型已保存: models/{model_name}.h5")

        return model, history.history

    def evaluate_model(self, model: tf.keras.Model,
                       X_test_text: np.ndarray,
                       X_test_features: np.ndarray,
                       y_test: np.ndarray,
                       model_name: str) -> Dict:
        """
        评估模型性能

        Args:
            model: 要评估的模型
            X_test_text: 测试文本数据
            X_test_features: 测试文本特征
            y_test: 测试标签
            model_name: 模型名称

        Returns:
            评估指标字典
        """
        # 准备输入数据
        if self.use_text_features and X_test_features.shape[1] > 0:
            test_inputs = [X_test_text, X_test_features]
        else:
            test_inputs = X_test_text

        # 预测
        y_pred_prob = model.predict(test_inputs, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        # 计算指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred,
            'probabilities': y_pred_prob
        }

        print(f"✅ {model_name} 评估完成:")
        print(f"   准确率: {accuracy:.4f}")
        print(f"   精确率: {precision:.4f}")
        print(f"   召回率: {recall:.4f}")
        print(f"   F1分数: {f1:.4f}")

        return metrics

    def create_weighted_ensemble(self, models_dict: Dict,
                                 X_test_text: np.ndarray,
                                 X_test_features: np.ndarray,
                                 y_test: np.ndarray) -> np.ndarray:
        """
        创建加权集成模型

        Args:
            models_dict: 模型字典
            X_test_text: 测试文本数据
            X_test_features: 测试文本特征
            y_test: 测试标签

        Returns:
            集成预测结果
        """
        print("\n🔗 创建加权集成模型...")

        # 收集模型和准确率
        model_names = list(models_dict.keys())
        accuracies = [self.metrics[name]['accuracy'] for name in model_names]

        # 计算权重（基于准确率）
        total_accuracy = sum(accuracies)
        weights = {name: acc / total_accuracy for name, acc in zip(model_names, accuracies)}

        print("模型权重:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.4f}")

        # 加权集成预测
        ensemble_predictions = None

        for name, weight in weights.items():
            model = models_dict[name]

            # 准备输入
            if self.use_text_features and X_test_features.shape[1] > 0:
                test_inputs = [X_test_text, X_test_features]
            else:
                test_inputs = X_test_text

            # 获取预测概率
            pred_prob = model.predict(test_inputs, verbose=0)

            # 加权求和
            if ensemble_predictions is None:
                ensemble_predictions = weight * pred_prob
            else:
                ensemble_predictions += weight * pred_prob

        # 转换为分类
        y_pred_ensemble = (ensemble_predictions > 0.5).astype(int).flatten()

        # 保存权重
        self.ensemble_weights = weights

        return y_pred_ensemble

    def evaluate_ensemble(self, y_test: np.ndarray,
                          y_pred_ensemble: np.ndarray,
                          model_name: str = "ensemble") -> Dict:
        """
        评估集成模型

        Args:
            y_test: 真实标签
            y_pred_ensemble: 集成预测
            model_name: 模型名称

        Returns:
            评估指标
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(y_test, y_pred_ensemble)
        precision = precision_score(y_test, y_pred_ensemble, zero_division=0)
        recall = recall_score(y_test, y_pred_ensemble, zero_division=0)
        f1 = f1_score(y_test, y_pred_ensemble, zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        print(f"\n✅ {model_name} 集成模型性能:")
        print(f"   准确率: {accuracy:.4f}")
        print(f"   精确率: {precision:.4f}")
        print(f"   召回率: {recall:.4f}")
        print(f"   F1分数: {f1:.4f}")

        # 与最佳单模型比较
        best_single_accuracy = max([m['accuracy'] for m in self.metrics.values()])
        improvement = accuracy - best_single_accuracy

        if improvement > 0:
            print(f"📈 相比最佳单模型提升: +{improvement:.4f}")
        else:
            print(f"📉 相比最佳单模型下降: {improvement:.4f}")

        return metrics

    def plot_training_history(self):
        """绘制训练历史"""
        n_models = len(self.histories)
        if n_models == 0:
            print("⚠️  没有训练历史可绘制")
            return

        fig, axes = plt.subplots(n_models, 2, figsize=(15, 5 * n_models))

        if n_models == 1:
            axes = np.array([axes])

        for idx, (model_type, history) in enumerate(self.histories.items()):
            # 准确率图
            ax1 = axes[idx, 0]
            ax1.plot(history['accuracy'], label='训练准确率', linewidth=2)
            ax1.plot(history['val_accuracy'], label='验证准确率', linewidth=2)
            ax1.set_title(f'{model_type.upper()}模型 - 准确率')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 损失图
            ax2 = axes[idx, 1]
            ax2.plot(history['loss'], label='训练损失', linewidth=2)
            ax2.plot(history['val_loss'], label='验证损失', linewidth=2)
            ax2.set_title(f'{model_type.upper()}模型 - 损失')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
        print("📈 训练历史图已保存: results/training_history.png")
        plt.show()

    def plot_model_comparison(self):
        """绘制模型比较图"""
        if not self.metrics:
            print("⚠️  没有评估指标可绘制")
            return

        # 收集模型性能
        model_names = list(self.metrics.keys())
        accuracies = [self.metrics[name]['accuracy'] for name in model_names]
        precisions = [self.metrics[name]['precision'] for name in model_names]
        recalls = [self.metrics[name]['recall'] for name in model_names]
        f1_scores = [self.metrics[name]['f1'] for name in model_names]

        # 创建DataFrame
        metrics_df = pd.DataFrame({
            'Model': model_names,
            'Accuracy': accuracies,
            'Precision': precisions,
            'Recall': recalls,
            'F1-Score': f1_scores
        })

        # 绘制条形图
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(model_names))
        width = 0.2

        ax.bar(x - 1.5 * width, accuracies, width, label='准确率', alpha=0.8)
        ax.bar(x - 0.5 * width, precisions, width, label='精确率', alpha=0.8)
        ax.bar(x + 0.5 * width, recalls, width, label='召回率', alpha=0.8)
        ax.bar(x + 1.5 * width, f1_scores, width, label='F1分数', alpha=0.8)

        ax.set_xlabel('模型')
        ax.set_ylabel('分数')
        ax.set_title('模型性能比较')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加数值标签
        for i, (acc, prec, rec, f1) in enumerate(zip(accuracies, precisions, recalls, f1_scores)):
            ax.text(i - 1.5 * width, acc + 0.01, f'{acc:.3f}', ha='center', fontsize=9)
            ax.text(i - 0.5 * width, prec + 0.01, f'{prec:.3f}', ha='center', fontsize=9)
            ax.text(i + 0.5 * width, rec + 0.01, f'{rec:.3f}', ha='center', fontsize=9)
            ax.text(i + 1.5 * width, f1 + 0.01, f'{f1:.3f}', ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        print("📊 模型比较图已保存: results/model_comparison.png")
        plt.show()

        # 保存指标到CSV
        metrics_df.to_csv('results/model_metrics.csv', index=False, encoding='utf-8-sig')
        print("📋 模型指标已保存: results/model_metrics.csv")

        return metrics_df

    def plot_confusion_matrices(self, X_test_text: np.ndarray,
                                X_test_features: np.ndarray,
                                y_test: np.ndarray):
        """绘制混淆矩阵"""
        from sklearn.metrics import confusion_matrix

        n_models = len(self.models)
        if n_models == 0:
            print("⚠️  没有模型可绘制混淆矩阵")
            return

        fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 5))

        if n_models == 1:
            axes = np.array([axes])

        # 绘制每个模型的混淆矩阵
        for idx, (model_name, model) in enumerate(self.models.items()):
            ax = axes[idx]

            # 准备输入
            if self.use_text_features and X_test_features.shape[1] > 0:
                test_inputs = [X_test_text, X_test_features]
            else:
                test_inputs = X_test_text

            # 预测
            y_pred_prob = model.predict(test_inputs, verbose=0)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()

            # 计算混淆矩阵
            cm = confusion_matrix(y_test, y_pred)

            # 绘制热图
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{model_name} 混淆矩阵')
            ax.set_xlabel('预测标签')
            ax.set_ylabel('真实标签')

        # 绘制集成模型的混淆矩阵（如果有）
        if hasattr(self, 'ensemble_predictions'):
            ax = axes[-1]
            cm = confusion_matrix(y_test, self.ensemble_predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
            ax.set_title('集成模型 混淆矩阵')
            ax.set_xlabel('预测标签')
            ax.set_ylabel('真实标签')

        plt.tight_layout()
        plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("📊 混淆矩阵图已保存: results/confusion_matrices.png")
        plt.show()

    def train(self, X_train_text: np.ndarray,
              X_train_features: np.ndarray,
              y_train: np.ndarray,
              X_val_text: np.ndarray,
              X_val_features: np.ndarray,
              y_val: np.ndarray,
              X_test_text: np.ndarray,
              X_test_features: np.ndarray,
              y_test: np.ndarray,
              vocab_size: int,
              max_length: int):
        """
        训练所有模型并创建集成

        Args:
            X_train_text: 训练文本数据
            X_train_features: 训练文本特征
            y_train: 训练标签
            X_val_text: 验证文本数据
            X_val_features: 验证文本特征
            y_val: 验证标签
            X_test_text: 测试文本数据
            X_test_features: 测试文本特征
            y_test: 测试标签
            vocab_size: 词汇表大小
            max_length: 序列最大长度
        """
        start_time = time.time()

        # 特征数量
        num_features = X_train_features.shape[1] if self.use_text_features else 0

        print("🎯 开始集成学习训练流程...")
        print(f"训练样本: {len(X_train_text)}")
        print(f"验证样本: {len(X_val_text)}")
        print(f"测试样本: {len(X_test_text)}")
        print(f"特征数量: {num_features}")

        # 训练每个模型
        for model_type in self.model_types:
            try:
                model, history = self.train_single_model(
                    model_type=model_type,
                    X_train_text=X_train_text,
                    X_train_features=X_train_features,
                    y_train=y_train,
                    X_val_text=X_val_text,
                    X_val_features=X_val_features,
                    y_val=y_val,
                    vocab_size=vocab_size,
                    max_length=max_length,
                    num_features=num_features
                )

                # 保存模型和历史
                self.models[model_type] = model
                self.histories[model_type] = history

                # 评估模型
                metrics = self.evaluate_model(
                    model=model,
                    X_test_text=X_test_text,
                    X_test_features=X_test_features,
                    y_test=y_test,
                    model_name=f"{model_type}_model"
                )

                self.metrics[model_type] = metrics

            except Exception as e:
                print(f"❌ 训练{model_type}模型时出错: {e}")
                import traceback
                traceback.print_exc()

        # 创建集成模型（如果有多个模型训练成功）
        if len(self.models) >= 2 and self.ensemble_method == 'weighted':
            try:
                ensemble_predictions = self.create_weighted_ensemble(
                    models_dict=self.models,
                    X_test_text=X_test_text,
                    X_test_features=X_test_features,
                    y_test=y_test
                )

                self.ensemble_predictions = ensemble_predictions

                # 评估集成模型
                ensemble_metrics = self.evaluate_ensemble(
                    y_test=y_test,
                    y_pred_ensemble=ensemble_predictions,
                    model_name=f"{self.ensemble_method} ensemble"
                )

                self.metrics['ensemble'] = ensemble_metrics
            except Exception as e:
                print(f"❌ 创建集成模型时出错: {e}")

        # 绘制图表
        print("\n📊 生成可视化结果...")
        try:
            self.plot_training_history()
            metrics_df = self.plot_model_comparison()
            self.plot_confusion_matrices(X_test_text, X_test_features, y_test)
        except Exception as e:
            print(f"⚠️  绘制图表时出错: {e}")

        # 打印详细结果
        print("\n📈 模型性能详细分析:")
        print("=" * 60)

        if self.metrics:
            summary_data = []
            for model_name, metric in self.metrics.items():
                if model_name != 'ensemble':
                    summary_data.append([
                        model_name,
                        f"{metric['accuracy']:.4f}",
                        f"{metric['precision']:.4f}",
                        f"{metric['recall']:.4f}",
                        f"{metric['f1']:.4f}"
                    ])

            # 创建汇总表
            if summary_data:
                import pandas as pd
                summary_df = pd.DataFrame(summary_data,
                                          columns=['模型', '准确率', '精确率', '召回率', 'F1分数'])
                print(summary_df.to_string(index=False))

        if 'ensemble' in self.metrics:
            ensemble_metric = self.metrics['ensemble']
            print(f"\n🌟 集成模型结果 ({self.ensemble_method}):")
            print(f"   准确率: {ensemble_metric['accuracy']:.4f}")
            print(f"   精确率: {ensemble_metric['precision']:.4f}")
            print(f"   召回率: {ensemble_metric['recall']:.4f}")
            print(f"   F1分数: {ensemble_metric['f1']:.4f}")

        # 计算总训练时间
        end_time = time.time()
        total_time = end_time - start_time

        print(f"\n⏱️  训练时间: {total_time:.1f}秒")

        # 保存训练总结
        self.save_training_summary(total_time)

    def save_training_summary(self, total_time: float):
        """保存训练总结"""
        import json
        import numpy as np

        # 创建一个可序列化的字典
        summary = {
            'total_time_seconds': total_time,
            'model_types': self.model_types,
            'ensemble_method': self.ensemble_method,
            'use_text_features': self.use_text_features,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate
        }

        # 处理模型指标，将numpy数组转换为列表
        if self.metrics:
            serializable_metrics = {}
            for model_name, metric in self.metrics.items():
                serializable_metrics[model_name] = {
                    'accuracy': float(metric['accuracy']),
                    'precision': float(metric['precision']),
                    'recall': float(metric['recall']),
                    'f1': float(metric['f1'])
                    # 不保存predictions和probabilities，因为它们很大
                }
            summary['model_metrics'] = serializable_metrics

        # 处理集成权重
        if self.ensemble_weights:
            serializable_weights = {}
            for model_name, weight in self.ensemble_weights.items():
                serializable_weights[model_name] = float(weight)
            summary['ensemble_weights'] = serializable_weights

        try:
            with open('results/training_summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print("💾 训练总结已保存: results/training_summary.json")
        except Exception as e:
            print(f"⚠️  保存训练总结时出错: {e}")

    def predict_single(self, text: str) -> int:
        """
        预测单个文本的情感

        Args:
            text: 输入文本

        Returns:
            预测结果 (0: 负面, 1: 正面)
        """
        # 这里需要实现文本预处理和预测
        # 由于需要加载预处理工具，这里简化实现
        print(f"预测文本: {text}")

        # 随机预测（实际应使用训练好的模型）
        import random
        return random.randint(0, 1)