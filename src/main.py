#!/usr/bin/env python
"""
情感分析系统主程序
支持完整IMDB数据集，解决过拟合问题
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# 创建必要的目录
def setup_directories():
    """创建必要的目录结构"""
    directories = [
        'models',
        'results',
        'logs',
        'data/raw',
        'data/processed'
    ]

    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 创建目录: {dir_name}")

    return True


# 设置日志
def setup_logging():
    """设置日志系统"""
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def print_banner():
    """打印项目横幅"""
    banner = """
==================================================
    情感分析系统 v2.0.0 - 完整数据集版本
==================================================
"""
    print(banner)


def main():
    """主函数"""
    start_time = time.time()

    # 打印横幅
    print_banner()
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # 创建目录
    print("🔧 设置项目环境...")
    setup_directories()
    print("✅ 环境设置完成")

    # 设置日志
    logger = setup_logging()
    logger.info("情感分析系统启动")

    try:
        # 直接导入 DataLoader，避免通过 __init__.py
        print("📊 加载IMDB完整数据集...")

        # 直接导入 data_loader 模块中的 DataLoader 类
        sys.path.insert(0, str(Path(__file__).parent))
        from data_loader import DataLoader

        data_loader = DataLoader(
            use_full_dataset=True,  # 使用完整数据集
            max_vocab_size=10000,  # 词汇表大小
            max_sequence_length=200,  # 序列最大长度
            test_size=0.2,  # 测试集比例
            val_size=0.2  # 验证集比例
        )

        # 加载数据
        data_loader.load_data()

        # 获取处理后的数据
        (X_train_text, X_train_features, y_train,
         X_val_text, X_val_features, y_val,
         X_test_text, X_test_features, y_test) = data_loader.get_processed_data()

        # 获取数据信息
        data_info = {
            'vocab_size': data_loader.vocab_size,
            'max_length': data_loader.max_sequence_length,
            'num_features': X_train_features.shape[1],
            'train_samples': len(X_train_text),
            'val_samples': len(X_val_text),
            'test_samples': len(X_test_text),
            'positive_ratio': y_train.mean()
        }

        # 打印数据信息
        print("✅ 数据加载完成:")
        print(f"   训练集: {data_info['train_samples']} 条样本")
        print(f"   验证集: {data_info['val_samples']} 条样本")
        print(f"   测试集: {data_info['test_samples']} 条样本")
        print(f"   正面评论比例: {data_info['positive_ratio']:.3f}")
        print(f"   词汇表大小: {data_info['vocab_size']}")
        print(f"   提取特征数量: {data_info['num_features']}")

        print("\n📈 数据分析:")
        print("-" * 30)
        print(f"训练样本: {data_info['train_samples']}")
        print(f"验证样本: {data_info['val_samples']}")
        print(f"测试样本: {data_info['test_samples']}")
        print(f"正面评论比例: {data_info['positive_ratio']:.3f}")

        # 检查是否有ensemble_trainer模块
        print("\n🧠 初始化模型训练器...")
        print("-" * 30)

        try:
            from ensemble_trainer import EnsembleTrainer

            trainer = EnsembleTrainer(
                model_types=['cnn', 'lstm', 'hybrid'],  # 要训练的模型类型
                ensemble_method='weighted',  # 集成方法
                use_text_features=True,  # 使用文本特征
                use_advanced_features=True  # 使用高级特征
            )

            # 训练模型
            trainer.train(
                X_train_text=X_train_text,
                X_train_features=X_train_features,
                y_train=y_train,
                X_val_text=X_val_text,
                X_val_features=X_val_features,
                y_val=y_val,
                X_test_text=X_test_text,
                X_test_features=X_test_features,
                y_test=y_test,
                vocab_size=data_info['vocab_size'],
                max_length=data_info['max_length']
            )

        except ImportError as e:
            print(f"⚠️  无法导入EnsembleTrainer: {e}")
            print("使用简化训练流程...")

            # 简化训练流程
            train_simple_models(
                X_train_text, X_train_features, y_train,
                X_val_text, X_val_features, y_val,
                X_test_text, X_test_features, y_test,
                data_info
            )

        # 演示预测
        print("\n🔮 预测演示:")
        print("-" * 30)

        demo_texts = [
            "This movie is absolutely fantastic! Great acting and story.",
            "Terrible film, waste of time. Boring and poorly made.",
            "An average movie with some good moments but overall disappointing.",
            "Excellent performance by the cast, highly recommended!",
            "The plot was confusing and the characters were poorly developed."
        ]

        # 简单的预测演示
        for i, text in enumerate(demo_texts, 1):
            # 这里可以调用实际的预测函数
            # 为了演示，我们使用简单的逻辑
            if any(word in text.lower() for word in ['fantastic', 'great', 'excellent', 'recommended']):
                sentiment = "正面"
            elif any(word in text.lower() for word in
                     ['terrible', 'waste', 'boring', 'poorly', 'disappointing', 'confusing']):
                sentiment = "负面"
            else:
                sentiment = "中性"

            print(f"{i}. 文本: {text}")
            print(f"   预测情感: {sentiment}")
            print("-" * 20)

        # 计算总时间
        end_time = time.time()
        total_time = end_time - start_time

        print(f"\n🎉 流程完成!")
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总耗时: {total_time:.1f}秒")

        logger.info(f"训练完成，总耗时: {total_time:.1f}秒")

    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}", exc_info=True)
        print(f"❌ 程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def train_simple_models(X_train_text, X_train_features, y_train,
                        X_val_text, X_val_features, y_val,
                        X_test_text, X_test_features, y_test,
                        data_info):
    """简化训练流程"""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    print("使用简化训练流程训练单个模型...")

    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 尝试导入model_architectures
    try:
        from model_architectures import create_cnn_model

        model = create_cnn_model(
            vocab_size=data_info['vocab_size'],
            max_length=data_info['max_length'],
            num_features=data_info['num_features'],
            use_features=True
        )
    except ImportError:
        print("⚠️  无法导入model_architectures，创建简单模型...")
        # 创建简单的模型
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(data_info['vocab_size'], 128, input_length=data_info['max_length']),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    # 准备输入数据
    train_inputs = [X_train_text, X_train_features]
    val_inputs = [X_val_text, X_val_features]
    test_inputs = [X_test_text, X_test_features]

    # 设置回调函数
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]

    # 训练模型
    print("开始训练CNN模型...")
    history = model.fit(
        train_inputs, y_train,
        batch_size=64,
        epochs=10,
        validation_data=(val_inputs, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # 保存模型
    model.save('models/cnn_model_final.h5')
    print("💾 模型已保存: models/cnn_model_final.h5")

    # 评估模型
    print("\n📊 评估模型性能...")
    test_loss, test_acc = model.evaluate(test_inputs, y_test, verbose=0)
    print(f"测试准确率: {test_acc:.4f}")
    print(f"测试损失: {test_loss:.4f}")

    # 绘制训练历史
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        # 准确率图
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='训练准确率')
        plt.plot(history.history['val_accuracy'], label='验证准确率')
        plt.title('模型准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 损失图
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('模型损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=300)
        print("📈 训练历史图已保存: results/training_history.png")
        plt.show()

    except Exception as e:
        print(f"⚠️  无法绘制训练历史图: {e}")

    return model


if __name__ == "__main__":
    main()