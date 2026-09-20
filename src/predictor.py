import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import joblib

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))


class SentimentPredictor:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.tokenizer = None
        self.scaler = None
        self.vectorizer = None
        self.max_length = 200
        self.load_models()

    def load_models(self):
        try:
            print("加载模型...")

            if not os.path.exists(self.models_dir):
                print(f"模型目录不存在: {self.models_dir}")
                return False

            # 加载 tokenizer
            tokenizer_path = f'{self.models_dir}/tokenizer.pkl'
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = joblib.load(f)
                print("Tokenizer加载成功")
            else:
                print("警告: 未找到 tokenizer.pkl")

            # 加载 scaler
            scaler_path = f'{self.models_dir}/scaler.pkl'
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = joblib.load(f)
                print("Scaler加载成功")

            # 加载 vectorizer
            vectorizer_path = f'{self.models_dir}/vectorizer.pkl'
            if os.path.exists(vectorizer_path):
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = joblib.load(f)
                print("Vectorizer加载成功")

            # 加载模型
            model_names = ['cnn_model', 'lstm_model', 'hybrid_model']
            loaded_models = 0

            for name in model_names:
                # 尝试带 .h5 后缀
                model_path = f'{self.models_dir}/{name}.h5'
                if not os.path.exists(model_path):
                    model_path = f'{self.models_dir}/{name}'
                
                if os.path.exists(model_path):
                    try:
                        self.models[name] = tf.keras.models.load_model(model_path)
                        print(f"加载模型: {name}")
                        loaded_models += 1
                    except Exception as e:
                        print(f"加载 {name} 失败: {e}")

            print(f"成功加载 {loaded_models} 个模型")
            return loaded_models > 0

        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def preprocess_text(self, text):
        """预处理文本，返回序列和特征"""
        from data_loader import tokenize_chinese, chinese_text_features

        # jieba 分词
        processed_text = tokenize_chinese(text)

        # 序列填充
        if self.tokenizer is not None:
            sequences = self.tokenizer.texts_to_sequences([processed_text])
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        else:
            raise ValueError("没有 tokenizer")

        # 提取文本特征
        features = chinese_text_features(processed_text)

        # TF-IDF 特征
        if self.vectorizer is not None:
            tfidf_features = self.vectorizer.transform([processed_text]).toarray()
            features = np.hstack([features, tfidf_features])

        # 标准化
        if self.scaler is not None:
            features = self.scaler.transform(features)

        return padded_sequences, features

    def predict(self, text):
        if not self.models:
            return {'error': '没有可用的模型'}

        try:
            # 预处理
            text_seq, text_features = self.preprocess_text(text)

            predictions = {}
            for name, model in self.models.items():
                # 多输入模型
                pred_proba = model.predict([text_seq, text_features], verbose=0)[0][0]
                predictions[name] = {
                    'probability': float(pred_proba),
                    'sentiment': '正面' if pred_proba > 0.5 else '负面'
                }

            # 集成预测（平均）
            ensemble_proba = np.mean([pred['probability'] for pred in predictions.values()])
            ensemble_sentiment = '正面' if ensemble_proba > 0.5 else '负面'
            ensemble_confidence = ensemble_proba if ensemble_proba > 0.5 else 1 - ensemble_proba

            result = {
                'text': text,
                'sentiment': ensemble_sentiment,
                'confidence': float(ensemble_confidence),
                'positive_prob': float(ensemble_proba),
                'negative_prob': float(1 - ensemble_proba),
                'individual_predictions': predictions
            }

            return result

        except Exception as e:
            return {'error': f'预测失败: {str(e)}'}

    def get_model_info(self):
        return {
            'loaded_models': list(self.models.keys()),
            'model_count': len(self.models),
            'has_tokenizer': self.tokenizer is not None,
            'has_scaler': self.scaler is not None,
            'has_vectorizer': self.vectorizer is not None
        }