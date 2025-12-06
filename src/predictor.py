import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import joblib


class SentimentPredictor:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.tokenizer = None
        self.load_models()

    def load_models(self):
        try:
            print("加载模型...")

            if not os.path.exists(self.models_dir):
                print(f"模型目录不存在: {self.models_dir}")
                return False

            tokenizer_path = f'{self.models_dir}/tokenizer.pkl'
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = joblib.load(f)
                print("Tokenizer加载成功")

            model_names = ['cnn_model', 'lstm_model', 'hybrid_model']
            loaded_models = 0

            for name in model_names:
                model_path = f'{self.models_dir}/{name}'
                if os.path.exists(model_path):
                    self.models[name] = tf.keras.models.load_model(model_path)
                    print(f"加载模型: {name}")
                    loaded_models += 1

            print(f"成功加载 {loaded_models} 个模型")
            return loaded_models > 0

        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def preprocess_text(self, text):
        if self.tokenizer is not None:
            sequences = self.tokenizer.texts_to_sequences([text])
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            padded_sequences = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
            return padded_sequences
        else:
            raise ValueError("没有可用的文本预处理组件")

    def predict(self, text):
        if not self.models:
            return {'error': '没有可用的模型'}

        try:
            text_processed = self.preprocess_text(text)

            predictions = {}
            for name, model in self.models.items():
                pred_proba = model.predict(text_processed, verbose=0)[0][0]
                predictions[name] = {
                    'probability': float(pred_proba),
                    'sentiment': '正面' if pred_proba > 0.5 else '负面'
                }

            ensemble_proba = np.mean([pred['probability'] for pred in predictions.values()])
            ensemble_sentiment = '正面' if ensemble_proba > 0.5 else '负面'
            ensemble_confidence = ensemble_proba if ensemble_proba > 0.5 else 1 - ensemble_proba

            result = {
                'text': text,
                'sentiment': ensemble_sentiment,
                'confidence': float(ensemble_confidence),
                'ensemble_score': float(ensemble_proba),
                'individual_predictions': predictions
            }

            return result

        except Exception as e:
            return {'error': f'预测失败: {e}'}

    def get_model_info(self):
        return {
            'loaded_models': list(self.models.keys()),
            'model_count': len(self.models),
            'has_tokenizer': self.tokenizer is not None
        }