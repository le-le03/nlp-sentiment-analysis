"""
数据加载和预处理模块
支持完整IMDB数据集
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import jieba
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings

warnings.filterwarnings('ignore')

# 尝试导入可选模块
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("⚠️  vaderSentiment not available, using TextBlob only")

try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️  TextBlob not available, using basic features only")

# 下载NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('punkt')
    nltk.download('stopwords')


class DataLoader:
    """数据加载和预处理类"""

    def __init__(self,
                 use_full_dataset: bool = True,
                 max_vocab_size: int = 10000,
                 max_sequence_length: int = 200,
                 test_size: float = 0.2,
                 val_size: float = 0.2,
                 random_state: int = 42):
        """
        初始化数据加载器

        Args:
            use_full_dataset: 是否使用完整数据集
            max_vocab_size: 词汇表最大大小
            max_sequence_length: 序列最大长度
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子
        """
        self.use_full_dataset = use_full_dataset
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        # 初始化组件
        self.tokenizer = None
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(max_features=1000)

        # 初始化VADER（如果可用）
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        else:
            self.vader = None

        # 数据存储
        self.X_train_text = None
        self.X_val_text = None
        self.X_test_text = None
        self.X_train_features = None
        self.X_val_features = None
        self.X_test_features = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.vocab_size = max_vocab_size
        self.max_length = max_sequence_length

        self.logger = logging.getLogger(__name__)

    def load_imdb_data(self) -> Tuple[List[str], np.ndarray]:
        """
        加载IMDB数据集

        Returns:
            文本列表和标签数组
        """
        self.logger.info("加载IMDB数据集...")

        # 加载完整IMDB数据集
        (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
            num_words=self.max_vocab_size
        )

        # 获取词汇表
        word_index = imdb.get_word_index()
        reverse_word_index = {value: key for (key, value) in word_index.items()}

        # 解码文本
        self.logger.info("解码文本数据...")

        def decode_text(sequences):
            texts = []
            for seq in sequences:
                # 解码，索引从3开始，0:填充, 1:起始, 2:未知
                words = [reverse_word_index.get(i - 3, '?') for i in seq if i >= 3]
                texts.append(' '.join(words))
            return texts

        train_texts = decode_text(train_data)
        test_texts = decode_text(test_data)

        # 合并训练和测试集以获得完整数据集
        all_texts = train_texts + test_texts
        all_labels = np.concatenate([train_labels, test_labels])

        self.logger.info(f"加载完成: {len(all_texts)} 条样本")

        return all_texts, all_labels

    def preprocess_text(self, texts: List[str]) -> List[str]:
        """
        预处理文本

        Args:
            texts: 原始文本列表

        Returns:
            预处理后的文本列表
        """
        self.logger.info("高级文本预处理...")

        processed_texts = []

        try:
            stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))

        for text in texts:
            # 转换为小写
            text = text.lower()

            # 移除HTML标签
            text = re.sub(r'<[^>]+>', '', text)

            # 移除特殊字符和数字
            text = re.sub(r'[^a-zA-Z\s]', '', text)

            # 分词
            words = word_tokenize(text)

            # 移除停用词
            words = [word for word in words if word not in stop_words]

            # 重新组合
            processed_text = ' '.join(words)
            processed_texts.append(processed_text)

        return processed_texts

    def extract_text_features(self, texts: List[str]) -> np.ndarray:
        """
        提取文本特征

        Args:
            texts: 文本列表

        Returns:
            特征矩阵
        """
        self.logger.info("提取文本特征...")

        features = []

        for text in texts:
            feature_vector = []

            # TextBlob情感分析（如果可用）
            if TEXTBLOB_AVAILABLE:
                try:
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity
                    feature_vector.extend([polarity, subjectivity])
                except:
                    feature_vector.extend([0.0, 0.0])
            else:
                feature_vector.extend([0.0, 0.0])

            # VADER情感分析（如果可用）
            if self.vader:
                try:
                    vader_scores = self.vader.polarity_scores(text)
                    feature_vector.extend([
                        vader_scores['pos'],
                        vader_scores['neg'],
                        vader_scores['neu'],
                        vader_scores['compound']
                    ])
                except:
                    feature_vector.extend([0.0, 0.0, 0.0, 0.0])
            else:
                feature_vector.extend([0.0, 0.0, 0.0, 0.0])

            # 文本统计特征
            words = text.split()
            word_count = len(words)
            char_count = len(text)
            avg_word_length = char_count / max(word_count, 1) if word_count > 0 else 0

            # 感叹号和问号数量
            exclamation_count = text.count('!')
            question_count = text.count('?')

            # 情感词比例（简单版）
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful',
                              'fantastic', 'best', 'love', 'liked', 'enjoyed']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst',
                              'hate', 'dislike', 'boring', 'poor', 'waste']

            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            pos_ratio = pos_count / max(word_count, 1)
            neg_ratio = neg_count / max(word_count, 1)

            # 组合特征
            feature_vector.extend([
                word_count, char_count, avg_word_length,
                exclamation_count, question_count,
                pos_ratio, neg_ratio
            ])

            features.append(feature_vector)

        return np.array(features)

    def vectorize_text(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """
        向量化文本

        Args:
            texts: 文本列表
            fit: 是否拟合向量化器

        Returns:
            向量化后的文本
        """
        if fit:
            self.vectorizer.fit(texts)

        tfidf_features = self.vectorizer.transform(texts).toarray()
        return tfidf_features

    def create_sequences(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """
        创建文本序列

        Args:
            texts: 文本列表
            fit: 是否拟合tokenizer

        Returns:
            序列化文本
        """
        if fit or self.tokenizer is None:
            self.logger.info("创建tokenizer...")
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
                num_words=self.max_vocab_size,
                oov_token='<OOV>'
            )
            self.tokenizer.fit_on_texts(texts)

        # 转换为序列
        sequences = self.tokenizer.texts_to_sequences(texts)

        # 填充序列
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )

        return padded_sequences

    def split_data(self, texts: List[str], labels: np.ndarray) -> Tuple:
        """
        分割数据集

        Args:
            texts: 文本列表
            labels: 标签数组

        Returns:
            分割后的数据集
        """
        # 首先分割出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels,
            test_size=self.test_size,
            stratify=labels,
            random_state=self.random_state
        )

        # 从剩余数据中分割出验证集
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=self.random_state
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def load_data(self):
        """加载和预处理数据"""
        try:
            # 1. 加载原始数据
            texts, labels = self.load_imdb_data()

            # 2. 预处理文本
            processed_texts = self.preprocess_text(texts)

            # 3. 分割数据
            X_train_text, X_val_text, X_test_text, y_train, y_val, y_test = self.split_data(
                processed_texts, labels
            )

            # 4. 创建文本序列
            X_train_seq = self.create_sequences(X_train_text, fit=True)
            X_val_seq = self.create_sequences(X_val_text, fit=False)
            X_test_seq = self.create_sequences(X_test_text, fit=False)

            # 5. 提取文本特征
            X_train_features = self.extract_text_features(X_train_text)
            X_val_features = self.extract_text_features(X_val_text)
            X_test_features = self.extract_text_features(X_test_text)

            # 6. 向量化文本（TF-IDF）
            X_train_tfidf = self.vectorize_text(X_train_text, fit=True)
            X_val_tfidf = self.vectorize_text(X_val_text, fit=False)
            X_test_tfidf = self.vectorize_text(X_test_text, fit=False)

            # 7. 合并特征
            X_train_features = np.hstack([X_train_features, X_train_tfidf])
            X_val_features = np.hstack([X_val_features, X_val_tfidf])
            X_test_features = np.hstack([X_test_features, X_test_tfidf])

            # 8. 标准化特征
            X_train_features = self.scaler.fit_transform(X_train_features)
            X_val_features = self.scaler.transform(X_val_features)
            X_test_features = self.scaler.transform(X_test_features)

            # 存储处理后的数据
            self.X_train_text = X_train_seq
            self.X_val_text = X_val_seq
            self.X_test_text = X_test_seq
            self.X_train_features = X_train_features
            self.X_val_features = X_val_features
            self.X_test_features = X_test_features
            self.y_train = y_train
            self.y_val = y_val
            self.y_test = y_test

            # 更新词汇表大小
            self.vocab_size = min(self.max_vocab_size, len(self.tokenizer.word_index) + 1)

            self.logger.info("数据预处理完成")
            self.logger.info(f"词汇表大小: {self.vocab_size}")
            self.logger.info(f"特征数量: {X_train_features.shape[1]}")

        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            raise

    def get_processed_data(self) -> Tuple:
        """
        获取处理后的数据

        Returns:
            处理后的训练、验证、测试数据
        """
        return (self.X_train_text, self.X_train_features, self.y_train,
                self.X_val_text, self.X_val_features, self.y_val,
                self.X_test_text, self.X_test_features, self.y_test)

    def get_tokenizer(self):
        """获取tokenizer"""
        return self.tokenizer

    def get_scaler(self):
        """获取特征缩放器"""
        return self.scaler

    def get_vectorizer(self):
        """获取向量化器"""
        return self.vectorizer

    def get_data_info(self) -> Dict:
        """获取数据信息"""
        return {
            'vocab_size': self.vocab_size,
            'max_length': self.max_sequence_length,
            'num_features': self.X_train_features.shape[1] if self.X_train_features is not None else 0,
            'train_samples': len(self.X_train_text) if self.X_train_text is not None else 0,
            'val_samples': len(self.X_val_text) if self.X_val_text is not None else 0,
            'test_samples': len(self.X_test_text) if self.X_test_text is not None else 0,
            'positive_ratio': self.y_train.mean() if self.y_train is not None else 0
        }