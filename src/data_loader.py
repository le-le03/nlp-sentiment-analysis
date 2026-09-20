import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Dict, Any
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import jieba
import re
import warnings

warnings.filterwarnings('ignore')

# 关闭 jieba 加载日志
jieba.setLogLevel(20)

# 中文情感种子词
POSITIVE_WORDS = {
    "好", "不错", "喜欢", "满意", "推荐", "棒", "优秀", "舒服", "干净",
    "热情", "周到", "方便", "精致", "惊喜", "值得", "赞", "实惠", "舒适",
    "贴心", "漂亮", "愉快", "一流", "超值",
}
NEGATIVE_WORDS = {
    "差", "糟糕", "讨厌", "失望", "垃圾", "烂", "恶心", "挤", "吵",
    "脏", "贵", "慢", "敷衍", "坑", "不值", "差劲", "难吃", "难用",
    "不满意", "坑人", "忽悠", "虚假",
}
NEGATION_WORDS = {"不", "没", "没有", "无", "非", "未", "别", "莫", "毫不", "并不"}

logger = logging.getLogger(__name__)


def clean_chinese_text(text: str) -> str:
    """中文清洗：去 HTML/链接/邮箱，保留中文、英文、数字与空白。"""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_chinese(text: str) -> str:
    """jieba 分词，输出空格连接的词序列（供 keras Tokenizer 使用）。"""
    cleaned = clean_chinese_text(text)
    words = [w for w in jieba.cut(cleaned) if w.strip()]
    return " ".join(words)


def chinese_text_features(text: str) -> np.ndarray:
    """轻量中文统计特征（10 维，不依赖 TextBlob/VADER）。"""
    cleaned = clean_chinese_text(text)
    words = cleaned.split()
    word_count = len(words)
    char_count = len(cleaned)
    avg_word_len = char_count / max(word_count, 1)

    pos_hits = sum(1 for w in words if any(p in w for p in POSITIVE_WORDS))
    neg_hits = sum(1 for w in words if any(p in w for p in NEGATIVE_WORDS))
    negation_hits = sum(1 for w in words if any(n in w for n in NEGATION_WORDS))

    pos_ratio = pos_hits / max(word_count, 1)
    neg_ratio = neg_hits / max(word_count, 1)

    exclamation_count = cleaned.count('！') + cleaned.count('!')
    question_count = cleaned.count('？') + cleaned.count('?')

    return np.array([[
        word_count, char_count, avg_word_len,
        pos_hits, neg_hits, pos_ratio, neg_ratio, negation_hits,
        exclamation_count, question_count,
    ]], dtype=float)


class DataLoader:
    """中文情感数据加载器：ChnSentiCorp（酒店评论）/ 本地 CSV。"""

    def __init__(self,
                 use_full_dataset: bool = True,
                 max_vocab_size: int = 10000,
                 max_sequence_length: int = 200,
                 test_size: float = 0.2,
                 val_size: float = 0.2,
                 random_state: int = 42,
                 dataset_path: str = ""):
        self.use_full_dataset = use_full_dataset
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.dataset_path = dataset_path

        self.tokenizer = None
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(max_features=1000)

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

    def load_chinese_dataset(self) -> Tuple[List[str], np.ndarray]:
        """加载中文数据集：合并所有 TSV 文件，最大化训练数据量。"""
        data_dir = Path("data/data")

        # 合并所有 tsv 文件
        all_texts = []
        all_labels = []

        tsv_files = list(data_dir.glob("*.tsv"))
        if not tsv_files:
            raise RuntimeError("data/data/ 目录下没有找到 TSV 文件")

        for tsv_file in sorted(tsv_files):
            try:
                df = pd.read_csv(tsv_file, sep="\t")
                # 兼容列名
                text_col = None
                for col in ["text", "text_a", "sentence", "review"]:
                    if col in df.columns:
                        text_col = col
                        break
                if text_col is None:
                    text_col = df.columns[0]

                label_col = None
                for col in ["label", "sentiment", "class"]:
                    if col in df.columns:
                        label_col = col
                        break
                if label_col is None:
                    label_col = df.columns[1]

                texts = df[text_col].astype(str).tolist()
                labels = df[label_col].astype(int).tolist()
                all_texts.extend(texts)
                all_labels.extend(labels)
                self.logger.info(f"加载 {tsv_file.name}: {len(texts)} 条")
            except Exception as e:
                self.logger.warning(f"跳过 {tsv_file.name}: {e}")

        if not all_texts:
            raise RuntimeError("没有加载到任何数据")

        self.logger.info(f"总共加载: {len(all_texts)} 条样本")
        return all_texts, np.array(all_labels)

        try:
            import os
            # 使用国内 HuggingFace 镜像站，避免官网连接超时
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from datasets import load_dataset
            self.logger.info("从 HuggingFace 镜像站加载 ChnSentiCorp ...")
            ds = load_dataset("liwanlin/ChnSentiCorp", split="train")
            texts = [str(d["text"]) for d in ds]
            labels = np.array([int(d["label"]) for d in ds])
            self.logger.info(f"加载完成: {len(texts)} 条样本")
            return texts, labels
        except Exception as e:
            self.logger.warning(f"HuggingFace 加载失败: {e}")
            raise RuntimeError(
                "无法加载 ChnSentiCorp。请先运行 python prepare_data.py 下载数据，"
                "或在 data/dataset.csv 放置数据（含 text、label 两列）。"
            )

    def _load_local(self, path: str) -> Tuple[List[str], np.ndarray]:
        from pathlib import Path
        p = Path(path)
        if p.suffix == ".csv":
            df = pd.read_csv(p)
        elif p.suffix in (".tsv", ".txt"):
            df = pd.read_csv(p, sep="\t")
        else:
            raise ValueError(f"不支持的数据格式: {p.suffix}")

        # 兼容 ChnSentiCorp 的 text_a 列名
        text_col = None
        for col in ["text", "text_a", "sentence", "review"]:
            if col in df.columns:
                text_col = col
                break
        if text_col is None:
            text_col = df.columns[0]

        label_col = None
        for col in ["label", "sentiment", "class"]:
            if col in df.columns:
                label_col = col
                break
        if label_col is None:
            label_col = df.columns[1]

        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(int).to_numpy()
        self.logger.info(f"从本地加载 {len(texts)} 条: {p}")
        return texts, labels

    def preprocess_text(self, texts: List[str]) -> List[str]:
        """中文清洗 + jieba 分词。"""
        self.logger.info("中文清洗 + jieba 分词 ...")
        return [tokenize_chinese(t) for t in texts]

    def extract_text_features(self, texts: List[str]) -> np.ndarray:
        """提取中文统计特征。"""
        self.logger.info("提取中文统计特征 ...")
        rows = [chinese_text_features(t) for t in texts]
        return np.vstack(rows)

    def vectorize_text(self, texts: List[str], fit: bool = False) -> np.ndarray:
        if fit:
            self.vectorizer.fit(texts)
        return self.vectorizer.transform(texts).toarray()

    def create_sequences(self, texts: List[str], fit: bool = False) -> np.ndarray:
        if fit or self.tokenizer is None:
            self.logger.info("创建 tokenizer ...")
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
                num_words=self.max_vocab_size,
                oov_token='<OOV>'
            )
            self.tokenizer.fit_on_texts(texts)

        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )
        return padded_sequences

    def split_data(self, texts: List[str], labels: np.ndarray) -> Tuple:
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels,
            test_size=self.test_size,
            stratify=labels,
            random_state=self.random_state
        )
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=self.random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def load_data(self):
        """加载和预处理中文数据。"""
        try:
            self.logger.info("加载中文数据集 ...")
            texts, labels = self.load_chinese_dataset()

            processed_texts = self.preprocess_text(texts)

            X_train_text, X_val_text, X_test_text, y_train, y_val, y_test = self.split_data(
                processed_texts, labels
            )

            X_train_seq = self.create_sequences(X_train_text, fit=True)
            X_val_seq = self.create_sequences(X_val_text, fit=False)
            X_test_seq = self.create_sequences(X_test_text, fit=False)

            X_train_features = self.extract_text_features(X_train_text)
            X_val_features = self.extract_text_features(X_val_text)
            X_test_features = self.extract_text_features(X_test_text)

            X_train_tfidf = self.vectorize_text(X_train_text, fit=True)
            X_val_tfidf = self.vectorize_text(X_val_text, fit=False)
            X_test_tfidf = self.vectorize_text(X_test_text, fit=False)

            X_train_features = np.hstack([X_train_features, X_train_tfidf])
            X_val_features = np.hstack([X_val_features, X_val_tfidf])
            X_test_features = np.hstack([X_test_features, X_test_tfidf])

            X_train_features = self.scaler.fit_transform(X_train_features)
            X_val_features = self.scaler.transform(X_val_features)
            X_test_features = self.scaler.transform(X_test_features)

            self.X_train_text = X_train_seq
            self.X_val_text = X_val_seq
            self.X_test_text = X_test_seq
            self.X_train_features = X_train_features
            self.X_val_features = X_val_features
            self.X_test_features = X_test_features
            self.y_train = y_train
            self.y_val = y_val
            self.y_test = y_test

            self.vocab_size = min(self.max_vocab_size, len(self.tokenizer.word_index) + 1)

            self.logger.info("数据预处理完成")
            self.logger.info(f"词汇表大小: {self.vocab_size}")
            self.logger.info(f"特征数量: {X_train_features.shape[1]}")

        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            raise

    def get_processed_data(self) -> Tuple:
        return (self.X_train_text, self.X_train_features, self.y_train,
                self.X_val_text, self.X_val_features, self.y_val,
                self.X_test_text, self.X_test_features, self.y_test)

    def get_tokenizer(self):
        return self.tokenizer

    def get_scaler(self):
        return self.scaler

    def get_vectorizer(self):
        return self.vectorizer

    def get_data_info(self) -> Dict:
        return {
            'vocab_size': self.vocab_size,
            'max_length': self.max_sequence_length,
            'num_features': self.X_train_features.shape[1] if self.X_train_features is not None else 0,
            'train_samples': len(self.X_train_text) if self.X_train_text is not None else 0,
            'val_samples': len(self.X_val_text) if self.X_val_text is not None else 0,
            'test_samples': len(self.X_test_text) if self.X_test_text is not None else 0,
            'positive_ratio': self.y_train.mean() if self.y_train is not None else 0
        }
