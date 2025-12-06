import nltk
import re
import string
import jieba
import jieba.posseg as pseg
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import numpy as np

# 可选进度条
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    tqdm = lambda x: x
    TQDM_AVAILABLE = False


class MultiLanguageTextPreprocessor:
    def __init__(self, language='english'):
        self.language = language.lower()
        self.lemmatizer = None
        self.stemmer = None
        self.sia = None
        self.stop_words = set()

        self._setup_language_tools()

    def _setup_language_tools(self):
        if self.language == 'english':
            self.lemmatizer = WordNetLemmatizer()
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
            self._setup_english_stopwords()
            self.sia = SentimentIntensityAnalyzer()

        elif self.language == 'chinese':
            self._setup_chinese_stopwords()
        else:
            raise ValueError(f"不支持的语言: {self.language}")

    def _setup_english_stopwords(self):
        movie_stopwords = [
            'movie', 'film', 'movies', 'films', 'cinema', 'cinematic',
            'watch', 'watching', 'seen', 'see', 'director', 'directed',
            'actor', 'actress', 'acting', 'performance', 'performances',
            'scene', 'scenes', 'story', 'plot', 'character', 'characters',
            'ending', 'beginning', 'part', 'parts', 'time', 'times'
        ]
        self.stop_words.update(movie_stopwords)

    def _setup_chinese_stopwords(self):
        chinese_stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到',
            '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '他', '她', '它'
        }

        movie_stopwords = {
            '电影', '片子', '影片', '导演', '演员', '表演', '剧情', '故事', '情节', '角色',
            '镜头', '画面', '特效', '结尾', '开头', '部分', '时候', '觉得', '感觉', '真的'
        }

        self.stop_words = chinese_stopwords.union(movie_stopwords)

    def clean_text(self, text):
        if not isinstance(text, str) or not text.strip():
            return ""

        if self.language == 'english':
            return self._clean_english_text(text)
        elif self.language == 'chinese':
            return self._clean_chinese_text(text)
        else:
            return text

    def _clean_english_text(self, text):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s!?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _clean_chinese_text(self, text):
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\u4e00-\u9fa5！？。，；：""''（）《》\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_text(self, text, method='word'):
        if not text:
            return []

        try:
            if self.language == 'english':
                if method == 'word':
                    return word_tokenize(text)
                elif method == 'sentence':
                    return sent_tokenize(text)
                else:
                    return text.split()

            elif self.language == 'chinese':
                if method == 'word':
                    return list(jieba.cut(text, cut_all=False))
                elif method == 'sentence':
                    sentences = re.split(r'[！？。!?]', text)
                    return [s.strip() for s in sentences if s.strip()]
                else:
                    return [char for char in text if char.strip()]

        except Exception as e:
            print(f"分词失败: {e}")
            return text.split() if self.language == 'english' else [char for char in text]

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stop_words and len(token) > 1]

    def lemmatize_tokens(self, tokens, pos_tags=None):
        if self.language != 'english' or self.lemmatizer is None:
            return tokens

        if pos_tags:
            return [self.lemmatizer.lemmatize(token, pos=self._get_wordnet_pos(tag))
                    for token, tag in zip(tokens, pos_tags)]
        else:
            return [self.lemmatizer.lemmatize(token) for token in tokens]

    def stem_tokens(self, tokens):
        if self.language != 'english' or self.stemmer is None:
            return tokens
        return [self.stemmer.stem(token) for token in tokens]

    def _get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return 'a'
        elif treebank_tag.startswith('V'):
            return 'v'
        elif treebank_tag.startswith('N'):
            return 'n'
        elif treebank_tag.startswith('R'):
            return 'r'
        else:
            return 'n'

    def get_pos_tags(self, tokens):
        if not tokens:
            return []

        if self.language == 'english':
            return pos_tag(tokens)
        elif self.language == 'chinese':
            words = pseg.cut(''.join(tokens))
            return [(word, flag) for word, flag in words]
        else:
            return []

    def extract_text_features(self, text):
        features = {}

        if not text:
            return self._get_empty_features()

        try:
            # 基础统计特征
            features['char_count'] = len(text)

            if self.language == 'english':
                features['word_count'] = len(text.split())
                features['sentence_count'] = len(sent_tokenize(text))
            elif self.language == 'chinese':
                features['word_count'] = len([char for char in text if '\u4e00' <= char <= '\u9fff'])
                features['sentence_count'] = len(re.split(r'[！？。!?]', text))

            features['avg_word_length'] = features['char_count'] / max(features['word_count'], 1)
            features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)

            # 情感特征
            if self.language == 'english':
                blob = TextBlob(text)
                features['textblob_polarity'] = blob.sentiment.polarity
                features['textblob_subjectivity'] = blob.sentiment.subjectivity

                if self.sia:
                    vader_scores = self.sia.polarity_scores(text)
                    features['vader_compound'] = vader_scores['compound']
                    features['vader_positive'] = vader_scores['pos']
                    features['vader_negative'] = vader_scores['neg']
                    features['vader_neutral'] = vader_scores['neu']
                else:
                    features.update({
                        'vader_compound': 0,
                        'vader_positive': 0,
                        'vader_negative': 0,
                        'vader_neutral': 1
                    })
            else:
                features['textblob_polarity'] = 0
                features['textblob_subjectivity'] = 0
                features.update({
                    'vader_compound': 0,
                    'vader_positive': 0,
                    'vader_negative': 0,
                    'vader_neutral': 1
                })

            # 词汇特征
            tokens = self.tokenize_text(text)
            features['unique_word_ratio'] = len(set(tokens)) / max(len(tokens), 1)

            # 词性分布特征
            if tokens:
                pos_tags = self.get_pos_tags(tokens)
                pos_counts = {}
                for _, tag in pos_tags:
                    if self.language == 'english':
                        pos_type = tag[:2]
                    else:
                        pos_type = tag
                    pos_counts[pos_type] = pos_counts.get(pos_type, 0) + 1

                common_pos = ['NN', 'VB', 'JJ', 'RB'] if self.language == 'english' else ['n', 'v', 'a', 'd']
                for pos_type in common_pos:
                    features[f'pos_ratio_{pos_type}'] = pos_counts.get(pos_type, 0) / len(tokens)
            else:
                common_pos = ['NN', 'VB', 'JJ', 'RB'] if self.language == 'english' else ['n', 'v', 'a', 'd']
                for pos_type in common_pos:
                    features[f'pos_ratio_{pos_type}'] = 0

            # 标点特征
            if self.language == 'english':
                features['exclamation_count'] = text.count('!')
                features['question_count'] = text.count('?')
            else:
                features['exclamation_count'] = text.count('！')
                features['question_count'] = text.count('？')

            features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)

            # 情感词特征
            if self.language == 'english':
                positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                                  'awesome', 'brilliant', 'outstanding', 'superb']
                negative_words = ['bad', 'terrible', 'awful', 'horrible', 'boring', 'disappointing',
                                  'poor', 'weak', 'stupid', 'ridiculous']
            else:
                positive_words = ['好', '很好', '优秀', '精彩', '很棒', '完美', '出色', '精彩', '优秀', '超赞']
                negative_words = ['差', '很差', '糟糕', '烂', '无聊', '失望', '差劲', '难看', '垃圾', '无聊']

            text_lower = text.lower() if self.language == 'english' else text
            features['positive_word_count'] = sum(1 for word in positive_words if word in text_lower)
            features['negative_word_count'] = sum(1 for word in negative_words if word in text_lower)
            features['sentiment_word_ratio'] = (features['positive_word_count'] + features[
                'negative_word_count']) / max(features['word_count'], 1)

            return features

        except Exception as e:
            print(f"特征提取失败: {e}")
            return self._get_empty_features()

    def _get_empty_features(self):
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'avg_sentence_length': 0,
            'textblob_polarity': 0,
            'textblob_subjectivity': 0,
            'vader_compound': 0,
            'vader_positive': 0,
            'vader_negative': 0,
            'vader_neutral': 1,
            'unique_word_ratio': 0,
            'pos_ratio_NN': 0, 'pos_ratio_VB': 0, 'pos_ratio_JJ': 0, 'pos_ratio_RB': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'uppercase_ratio': 0,
            'positive_word_count': 0,
            'negative_word_count': 0,
            'sentiment_word_ratio': 0
        }

    def advanced_preprocess(self, text, include_features=False, use_stemming=False):
        if not text:
            if include_features:
                return "", self._get_empty_features()
            else:
                return ""

        try:
            cleaned_text = self.clean_text(text)
            tokens = self.tokenize_text(cleaned_text)

            if not tokens:
                if include_features:
                    return "", self.extract_text_features(text)
                else:
                    return ""

            tokens = self.remove_stopwords(tokens)

            if self.language == 'english':
                if use_stemming:
                    tokens = self.stem_tokens(tokens)
                else:
                    pos_tags = self.get_pos_tags(tokens)
                    tokens = self.lemmatize_tokens(tokens, [tag for _, tag in pos_tags])

            processed_text = ' '.join(tokens) if self.language == 'english' else ''.join(tokens)

            if include_features:
                features = self.extract_text_features(text)
                return processed_text, features
            else:
                return processed_text

        except Exception as e:
            print(f"文本预处理失败: {e}")
            if include_features:
                return text, self.extract_text_features(text)
            else:
                return text

    def batch_preprocess(self, texts, include_features=False, show_progress=True, **kwargs):
        processed_texts = []
        feature_list = [] if include_features else None

        iterator = tqdm(texts, desc="文本预处理") if show_progress and TQDM_AVAILABLE else texts

        for text in iterator:
            if include_features:
                processed_text, features = self.advanced_preprocess(text, include_features=True, **kwargs)
                processed_texts.append(processed_text)
                feature_list.append(features)
            else:
                processed_text = self.advanced_preprocess(text, include_features=False, **kwargs)
                processed_texts.append(processed_text)

        if include_features:
            return processed_texts, feature_list
        else:
            return processed_texts


# 向后兼容
AdvancedTextPreprocessor = MultiLanguageTextPreprocessor

if __name__ == "__main__":
    # 简单测试
    print("测试文本预处理器...")

    en_processor = MultiLanguageTextPreprocessor('english')
    en_text = "This movie is great! I really enjoyed it."
    processed, features = en_processor.advanced_preprocess(en_text, include_features=True)
    print(f"英文测试: {processed}")

    zh_processor = MultiLanguageTextPreprocessor('chinese')
    zh_text = "这部电影很棒！我非常喜欢。"
    processed, features = zh_processor.advanced_preprocess(zh_text, include_features=True)
    print(f"中文测试: {processed}")