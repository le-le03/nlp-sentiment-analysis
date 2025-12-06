__version__ = "2.0.0"
__author__ = "lele"
__description__ = "基于深度学习的文本情感分析系统，集成NLTK和jieba，支持多语言"

# 导入关键模块
from .data_loader import DataLoader

# 检查模型架构模块是否存在
try:
    from .model_architectures import (
        create_cnn_model, create_lstm_model, create_hybrid_model,
        create_simple_model, get_model_config, MODEL_CONFIGS
    )
    MODEL_ARCHITECTURES_AVAILABLE = True
except ImportError:
    MODEL_ARCHITECTURES_AVAILABLE = False
    print("⚠️  model_architectures module not available")

# 检查特征融合模块是否存在
try:
    from .feature_fusion_models import (
        create_cnn_model_with_features, create_hybrid_model_with_features,
        create_attention_fusion_model, get_feature_fusion_model_config, FEATURE_FUSION_MODELS
    )
    FEATURE_FUSION_AVAILABLE = True
except ImportError:
    FEATURE_FUSION_AVAILABLE = False
    print("⚠️  feature_fusion_models module not available")

# 检查训练器模块是否存在
try:
    from .ensemble_trainer import EnsembleTrainer
    ENSEMBLE_TRAINER_AVAILABLE = True
except ImportError:
    ENSEMBLE_TRAINER_AVAILABLE = False
    print("⚠️  ensemble_trainer module not available")

# 检查预处理模块是否存在
try:
    from .text_preprocessor import MultiLanguageTextPreprocessor, AdvancedTextPreprocessor
    TEXT_PREPROCESSOR_AVAILABLE = True
except ImportError:
    TEXT_PREPROCESSOR_AVAILABLE = False
    print("⚠️  text_preprocessor module not available")

# 检查特征分析模块是否存在
try:
    from .feature_analyzer import FeatureAnalyzer
    FEATURE_ANALYZER_AVAILABLE = True
except ImportError:
    FEATURE_ANALYZER_AVAILABLE = False
    print("⚠️  feature_analyzer module not available")

# 动态生成 __all__ 列表
__all__ = ['DataLoader']

if MODEL_ARCHITECTURES_AVAILABLE:
    __all__.extend([
        'create_cnn_model', 'create_lstm_model', 'create_hybrid_model',
        'create_simple_model', 'get_model_config', 'MODEL_CONFIGS'
    ])

if FEATURE_FUSION_AVAILABLE:
    __all__.extend([
        'create_cnn_model_with_features', 'create_hybrid_model_with_features',
        'create_attention_fusion_model', 'get_feature_fusion_model_config', 'FEATURE_FUSION_MODELS'
    ])

if ENSEMBLE_TRAINER_AVAILABLE:
    __all__.append('EnsembleTrainer')

if TEXT_PREPROCESSOR_AVAILABLE:
    __all__.extend(['MultiLanguageTextPreprocessor', 'AdvancedTextPreprocessor'])

if FEATURE_ANALYZER_AVAILABLE:
    __all__.append('FeatureAnalyzer')