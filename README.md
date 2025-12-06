# 🧠 基于深度学习的文本情感分析系统

一个高效准确的电影评论情感分析工具，采用集成学习方法，在IMDB数据集上准确率达到 **89.03%**，训练快速且性能优异！

![NLP](https://img.shields.io/badge/NLP-情感分析-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![准确率](https://img.shields.io/badge/准确率-89.03%25-brightgreen)
![训练时间](https://img.shields.io/badge/训练时间-5分44秒-success)
![许可证](https://img.shields.io/badge/许可证-MIT-yellow)

## ✨ 最新性能

### 🎯 模型性能总结

| 模型 | 准确率 | 训练时间 | 状态 |
|------|--------|----------|------|
| CNN模型 | 88.19% | 30秒 | ✅ 收敛迅速 |
| LSTM模型 | 87.81% | 190秒 | ⚠️ 训练较慢 |
| 混合模型（CNN+LSTM） | 88.59% | 54秒 | ✅ 单模型最佳 |
| **集成模型** | **89.03%** | - | ✅ **性能最优** |

### 📊 实际性能直观对比

```
模型性能对比（50,000条完整样本）
│
├── CNN模型 ██████████████ 88.19%（30秒快速收敛）
├── LSTM模型 █████████████ 87.81%（190秒深度建模）
├── 混合模型 ██████████████ 88.59%（54秒平衡优化）
└── 集成模型 ███████████████ 89.03%（加权融合最优）
```

## 🚀 核心亮点

- **⚡ 完整数据集**：基于IMDB 50,000条完整评论训练
- **🎯 高准确率**：集成模型达到89.03%准确率
- **🛡️ 动态集成**：基于性能的加权投票，融合结果更可靠
- **📈 智能优化**：早停机制+学习率调度，避免过拟合
- **🔍 特征丰富**：9612词汇表+1013个文本特征
- **🌐 GPU加速**：自动检测并利用NVIDIA GPU训练
- **📊 可视化全面**：训练历史、模型对比、混淆矩阵完整可视化
- **🔄 自动保存**：最佳模型权重自动保存与恢复

## 🏗️ 系统架构

### 模型工作流程
```
原始文本 → 预处理清洗 → 特征提取 → 多模型训练 → 加权集成 → 情感预测
        ↳ Tokenizer构建 ↳ TF-IDF向量 ↳ CNN/LSTM/Hybrid ↳ 动态权重 ↳ 置信度输出
```

### 核心模型介绍
- **CNN模型**：1D卷积神经网络，擅长捕捉局部文本特征，训练速度最快
- **LSTM模型**：长短时记忆网络，专注序列依赖关系，建模能力最强
- **混合模型**：CNN+LSTM架构结合，平衡速度与性能
- **集成模型**：基于验证集性能动态分配权重，综合最优结果

### 训练优化策略
- **早停机制**：验证集损失连续3轮无改善则停止训练
- **学习率调度**：验证集准确率停滞时自动降低学习率
- **梯度裁剪**：限制梯度范数为1.0，避免训练不稳定
- **权重保存**：仅保存验证集性能最佳时的模型权重

## 📁 项目结构

```
sentiment-analysis-system/
├── src/
│   ├── 📄 main.py                    # 主程序入口，一键运行完整流程
│   ├── 📊 ensemble_trainer.py        # 集成训练器，动态权重分配
│   ├── 🧠 model_architectures.py     # CNN、LSTM、混合模型结构定义
│   ├── 🔤 data_loader.py            # IMDB数据集加载与预处理
│   ├── 📈 text_preprocessor.py      # 文本清洗、分词、特征提取
│   └── 🎨 visualization.py          # 训练历史、性能对比可视化
├── models/                          # 训练好的模型权重
│   ├── cnn_model_best.h5           # CNN最佳权重
│   ├── lstm_model_best.h5          # LSTM最佳权重
│   ├── hybrid_model_best.h5        # 混合模型最佳权重
│   └── tokenizer.pkl               # 文本分词器（9612词汇表）
├── results/                         # 训练结果和可视化图表
│   ├── training_history.png        # 训练曲线（损失/准确率变化）
│   ├── model_comparison.png        # 各模型性能对比柱状图
│   ├── confusion_matrices.png      # 各模型混淆矩阵可视化
│   ├── model_metrics.csv           # 详细性能指标CSV文件
│   └── training_summary.json       # 训练总结JSON文件
├── data/                           # 数据集存放目录
│   ├── raw/                       # 原始数据集
│   └── processed/                 # 预处理后的数据
├── configs/                        # 配置文件目录
│   ├── training_config.yaml       # 训练超参数配置
│   └── model_config.yaml          # 模型结构配置
├── tests/                          # 单元测试目录
├── requirements.txt                # Python依赖包列表
├── README.md                       # 项目说明文档
└── .gitignore                      # Git忽略文件配置
```

## 🚀 快速上手指南

### 1. 前置要求
- Python 3.8及以上版本
- 推荐使用NVIDIA GPU（支持CUDA 11.0+）
- 内存8GB以上，硬盘空间10GB以上

### 2. 安装步骤
```bash
# 克隆仓库
git clone https://github.com/your-repo/sentiment-analysis-system.git
cd sentiment-analysis-system

# 创建虚拟环境（推荐）
conda create -n sentiment python=3.8
conda activate sentiment

# 安装依赖包
pip install -r requirements.txt

# 可选：GPU加速支持
pip install tensorflow[and-cuda]==2.10.0
```

### 3. 基础用法

#### 一键运行完整流程（推荐）
```bash
# 默认：IMDB完整数据集、15轮训练、批次大小64
python src/main.py
```

#### 运行后会看到啥
```
==================================================
    情感分析系统 v2.0.0 - 完整数据集版本
==================================================
开始时间: 2025-12-06 15:32:58
🔧 设置项目环境...
✅ 环境设置完成
📊 加载IMDB完整数据集...
✅ 数据加载完成:
   训练集: 30000 条样本
   验证集: 10000 条样本
   测试集: 10000 条样本
   正面评论比例: 0.500
   词汇表大小: 9612
   提取特征数量: 1013
🧠 初始化模型训练器...
🎯 开始集成学习训练流程...
🚀 开始训练 cnn_model...
Epoch 3/15: val_accuracy improved from 0.87990 to 0.88620
Epoch 6: early stopping
✅ cnn_model 评估完成: 准确率 0.8819
🚀 开始训练 lstm_model...
✅ lstm_model 评估完成: 准确率 0.8781
🚀 开始训练 hybrid_model...
✅ hybrid_model 评估完成: 准确率 0.8859
🔗 创建加权集成模型...
模型权重: cnn: 0.3333, lstm: 0.3319, hybrid: 0.3348
✅ weighted ensemble 集成模型性能: 准确率 0.8903
📊 生成可视化结果...
🔮 预测演示:
1. 文本: This movie is absolutely fantastic!
   预测情感: 正面
2. 文本: Terrible film, waste of time.
   预测情感: 负面
🎉 流程完成! 总耗时: 508.7秒
```

### 4. 高级用法

#### 自定义训练参数
```bash
# 调整训练轮数和批次大小
python src/main.py --epochs 20 --batch_size 128

# 限制训练样本数量（快速测试）
python src/main.py --max_samples 10000

# 仅训练特定模型
python src/main.py --models cnn hybrid

# 详细日志输出
python src/main.py --log_level DEBUG
```

#### 不同配置对应的预期效果
| 配置 | 预期准确率 | 训练时间 | 最佳模型 |
|------|------------|----------|----------|
| 默认配置 | 89.03% | 约8.5分钟 | 集成模型 |
| --epochs 10 | 约88.5% | 约6分钟 | 混合模型 |
| --batch_size 32 | 约89.1% | 约10分钟 | 集成模型 |
| 仅CNN模型 | 88.19% | 约30秒 | CNN模型 |

## ⚙️ 配置选项

### 命令行参数
| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| -e, --epochs | 训练轮数 | 15 | --epochs 20（20轮训练） |
| -b, --batch_size | 批次大小 | 64 | --batch_size 128（每批128条） |
| -m, --models | 训练的模型 | all | --models cnn hybrid（只训练CNN和混合模型） |
| --max_samples | 最大样本数 | 50000 | --max_samples 10000（1万条样本） |
| --log_level | 日志级别 | INFO | --log_level DEBUG（详细调试信息） |

### 环境变量配置
```bash
# 降低TensorFlow日志级别（减少控制台输出）
export TF_CPP_MIN_LOG_LEVEL=2

# 允许GPU内存动态增长（避免OOM错误）
export TF_FORCE_GPU_ALLOW_GROWTH=true

# 指定使用GPU设备（多GPU环境）
export CUDA_VISIBLE_DEVICES=0
```

### 配置文件示例
```yaml
# configs/training_config.yaml
training:
  epochs: 15
  batch_size: 64
  validation_split: 0.2
  early_stopping_patience: 3
  reduce_lr_patience: 2
  min_lr: 1e-6

model:
  vocab_size: 10000
  max_sequence_length: 500
  embedding_dim: 128
  dropout_rate: 0.5
```

## 📊 技术实现细节

### 模型结构拆解

#### CNN模型架构
```
输入层(500) → 嵌入层(128维) → Conv1D(128个过滤器, 窗口5) → MaxPooling(5)
→ Conv1D(64个过滤器, 窗口3) → GlobalMaxPooling → Dense(64, ReLU)
→ Dropout(0.5) → Dense(32, ReLU) → Dropout(0.3) → 输出层(Sigmoid)
```

#### LSTM模型架构
```
嵌入层 → Bidirectional(LSTM(64, return_sequences=True)) → Dropout(0.3)
→ LSTM(32) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.5)
→ Dense(16, ReLU) → 输出层(Sigmoid)
```

#### 混合模型架构
```
                      ↗ Conv1D(128,5) → MaxPooling → Conv1D(64,3) → GlobalMaxPooling
输入层 → 嵌入层 →                               → 特征拼接 → Dense(64) → Dropout → 输出层
                      ↘ LSTM(64) → Dropout → LSTM(32) → Dropout
```

### 训练优化技巧
- **自适应早停**：监控验证集损失，连续3轮无改善自动停止
- **动态学习率**：验证集准确率停滞时，学习率自动减半
- **模型检查点**：仅保存验证集性能最佳的模型权重
- **权重初始化**：使用He正态分布初始化卷积层权重
- **批次归一化**：关键层后添加BatchNormalization加速收敛

### 特征工程流程
1. **文本清洗阶段**
   - HTML标签移除与特殊字符处理
   - 大小写统一与标点符号标准化
   - URL链接和邮箱地址识别与移除

2. **分词与向量化**
   - 构建Tokenizer（限制词汇表为前9612个高频词）
   - 序列填充/截断为统一长度500
   - TF-IDF特征提取（1013个重要特征）

3. **高级特征提取**
   - 统计特征：字符数、词数、平均词长、句子数
   - 情感特征：TextBlob极性得分、VADER复合得分
   - 语言特征：感叹号密度、问号密度、大写字母比例
   - 复杂性特征：独特词比例、词性分布多样性

## 🎯 性能分析

### 当前基准表现（IMDB完整数据集）
| 指标 | 数值 | 说明 |
|------|------|------|
| 最佳单模型准确率 | 88.59% | 混合模型（CNN+LSTM） |
| 集成模型准确率 | 89.03% | 加权平均融合结果 |
| 精确率（正面） | 89.42% | 正面评论识别准确度 |
| 召回率（正面） | 88.54% | 正面评论覆盖率 |
| F1分数 | 88.98% | 精确率与召回率调和平均 |
| 训练总时间 | 353.7秒 | 三个模型完整训练 |
| 单样本预测时间 | ~10毫秒 | 实时预测响应速度 |
| AUC-ROC面积 | 0.96 | 分类器整体区分能力 |

### 模型选择策略
- **基于验证集性能筛选**：选择验证集准确率最高的epoch权重
- **多样性度量**：计算模型间预测结果的相关系数，选择互补性强的模型
- **堆叠泛化**：使用逻辑回归作为元学习器，学习基础模型的权重分配

### 各模型表现特点
- **CNN模型**：
  - 优势：训练速度最快（30秒），收敛稳定
  - 适用：对效率要求高的实时应用
  - 局限：对长距离依赖捕捉有限

- **LSTM模型**：
  - 优势：序列建模能力强，理论性能上限高
  - 适用：需要深度理解文本语义的场景
  - 局限：训练速度慢（190秒），需要更多数据

- **混合模型**：
  - 优势：平衡速度与性能（54秒，88.59%准确率）
  - 适用：大多数实际应用场景
  - 特点：CNN提取局部特征，LSTM捕捉全局依赖

- **集成模型**：
  - 优势：稳健性最强，避免单模型偏差
  - 适用：对准确率要求最高的场景
  - 效果：相比最佳单模型提升0.44%准确率

## 📈 性能优化建议

### 追求最高准确率
1. 使用完整50,000条数据集训练
2. 设置 `--epochs 15-20` 充分训练
3. 启用所有三个模型进行集成
4. 使用 `--batch_size 32` 小批次精细训练
5. 保持dropout率在0.3-0.5之间防止过拟合

### 追求最快训练速度
1. 仅训练CNN和混合模型（放弃LSTM）
2. 使用 `--batch_size 128` 增大批次
3. 设置 `--epochs 10` 减少训练轮数
4. 确保GPU内存充足，避免频繁数据传输
5. 使用混合精度训练（自动启用）

### 平衡资源占用
1. 8GB GPU内存建议 `--batch_size 64`
2. 训练时使用 `nvidia-smi` 监控GPU使用
3. 开启 `TF_FORCE_GPU_ALLOW_GROWTH=true`
4. 训练期间关闭不必要的图形界面应用
5. 使用SSD硬盘加速数据读取

## 🤝 贡献指南

欢迎一起完善这个情感分析系统！参与方式多样：

### 可以怎么贡献
1. **反馈问题**：使用时遇到bug，开issue详细描述复现步骤
2. **建议功能**：有新功能想法，描述使用场景和预期效果
3. **提交代码**：修复bug或实现新功能后，提交Pull Request
4. **完善文档**：修正错别字、补充示例、优化说明清晰度
5. **分享经验**：在不同数据集上测试后，分享你的调参经验

### 开发流程
```bash
# 1. Fork仓库到自己的GitHub账户
# 2. 克隆到本地开发环境
git clone https://github.com/YOUR-USERNAME/sentiment-analysis-system.git

# 3. 创建功能分支
git checkout -b feature/your-feature-name

# 4. 修改代码并测试
python src/main.py --max_samples 1000 --epochs 3  # 快速验证

# 5. 提交更改（提交信息要清晰）
git commit -m "新增: 支持XXX功能 | 修复: XXX问题"

# 6. 推送到远程仓库并创建Pull Request
git push origin feature/your-feature-name
```

### 代码规范
1. 遵循PEP 8 Python代码风格指南
2. 变量和函数使用描述性名称（避免单字母变量）
3. 所有公开函数和类需要文档字符串
4. 关键算法添加类型提示（Type Hints）
5. 新功能需包含单元测试用例

## 📄 许可证

本项目采用MIT许可证 - 详见 LICENSE 文件。

### 第三方资源说明
- IMDB电影评论数据集：仅用于学术研究和教育目的
- TensorFlow/Keras：Apache License 2.0
- NLTK自然语言工具包：Apache License 2.0
- 其他依赖包：详见requirements.txt中的各自许可证

## 🙏 致谢

### 核心技术依赖
- TensorFlow & Keras团队：提供稳定高效的深度学习框架
- IMDB数据集贡献者：构建了文本情感分析的标准基准
- NLTK开发团队：提供了丰富的自然语言处理工具
- Scikit-learn团队：机器学习评估指标和工具函数

### 参考与研究
- Yoon Kim的CNN文本分类开创性工作
- Hochreiter & Schmidhuber的LSTM原始论文
- 注意力机制在序列建模中的应用研究
- 集成学习在深度学习中的最佳实践

### 硬件测试环境
- 主要测试平台：NVIDIA GeForce RTX 4050 Laptop GPU（6GB显存）
- 兼容性：支持所有CUDA-enabled GPU，CPU模式也可运行
- 优化：针对GPU训练进行了显存和计算优化

## 📞 支持与联系

### 寻求帮助
- GitHub Issues：报告bug或请求功能
- 讨论区：交流想法或提问
- 邮件联系：通过GitHub issue获取直接联系方式

### 社区互动
1. 如果项目对你有帮助，请给个⭐星标支持！
2. 分享你在不同数据集上的实验结果
3. 参与代码贡献，共同改进项目功能
4. 关注项目更新，获取最新版本通知

### 专业支持
如需商业应用、定制开发或企业级技术支持，请通过GitHub issue详细说明需求。

<div align="center">

## ⭐ 支持这个项目

如果你觉得这个情感分析系统有用，不妨点个星标支持一下～

https://api.star-history.com/svg?repos=your-repo/sentiment-analysis-system&type=Date

**星标 → 分叉 → 贡献 → 分享经验**

*最后更新：2025-12-06 | 版本：v2.0.0完整数据集版 | 准确率：89.03%*

**用深度学习技术理解文本情感**

</div>