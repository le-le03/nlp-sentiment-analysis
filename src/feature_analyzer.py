import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


class FeatureAnalyzer:
    def __init__(self):
        self.feature_importance = None
        self.correlation_matrix = None
        self.pca_results = None

    def analyze_features(self, features, labels, feature_names=None, method='mutual_info'):
        if isinstance(features, list):
            if feature_names is None:
                feature_names = list(features[0].keys())
            feature_array = np.array([[f[name] for name in feature_names] for f in features])
        else:
            feature_array = features
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(feature_array.shape[1])]

        feature_array = np.nan_to_num(feature_array)
        labels = np.array(labels)

        print(f"特征分析: {method}")
        print(f"特征数量: {len(feature_names)}")
        print(f"样本数量: {len(feature_array)}")

        if method == 'mutual_info':
            importance_scores = mutual_info_classif(feature_array, labels)
        elif method == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(feature_array, labels)
            importance_scores = rf.feature_importances_
        elif method == 'f_test':
            f_scores, _ = f_classif(feature_array, labels)
            importance_scores = f_scores
        else:
            raise ValueError(f"不支持的分析方法: {method}")

        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores,
            'rank': range(1, len(feature_names) + 1)
        }).sort_values('importance', ascending=False)

        self.feature_importance['rank'] = range(1, len(self.feature_importance) + 1)
        self.feature_importance['importance_normalized'] = (
                self.feature_importance['importance'] / self.feature_importance['importance'].sum()
        )

        print("特征分析完成")
        return self.feature_importance

    def plot_feature_importance(self, top_n=15, save_path=None):
        if self.feature_importance is None:
            print("请先运行analyze_features")
            return

        data = self.feature_importance.head(top_n)

        plt.figure(figsize=(12, 8))

        bars = plt.barh(range(len(data)), data['importance_normalized'],
                        color=plt.cm.viridis(np.linspace(0, 1, len(data))))

        plt.yticks(range(len(data)), data['feature'])
        plt.xlabel('归一化重要性')
        plt.title(f'Top {top_n} 特征重要性排序')
        plt.gca().invert_yaxis()

        for i, (bar, importance) in enumerate(zip(bars, data['importance_normalized'])):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{importance:.3f}', va='center', ha='left', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征重要性图已保存: {save_path}")

        plt.show()

        return plt.gcf()

    def analyze_feature_correlation(self, features, feature_names=None):
        if isinstance(features, list):
            if feature_names is None:
                feature_names = list(features[0].keys())
            feature_array = np.array([[f[name] for name in feature_names] for f in features])
        else:
            feature_array = features
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(feature_array.shape[1])]

        corr_matrix = pd.DataFrame(feature_array, columns=feature_names).corr()
        self.correlation_matrix = corr_matrix

        return corr_matrix

    def plot_feature_correlation(self, features, feature_names=None, save_path=None):
        if self.correlation_matrix is None:
            self.analyze_feature_correlation(features, feature_names)

        corr_matrix = self.correlation_matrix

        plt.figure(figsize=(14, 12))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8},
            annot_kws={'size': 8}
        )

        plt.title('特征相关性热图', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"相关性热图已保存: {save_path}")

        plt.show()

        return plt.gcf()

    def perform_pca_analysis(self, features, feature_names=None, n_components=2):
        if isinstance(features, list):
            if feature_names is None:
                feature_names = list(features[0].keys())
            feature_array = np.array([[f[name] for name in feature_names] for f in features])
        else:
            feature_array = features

        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(feature_array)

        self.pca_results = {
            'components': pca_result,
            'explained_variance': pca.explained_variance_ratio_,
            'loadings': pca.components_
        }

        print("PCA分析结果:")
        for i, variance in enumerate(pca.explained_variance_ratio_):
            print(f"主成分 {i + 1}: {variance:.3f} ({variance * 100:.1f}%)")
        print(f"累计解释方差: {sum(pca.explained_variance_ratio_):.3f}")

        return self.pca_results

    def plot_pca_results(self, labels, save_path=None):
        if self.pca_results is None:
            print("请先运行perform_pca_analysis")
            return

        pca_data = self.pca_results['components']

        if pca_data.shape[1] < 2:
            print("PCA结果维度不足")
            return

        plt.figure(figsize=(10, 8))

        scatter = plt.scatter(
            pca_data[:, 0], pca_data[:, 1],
            c=labels, cmap='viridis', alpha=0.7, s=50
        )

        plt.colorbar(scatter, label='类别')
        plt.xlabel(f'主成分 1 ({self.pca_results["explained_variance"][0] * 100:.1f}%)')
        plt.ylabel(f'主成分 2 ({self.pca_results["explained_variance"][1] * 100:.1f}%)')
        plt.title('PCA分析 - 特征降维可视化')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PCA图已保存: {save_path}")

        plt.show()

        return plt.gcf()


if __name__ == "__main__":
    print("测试特征分析器...")

    np.random.seed(42)
    n_samples = 100
    n_features = 10

    feature_names = [f'feature_{i}' for i in range(n_features)]
    mock_features = []
    mock_labels = []

    for i in range(n_samples):
        feature_dict = {}
        for j, name in enumerate(feature_names):
            if j == 0:
                feature_dict[name] = np.random.normal(0, 1)
            elif j == 1:
                feature_dict[name] = np.random.exponential(1)
            else:
                feature_dict[name] = np.random.uniform(0, 1)
        mock_features.append(feature_dict)
        mock_labels.append(np.random.randint(0, 2))

    analyzer = FeatureAnalyzer()
    importance = analyzer.analyze_features(mock_features, mock_labels)

    print("特征重要性排名 (前5):")
    print(importance.head())

    print("✅ 特征分析器测试完成")