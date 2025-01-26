import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from .dataset import BaseVFLDataset


class SyntheticVFLDataset(BaseVFLDataset):
    """合成数据的纵向联邦学习数据集类，支持可配置的特征分布"""

    def __init__(self,
                 n_samples=40000,
                 n_features=500,
                 n_informative=100,
                 n_features_party_a=250,  # Party A的特征数量
                 test_size=0.2,
                 random_state=42,
                 n_redundant=0,  # 冗余特征数量
                 n_repeated=0,  # 重复特征数量
                 n_classes=2,  # 分类类别数
                 n_clusters_per_class=2,  # 每个类别的簇数
                 class_sep=1.0,  # 类别间的分离度
                 flip_y=0.01):  # 标签噪声比例
        """
        初始化方法

        Args:
            n_samples: 样本数量
            n_features: 总特征数量
            n_informative: 信息特征数量
            n_features_party_a: Party A获得的特征数量
            test_size: 测试集比例
            random_state: 随机种子
            n_redundant: 冗余特征数量
            n_repeated: 重复特征数量
            n_classes: 分类类别数
            n_clusters_per_class: 每个类别的簇数
            class_sep: 类别间的分离度
            flip_y: 标签噪声比例
        """
        super().__init__(test_size, random_state)

        # 参数验证
        if n_features_party_a >= n_features:
            raise ValueError("Party A的特征数量必须小于总特征数量")

        if n_informative > n_features:
            raise ValueError("信息特征数量不能大于总特征数量")

        self.n_features_party_a = n_features_party_a
        self.n_features_party_b = n_features - n_features_party_a

        # 生成合成数据
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_repeated=n_repeated,
            n_classes=n_classes,
            n_clusters_per_class=n_clusters_per_class,
            class_sep=class_sep,
            flip_y=flip_y,
            random_state=random_state
        )

        # 标准化特征
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # 转换为DataFrame
        data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        data['target'] = y

        # 分割特征并处理数据
        self.preprocessed_data = self._preprocess_data(data)
        data_a, data_b = self._split_features(self.preprocessed_data)
        labels = y

        # 划分训练集和测试集
        (self.train_data_a, self.train_data_b, self.train_labels,
         self.test_data_a, self.test_data_b, self.test_labels) = self._split_data(
            data_a, data_b, labels)

    def get_feature_distribution(self):
        """获取特征分布信息

        Returns:
            dict: 包含特征分布信息的字典
        """
        return {
            'total_features': self.n_features_party_a + self.n_features_party_b,
            'party_a_features': self.n_features_party_a,
            'party_b_features': self.n_features_party_b,
            'train_samples': len(self.train_labels),
            'test_samples': len(self.test_labels)
        }

    def _preprocess_data(self, data):
        """数据预处理方法

        Args:
            data: 输入的DataFrame数据

        Returns:
            处理后的DataFrame数据
        """
        return data

    def _split_features(self, data):
        """将特征分为两部分

        Args:
            data: 输入的DataFrame数据

        Returns:
            (data_a, data_b): 分割后的两部分特征数组
        """
        # Party A的特征
        features_a = [f'feature_{i}' for i in range(self.n_features_party_a)]
        data_a = data[features_a].values

        # Party B的特征
        features_b = [f'feature_{i}' for i in range(self.n_features_party_a,
                                                    self.n_features_party_a + self.n_features_party_b)]
        data_b = data[features_b].values

        return data_a, data_b


# 使用示例
if __name__ == "__main__":
    dataset = SyntheticVFLDataset(
        n_samples=40000,  # 样本数量
        n_features=500,  # 总特征数量
        n_informative=100,  # 信息特征数量
        n_features_party_a=250,  # Party A的特征数量
        test_size=0.2,  # 测试集比例
        random_state=42,  # 随机种子
        n_redundant=0,  # 冗余特征数量
        n_repeated=0,  # 重复特征数量
        n_classes=2,  # 分类类别数
        n_clusters_per_class=2,  # 每个类别的簇数
        class_sep=1.0,  # 类别间的分离度
        flip_y=0.01  # 标签噪声比例
    )

    # 获取训练和测试数据
    train_x_a, train_y = dataset.get_train_data_for_a()
    train_x_b = dataset.get_train_data_for_b()
    test_x_a, test_y = dataset.get_test_data_for_a()
    test_x_b = dataset.get_test_data_for_b()

    # 打印数据分布信息
    distribution = dataset.get_feature_distribution()
    print("\n特征分布信息:")
    print(f"总特征数量: {distribution['total_features']}")
    print(f"Party A特征数量: {distribution['party_a_features']}")
    print(f"Party B特征数量: {distribution['party_b_features']}")
    print(f"训练样本数量: {distribution['train_samples']}")
    print(f"测试样本数量: {distribution['test_samples']}")

    # 打印数据形状
    print("\n数据集形状:")
    print("训练数据 Party A:", train_x_a.shape)
    print("训练数据 Party B:", train_x_b.shape)
    print("训练标签:", train_y.shape)
    print("测试数据 Party A:", test_x_a.shape)
    print("测试数据 Party B:", test_x_b.shape)
    print("测试标签:", test_y.shape)