import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class BaseVFLDataset:
    """通用的纵向联邦学习数据集基类"""

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.train_data_a = None
        self.train_data_b = None
        self.test_data_a = None
        self.test_data_b = None
        self.train_labels = None
        self.test_labels = None

    def _preprocess_data(self, data):
        """数据预处理方法，需要在子类中实现"""
        raise NotImplementedError

    def _split_features(self, data):
        """特征分割方法，需要在子类中实现"""
        raise NotImplementedError

    def _split_data(self, data_a, data_b, labels):
        """划分训练集和测试集"""
        train_data_a, test_data_a, train_data_b, test_data_b, train_labels, test_labels = train_test_split(
            data_a, data_b, labels,
            test_size=self.test_size,
            random_state=self.random_state
        )
        return (train_data_a, train_data_b, train_labels,
                test_data_a, test_data_b, test_labels)

    def get_train_data_for_a(self):
        return torch.tensor(self.train_data_a, dtype=torch.float32), \
            torch.tensor(self.train_labels, dtype=torch.long)

    def get_train_data_for_b(self):
        return torch.tensor(self.train_data_b, dtype=torch.float32)

    def get_test_data_for_a(self):
        return torch.tensor(self.test_data_a, dtype=torch.float32), \
            torch.tensor(self.test_labels, dtype=torch.long)

    def get_test_data_for_b(self):
        return torch.tensor(self.test_data_b, dtype=torch.float32)


class TitanicVFLDataset(BaseVFLDataset):
    def __init__(self, data_path, test_size=0.2, random_state=42):
        super().__init__(test_size, random_state)
        self.data = pd.read_csv(data_path)
        print("Dataset columns:", self.data.columns.tolist())
        self.preprocessed_data = self._preprocess_data(self.data)
        data_a, data_b = self._split_features(self.preprocessed_data)
        labels = self.preprocessed_data['survived'].values

        (self.train_data_a, self.train_data_b, self.train_labels,
         self.test_data_a, self.test_data_b, self.test_labels) = self._split_data(data_a, data_b, labels)

    def _preprocess_data(self, data):
        df = data.copy()

        # 处理缺失值
        df['age'] = df['age'].fillna(df['age'].median())
        df['fare'] = df['fare'].fillna(df['fare'].median())
        df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

        # 特征工程 - 提取称谓
        df['title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        title_mapping = {
            'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4,
            'Dr': 5, 'Rev': 5, 'Col': 5, 'Major': 5, 'Mlle': 2,
            'Countess': 3, 'Ms': 2, 'Lady': 3, 'Jonkheer': 1,
            'Don': 1, 'Mme': 3, 'Capt': 5, 'Sir': 5
        }
        df['title'] = df['title'].map(title_mapping).fillna(1)

        # 编码分类变量
        le = LabelEncoder()
        df['sex_encoded'] = le.fit_transform(df['sex'])
        df['embarked_encoded'] = le.fit_transform(df['embarked'].astype(str))

        # 处理数值型特征中的字符串或异常值
        df['fare'] = pd.to_numeric(df['fare'], errors='coerce')
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['pclass'] = pd.to_numeric(df['pclass'], errors='coerce')
        df['sibsp'] = pd.to_numeric(df['sibsp'], errors='coerce')
        df['parch'] = pd.to_numeric(df['parch'], errors='coerce')

        # 再次处理可能产生的NA值
        numeric_columns = ['fare', 'age', 'pclass', 'sibsp', 'parch']
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())

        # 标准化数值特征
        scaler = StandardScaler()
        numeric_features = ['age', 'fare']
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

        # 创建最终特征集
        final_features = {
            'survived': df['survived'],
            'sex_encoded': df['sex_encoded'],
            'age': df['age'],
            'title': df['title'],
            'pclass': df['pclass'],
            'sibsp': df['sibsp'],
            'parch': df['parch'],
            'fare': df['fare'],
            'embarked_encoded': df['embarked_encoded']
        }

        return pd.DataFrame(final_features)

    def _split_features(self, data):
        """将特征分为两部分"""
        # Party A的特征 - 人口统计学特征
        features_a = ['sex_encoded', 'age', 'title']
        data_a = data[features_a].values

        # Party B的特征 - 船票和旅行相关特征
        features_b = ['pclass', 'sibsp', 'parch', 'fare', 'embarked_encoded']
        data_b = data[features_b].values

        return data_a, data_b


# 使用示例
if __name__ == "__main__":
    # 创建数据集实例
    dataset = TitanicVFLDataset('titanic.csv')

    # 获取训练和测试数据
    train_x_a, train_y = dataset.get_train_data_for_a()
    train_x_b = dataset.get_train_data_for_b()
    test_x_a, test_y = dataset.get_test_data_for_a()
    test_x_b = dataset.get_test_data_for_b()

    # 打印数据形状
    print("\nDataset shapes:")
    print("Train data Party A:", train_x_a.shape)
    print("Train data Party B:", train_x_b.shape)
    print("Train labels:", train_y.shape)
    print("Test data Party A:", test_x_a.shape)
    print("Test data Party B:", test_x_b.shape)
    print("Test labels:", test_y.shape)