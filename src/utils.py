import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split

# 设置全局随机种子，保证论文结果可复现
RANDOM_SEED = 42


def load_data(dataset_name='breast_cancer'):
    """加载数据集"""
    if dataset_name == 'breast_cancer':
        data = load_breast_cancer()
    elif dataset_name == 'wine':
        data = load_wine()
    elif dataset_name == 'digits':
        data = load_digits()
    else:
        raise ValueError("Unknown dataset")

    return data.data, data.target


def add_label_noise(y, noise_level):
    """
    向标签中注入噪声
    y: 原始标签数组
    noise_level: 0.0 到 0.5 之间的浮点数 (比如 0.2 代表 20% 的标签是错的)
    """
    if noise_level <= 0:
        return y.copy()

    y_noisy = y.copy()
    n_samples = len(y)
    n_noise = int(n_samples * noise_level)

    # 随机选择要翻转的索引
    np.random.seed(RANDOM_SEED)  # 确保每次翻转的样本一样，控制变量
    noise_indices = np.random.choice(n_samples, n_noise, replace=False)

    classes = np.unique(y)

    for idx in noise_indices:
        current_label = y_noisy[idx]
        # 可处理多分类：随机选一个非当前类的标签
        candidate_labels = classes[classes != current_label]
        y_noisy[idx] = np.random.choice(candidate_labels)

    return y_noisy


def get_split_data(X, y, noise_level):
    """先划分，再只对训练集标签加噪声"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    y_train_noisy = add_label_noise(y_train, noise_level)
    return X_train, X_test, y_train_noisy, y_test
