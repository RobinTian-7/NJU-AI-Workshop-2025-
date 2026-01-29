import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.utils import RANDOM_SEED, get_split_data


def exp_depth_effect(X, y, noise_level=0.2, max_depth=20):
    """实验1：树深度对准确率的影响"""
    print(f"Running Experiment 1: Depth vs Accuracy (Noise={noise_level})...")
    X_train, X_test, y_train, y_test = get_split_data(X, y, noise_level)

    depths = range(1, max_depth + 1)
    train_acc = []
    test_acc = []

    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_SEED)
        clf.fit(X_train, y_train)
        train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
        test_acc.append(accuracy_score(y_test, clf.predict(X_test)))

    return depths, train_acc, test_acc


def exp_pruning_effect(X, y, noise_level=0.2):
    """实验2：剪枝参数(ccp_alpha)对准确率的影响"""
    print(f"Running Experiment 2: Pruning Analysis (Noise={noise_level})...")
    X_train, X_test, y_train, y_test = get_split_data(X, y, noise_level)

    # 计算有效的 alpha 路径
    clf = DecisionTreeClassifier(random_state=RANDOM_SEED)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas  # 包含最极端的“光杆司令”树

    # 为了图表清晰，如果点太多，进行采样
    if len(ccp_alphas) > 80:
        indices = np.linspace(0, len(ccp_alphas) - 1, 80, dtype=int)
        ccp_alphas = ccp_alphas[indices]

    train_acc = []
    test_acc = []

    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=RANDOM_SEED, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
        test_acc.append(accuracy_score(y_test, clf.predict(X_test)))

    return ccp_alphas, train_acc, test_acc


def exp_robustness(X, y, noise_levels):
    """实验3：决策树 vs 随机森林 抗噪能力对比"""
    print("Running Experiment 3: Robustness Comparison...")
    dt_scores = []
    rf_scores = []

    for noise in noise_levels:
        X_train, X_test, y_train, y_test = get_split_data(X, y, noise)

        # 单棵树
        dt = DecisionTreeClassifier(random_state=RANDOM_SEED)
        dt.fit(X_train, y_train)
        dt_scores.append(accuracy_score(y_test, dt.predict(X_test)))

        # 随机森林 (100棵树)
        rf = RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rf_scores.append(accuracy_score(y_test, rf.predict(X_test)))

    return noise_levels, dt_scores, rf_scores
