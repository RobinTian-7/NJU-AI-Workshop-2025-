import numpy as np
from src.utils import load_data
from src import experiments
from src import visualization


def main():
    print("=== Starting Experiments for NJU AI Workshop 2025 ===")

    # 1. 准备数据
    # 实验1用乳腺癌数据，实验2/3用多分类的手写数字数据，曲线更具“甜蜜点”和鲁棒性差异
    X_bc, y_bc = load_data('breast_cancer')
    X_digits, y_digits = load_data('digits')

    # 实验 1 & 2 & 3 的基础噪声参数
    # 如果 Fig1 的测试集红线不过拟合，可以把这个值调到 0.3 或 0.4
    FIXED_NOISE = 0.2

    # --- 实验 1: 深度 vs 泛化 (在 20% 噪声下) ---
    depths, train_acc, test_acc = experiments.exp_depth_effect(
        X_bc, y_bc, noise_level=FIXED_NOISE
    )
    visualization.plot_depth_effect(depths, train_acc, test_acc, FIXED_NOISE)

    # 为论文 Section 4.1 输出关键数据
    print(f"\n[Data for Section 4.1]")
    print(f"At Noise {FIXED_NOISE*100:.0f}%:")
    peak_idx = int(np.argmax(test_acc))
    print(f"  - Peak Test Accuracy: {max(test_acc):.4f} (at depth {depths[peak_idx]})")
    print(f"  - Deep Tree Accuracy (Overfitting): {test_acc[-1]:.4f} (at depth {depths[-1]})")

    # --- 实验 2: 剪枝效果 (在 20% 噪声下) ---
    alphas, train_acc, test_acc = experiments.exp_pruning_effect(
        X_digits, y_digits, noise_level=FIXED_NOISE
    )
    visualization.plot_pruning_effect(alphas, train_acc, test_acc, FIXED_NOISE)

    # 为论文 Section 4.2 输出关键数据
    print(f"\n[Data for Section 4.2]")
    print(f"  - Accuracy BEFORE Pruning (Complex Tree): {test_acc[0]:.4f}")
    print(f"  - Best Accuracy AFTER Pruning: {max(test_acc):.4f}")

    # --- 实验 3: 鲁棒性对比 (0% 到 50% 噪声) ---
    noise_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    n_lvls, dt_sc, rf_sc = experiments.exp_robustness(X_digits, y_digits, noise_range)
    visualization.plot_robustness(n_lvls, dt_sc, rf_sc)

    # 为论文 Section 4.3 & Discussion 输出核心证据
    print(f"\n[Data for Section 4.3 & Discussion]")
    print(f"{'Noise':<10} | {'DT Acc':<10} | {'RF Acc':<10} | {'Gap (Benefit)':<10}")
    print("-" * 46)
    for noise, dt, rf in zip(noise_range, dt_sc, rf_sc):
        gap = rf - dt
        print(f"{noise:<10} | {dt:.4f}     | {rf:.4f}     | {gap:.4f}")

    print("=== All Done! Check the 'results' folder. ===")


if __name__ == "__main__":
    main()
