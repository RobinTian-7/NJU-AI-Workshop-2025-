import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})  # 300dpi 适合打印


def save_plot(filename):
    """保存图片到 results 文件夹"""
    if not os.path.exists('results'):
        os.makedirs('results')
    path = os.path.join('results', filename)
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved plot to {path}")
    plt.close()


def plot_depth_effect(depths, train_acc, test_acc, noise_level):
    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_acc, 'o-', label='Training Accuracy', color='#3498db')
    plt.plot(depths, test_acc, 's-', label='Testing Accuracy', color='#e74c3c')

    plt.title(f'Figure 1: Impact of Tree Depth on Accuracy\n(Noise Level = {int(noise_level*100)}%)')
    plt.xlabel('Max Tree Depth')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xticks(range(0, 21, 2))
    save_plot('fig1_depth_analysis.png')


def plot_pruning_effect(alphas, train_acc, test_acc, noise_level):
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, train_acc, 'o-', label='Training Accuracy', color='#3498db')
    plt.plot(alphas, test_acc, 's-', label='Testing Accuracy', color='#e74c3c')

    plt.title(f'Figure 2: Impact of Pruning (CCP Alpha)\n(Noise Level = {int(noise_level*100)}%)')
    plt.xlabel('Effective Alpha (Pruning Intensity)')
    plt.ylabel('Accuracy')
    plt.legend()
    save_plot('fig2_pruning_analysis.png')


def plot_robustness(noise_levels, dt_scores, rf_scores):
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, dt_scores, 'o--', label='Decision Tree', color='#e74c3c')
    plt.plot(noise_levels, rf_scores, 's-', label='Random Forest', color='#2ecc71')

    plt.title('Figure 3: Robustness Comparison under Increasing Noise')
    plt.xlabel('Noise Level (Proportion of Flipped Labels)')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.ylim(0.5, 1.0)  # 限制Y轴范围，让差异更明显
    save_plot('fig3_robustness_comparison.png')
