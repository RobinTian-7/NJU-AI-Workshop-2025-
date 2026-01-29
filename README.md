# NJU AI Workshop 2025

快速上手指南：

- 安装依赖：`pip install -r requirements.txt`
- 运行实验：`python main.py`
- 结果会输出到 `results/` 目录下（自动创建），包含三张高分辨率 PNG 图。

目录结构：

- `src/utils.py`：数据加载与噪声注入（支持 breast_cancer、wine、digits 三个数据集）
- `src/experiments.py`：三组核心实验（深度/剪枝/鲁棒性）
- `src/visualization.py`：绘图与保存
- `main.py`：一键运行所有实验（当前设置：实验1用乳腺癌数据；实验2/3用手写数字数据，更清晰地展示剪枝“甜蜜点”和随机森林鲁棒性差异）
