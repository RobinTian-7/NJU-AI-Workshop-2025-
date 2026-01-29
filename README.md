# NJU AI Workshop 2025

实验指南：

- 安装依赖：`pip install -r requirements.txt`
- 运行实验：`python main.py`
- 图像结果会输出到 `results/` 目录下（自动创建），包含三张高 PNG 图。数据结果会在控制台中print

仓库结构：
NJU_AI_Workshop_2025/
│
├── data/
├── results/                # 存放运行生成的图片 (高清 PNG/PDF)
├── src/                    # 源代码目录
│   ├── __init__.py
│   ├── utils.py            # 工具函数：负责加载数据、注入噪声
│   ├── experiments.py      # 训练模型、记录数据
│   └── visualization.py    # 绘图
│
├── main.py                 # 一键运行所有实验
├── requirements.txt        # 依赖库列表
└── README.md