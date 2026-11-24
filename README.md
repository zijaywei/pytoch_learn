# 用 PyTorch 学线性回归

已知两个输入 `x₁` 和 `x₂`，怎样让程序从样本中学出它们各自的权重？本项目先按 `y = 2x₁ − 3.4x₂ + 4.2 + 噪声` 生成 1000 条样本，再让两个模型分别从随机初值开始训练。训练结束后，可以把学到的两个权重和一个偏置，与生成数据时设定的 `2`、`−3.4` 和 `4.2` 逐项比较。

两种实现和逐步中文注释都在同一个文件里。[打开 linear_regression.py](./linear_regression.py)

## 1. 准备代码和 Python 环境

以下命令适用于 **Windows 64 位、PowerShell、Python 3.11**，训练在 CPU 上完成。先安装 64 位的 Python 3.11，再打开 PowerShell。可从官方发布页的 **Files** 中选择 **Windows installer (64-bit)**。[Python 3.11.9 官方下载页](https://www.python.org/downloads/release/python-3119/)

如果已经安装 Git，可以这样取得代码并进入项目目录。

```powershell
git clone https://github.com/zijaywei/pytoch_learn.git
cd pytoch_learn
```

也可以在仓库首页选择 **Code → Download ZIP**，解压后，在能看到 `linear_regression.py` 的文件夹里打开 PowerShell。接下来的命令都在这个目录执行。

```powershell
py -3.11 --version
py -3.11 -m venv .venv
```

第一行应显示 `Python 3.11.x`。第二行创建项目专用环境 `.venv`。后续命令直接指定这个环境中的 `python.exe`，安装和运行会使用同一套依赖。

## 2. 安装 PyTorch 和 d2l

先从 PyTorch 官方 CPU 软件源安装配套的 `torch` 和 `torchvision`。脚本使用 `torch` 训练模型，`d2l` 的 PyTorch 模块在导入时也会加载 `torchvision`。[PyTorch 官方版本与安装命令](https://pytorch.org/get-started/previous-versions/#v291)

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
```

再从 PyPI 安装 `d2l`。该版本提供简洁实现用到的 `synthetic_data`，并会安装绘图所需的 Matplotlib。[d2l 1.0.3 发布页](https://pypi.org/project/d2l/1.0.3/)

```powershell
.\.venv\Scripts\python.exe -m pip install d2l==1.0.3 --index-url https://pypi.org/simple
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch, torchvision; from d2l import torch as d2l; from importlib.metadata import version; print(torch.__version__, torchvision.__version__, version('d2l')); print(d2l.synthetic_data.__name__)"
```

`pip check` 应显示 `No broken requirements found.`。最后一条命令应显示 `2.9.1`、`0.24.1`、`1.0.3` 三个版本号，以及函数名 `synthetic_data`。前两个版本号可能带有 `+cpu` 后缀。

## 3. 运行代码

```powershell
.\.venv\Scripts\python.exe .\linear_regression.py
```

一次运行会先执行**从零实现**，再执行**简洁实现**。程序自己生成数据。两版使用同一组合成样本，每批 10 条，学习率为 `0.03`，各训练 3 轮。每轮用完 1000 条样本，共更新参数 100 次，再用全部样本计算该轮的训练损失。

如果还想看数据分布，运行下面的命令。它会在当前目录生成 `linear_regression_scatter.png`，同时照常完成两次训练。

```powershell
.\.venv\Scripts\python.exe .\linear_regression.py --scatter linear_regression_scatter.png
```

图中每个点是一条样本。横轴取第二个特征 `x₂`，纵轴取标签 `y`。`x₂` 的权重为 `−3.4`，所以点云整体向右下方倾斜。第一个特征和噪声也会影响标签，使点云有一定宽度。

## 4. 两种实现分别输出什么

下面是随机种子为 `42` 时的一次 CPU 运行记录。可以先核对数据形状，再看三轮损失的变化，最后检查参数是否接近真值。不同运行环境得到的末位小数可能略有差异。

### 从零实现

从 `train_scratch()` 开始读。它用 `linreg()` 预测，用 `squared_loss()` 为每条样本计算 `(预测值 − 标签)² / 2`，再用手写的 `sgd()` 更新权重和偏置。输出包含第一条样本，便于对照两个输入怎样对应一个标签。

```text
从零实现
特征形状 (1000, 2)，标签形状 (1000, 1)
第一条样本 [1.9269152879714966, 1.4872840642929077]，标签 2.987137
首批特征 (10, 2)，首批标签 (10, 1)
第 1 轮平均半平方损失 0.047187
第 2 轮平均半平方损失 0.000199
第 3 轮平均半平方损失 0.000050
估计权重 [1.9994313716888428, -3.399848699569702]，估计偏置 4.199855
权重误差 [0.0005686283111572266, -0.0001513957977294922]，偏置误差 0.000145
```

`(1000, 2)` 表示 1000 行样本、每行两项输入。第一项权重从接近零的初值学到了约 `1.99943`。最后一行按「真实值减去估计值」计算误差，因此这一项的误差约为 `2 − 1.99943 = 0.00057`。

### 简洁实现

从 `train_concise()` 开始读。`DataLoader` 每次取一批数据，`nn.Linear(2, 1)` 将两个输入变成一个预测值，`nn.MSELoss()` 计算均方误差，`torch.optim.SGD` 根据梯度更新参数。

```text
简洁实现
特征形状 (1000, 2)，标签形状 (1000, 1)
首批特征 (10, 2)，首批标签 (10, 1)
第 1 轮均方误差 0.000353
第 2 轮均方误差 0.000103
第 3 轮均方误差 0.000101
估计权重 [2.0002474784851074, -3.4007930755615234]，估计偏置 4.200690
权重误差 [-0.0002474784851074219, 0.0007929801940917969]，偏置误差 -0.000690
```

两版打印的损失采用不同尺度。给定完全相同的预测值和标签，简洁版的均方误差等于从零版平均半平方损失的两倍。两次训练各自更新参数，所以逐轮输出会有不同的变化轨迹。两版最终学到的预测式都接近 `ŷ = 2x₁ − 3.4x₂ + 4.2`。

原理与示例参考《动手学深度学习》中文版第 3.1–3.3 节。[线性回归](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html) · [从零实现](https://zh.d2l.ai/chapter_linear-networks/linear-regression-scratch.html) · [简洁实现](https://zh.d2l.ai/chapter_linear-networks/linear-regression-concise.html)
