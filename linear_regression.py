"""PyTorch 线性回归。用同一组合成数据演示从零实现和简洁实现。

课件环境使用 Python 3.11、PyTorch 2.9.1、torchvision 0.24.1 和 d2l 1.0.3。
在已安装这些库的 Python 环境中运行本文件。合成数据由脚本在 CPU 上生成。

    python linear_regression.py
    python linear_regression.py --scatter linear_regression_scatter.png

第一条命令打印数据形状、每轮损失、估计参数和参数误差。
第二条命令额外把散点图保存为 PNG，可用图片查看器打开。
两种实现的权重应接近 [2.0, -3.4]，偏置应接近 4.2。
"""

import argparse
import random
from pathlib import Path

import torch
from d2l import torch as d2l
from torch import nn
from torch.utils import data


def synthetic_data(w, b, sample_count):
    """根据 y = Xw + b + 噪声构造特征和标签。"""
    # 每行代表一条样本，每列代表一个特征。两个权重对应两列特征。
    features = torch.normal(0.0, 1.0, (sample_count, len(w)))
    # 矩阵乘向量相当于分别计算每行的加权和，再给每行加同一个偏置。
    labels = features @ w + b
    # 噪声均值为 0，标准差为 0.01。每个样本独立加一项噪声。
    noise = torch.normal(0.0, 0.01, labels.shape)
    # 整理成 (样本数, 1)，便于与模型输出逐行相减。
    return features, (labels + noise).reshape(-1, 1)


def save_scatter(features, labels, path):
    """保存第二列特征与标签的散点图。"""
    figure, axes = d2l.plt.subplots(figsize=(6, 3.6))
    # 索引从 0 开始。冒号表示取所有行，1 表示第二列。
    x2 = features[:, 1].detach().numpy()
    y = labels[:, 0].detach().numpy()
    # 第二个权重为负，随着第二列特征变大，标签整体有下降趋势。
    axes.scatter(x2, y, s=6, alpha=0.5)
    axes.set_xlabel("feature 2")
    axes.set_ylabel("label")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    d2l.plt.close(figure)
    print(f"散点图已保存到 {path}")


def data_iter(batch_size, features, labels):
    """打乱样本下标，按批返回配对的特征和标签。"""
    order = list(range(len(features)))
    random.shuffle(order)
    for start in range(0, len(order), batch_size):
        # 每批使用同一组下标取特征和标签，确保行与行对应。
        indices = order[start : start + batch_size]
        yield features[indices], labels[indices]


def linreg(features, w, b):
    """执行线性模型的前向计算，得到一列预测值。"""
    # (批大小, 2) @ (2, 1) 得到 (批大小, 1)，随后把偏置加到每行。
    return features @ w + b


def squared_loss(predictions, labels):
    """为每个样本计算一半的平方误差。"""
    # 先对齐形状，确保每个预测值只与对应的标签计算误差。
    error = predictions - labels.reshape(predictions.shape)
    return error.square() / 2


def sgd(parameters, learning_rate, batch_count):
    """按当前小批量的平均梯度更新参数。"""
    # 更新本身无需计算梯度。下一次前向计算仍会自动记录梯度所需的信息。
    with torch.no_grad():
        for parameter in parameters:
            # backward() 前对损失求了和，这里除以批大小以得到平均梯度。
            parameter -= learning_rate * parameter.grad / batch_count
            # PyTorch 会累加梯度。清零后下一批从空白梯度开始。
            parameter.grad.zero_()


def train_scratch(features, labels, batch_size, learning_rate, epochs):
    """手写预测、损失和 SGD 更新。"""
    # 先看一批数据。默认设置下的形状分别是 (10, 2) 和 (10, 1)。
    first_x, first_y = next(data_iter(batch_size, features, labels))
    print(f"首批特征 {tuple(first_x.shape)}，首批标签 {tuple(first_y.shape)}")

    # 随机初始化权重，偏置从 0 开始。requires_grad=True 让 PyTorch 跟踪这两个参数。
    w = torch.normal(0.0, 0.01, (features.shape[1], 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    for epoch in range(1, epochs + 1):
        for x, y in data_iter(batch_size, features, labels):
            predictions = linreg(x, w, b)
            # 每个样本都有一个损失。先加成一个标量，再计算 w 和 b 的梯度。
            batch_losses = squared_loss(predictions, y)
            batch_losses.sum().backward()
            # 使用这一批的实际行数，最后一批较小时仍能正确求平均梯度。
            sgd((w, b), learning_rate, len(x))

        # 每轮走完全部批次，用当前参数计算整个训练集的平均损失。
        with torch.no_grad():
            average_loss = squared_loss(linreg(features, w, b), labels).mean().item()
        print(f"第 {epoch} 轮平均半平方损失 {average_loss:.6f}")

    # 整理成与真实权重一致的一维形状，方便逐项对比。
    return w.detach().reshape(-1), b.detach()


def train_concise(features, labels, batch_size, learning_rate, epochs):
    """用 PyTorch 提供的数据加载器、线性层、损失函数和优化器实现。"""
    # TensorDataset 把同一行的特征和标签配在一起，DataLoader 每轮打乱后按批取出。
    dataset = data.TensorDataset(features, labels)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    first_x, first_y = next(iter(loader))
    print(f"首批特征 {tuple(first_x.shape)}，首批标签 {tuple(first_y.shape)}")

    # 两个输入特征对应一个预测值。Linear 按 (输出个数, 输入个数) 保存权重。
    model = nn.Sequential(nn.Linear(features.shape[1], 1))
    with torch.no_grad():
        model[0].weight.normal_(0.0, 0.01)
        model[0].bias.zero_()

    # MSELoss 直接平均平方误差。对相同的预测值，它是前面半平方损失均值的两倍。
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        for x, y in loader:
            # 每一批依次清除旧梯度、预测、计算损失、求梯度和更新模型。
            optimizer.zero_grad()
            predictions = model(x)
            batch_loss = loss(predictions, y)
            batch_loss.backward()
            optimizer.step()

        with torch.no_grad():
            mse = loss(model(features), labels).item()
        print(f"第 {epoch} 轮均方误差 {mse:.6f}")

    return model[0].weight.detach().reshape(-1), model[0].bias.detach()


def print_result(w_hat, b_hat, true_w, true_b):
    """打印模型学到的参数以及对应的估计误差。"""
    print(f"估计权重 {w_hat.tolist()}，估计偏置 {b_hat.item():.6f}")
    print(f"权重误差 {(true_w - w_hat).tolist()}，偏置误差 {true_b - b_hat.item():.6f}")


def main():
    parser = argparse.ArgumentParser(description="在 CPU 上运行两种 PyTorch 线性回归实现。")
    parser.add_argument("--scatter", type=Path, help="将散点图保存到给定的 PNG 文件。")
    args = parser.parse_args()

    # 真实参数用于生成标签和检查结果。训练函数只接收生成好的特征和标签。
    true_w = torch.tensor([2.0, -3.4])
    true_b = 4.2
    sample_count = 1000
    batch_size = 10
    learning_rate = 0.03
    epochs = 3
    seed = 42

    print("从零实现")
    torch.manual_seed(seed)  # 特征、噪声和初始权重使用 PyTorch 的随机数。
    random.seed(seed)  # 手写迭代器使用 Python 的随机数打乱下标。
    features, labels = synthetic_data(true_w, true_b, sample_count)
    print(f"特征形状 {tuple(features.shape)}，标签形状 {tuple(labels.shape)}")
    print(f"第一条样本 {features[0].tolist()}，标签 {labels[0].item():.6f}")
    if args.scatter is not None:
        save_scatter(features, labels, args.scatter)
    w_hat, b_hat = train_scratch(features, labels, batch_size, learning_rate, epochs)
    print_result(w_hat, b_hat, true_w, true_b)

    print("\n简洁实现")
    # 从同一个随机种子重新开始，用 d2l 生成相同的合成数据。
    torch.manual_seed(seed)
    features, labels = d2l.synthetic_data(true_w, true_b, sample_count)
    print(f"特征形状 {tuple(features.shape)}，标签形状 {tuple(labels.shape)}")
    w_hat, b_hat = train_concise(features, labels, batch_size, learning_rate, epochs)
    print_result(w_hat, b_hat, true_w, true_b)


# 参考《动手学深度学习》中文版 §3.1–3.3。
# https://zh.d2l.ai/chapter_linear-networks/linear-regression.html
if __name__ == "__main__":
    main()
