# trainmon — 轻量级训练监控器

`trainmon` 是一个**零依赖项目逻辑**的通用 PyTorch 训练监控工具。  
它可以同时将训练过程中的 `loss`、`learning rate` 以及任意自定义指标记录到：
- **TensorBoard**（实时可视化曲线）
- **CSV 文件**（永久保存，方便离线分析）

每次实验自动创建带时间戳的独立日志目录，避免混淆。

---

## 安装

在你的 Python 环境中执行：

```bash
cd trainmon
pip install -e .
```

## 快速开始
1. 在训练脚本中导入
```python
from trainmon import TrainingMonitor
```
2. 初始化监控器
```python
monitor = TrainingMonitor(log_dir="logs", experiment_name="my_experiment")
```
参数说明：

log_dir：日志存放根目录（默认为 logs）。

experiment_name：实验名称，实际目录会追加时间戳，如 my_experiment_20260115_143022。

3. 在训练循环中记录
```python
for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        ...
        loss = ...
        lr = scheduler.get_last_lr()[0]
        
        # 记录基础信息
        monitor.log(step=global_step, loss=loss, lr=lr)

        # 也可记录额外指标（如准确率）
        # monitor.log(step=global_step, loss=loss, lr=lr, 
        #             metrics={"acc": train_acc, "f1": val_f1})
```
4. 训练结束后关闭
```python
monitor.close()
```
可视化监控
启动训练后，另开终端并执行：

```bash
tensorboard --logdir logs
```
浏览器打开 http://localhost:6006 即可看到动态更新的损失、学习率等曲线。

CSV 日志说明
每次运行会在日志目录下生成 training_log.csv，格式如下：

step	loss	learning_rate
1	3.456	2.00e-04
2	3.213	1.98e-04
...	...	...
若传入额外指标，CSV 会自动追加对应列。

完整示例
```python
import torch
from torch.utils.data import DataLoader
from trainmon import TrainingMonitor

# 假设已有 model, dataloader, optimizer, scheduler
monitor = TrainingMonitor(log_dir="outputs/logs", experiment_name="demo")
global_step = 0

for epoch in range(3):
    for batch in dataloader:
        # 训练步骤 ...
        loss = torch.rand(1).item()          # 模拟 loss
        lr = scheduler.get_last_lr()[0]
        global_step += 1

        monitor.log(step=global_step, loss=loss, lr=lr)

monitor.close()
```
特性
完全独立：不依赖任何特定项目或模型结构，可被任何 PyTorch 训练脚本导入。

双路记录：TensorBoard 实时曲线 + CSV 文件持久化。

时间戳隔离：每次实验自动生成独立目录，防止覆盖。

可扩展：通过 metrics 参数可轻松添加任意自定义指标。