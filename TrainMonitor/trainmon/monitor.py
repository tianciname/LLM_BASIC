import os
import csv
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class TrainingMonitor:
    """
    通用训练监控器：同时记录到 TensorBoard 和 CSV，并实时打印。
    可用于任何 PyTorch 训练循环，不依赖模型结构。
    """
    def __init__(self, log_dir="logs", experiment_name="experiment"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(self.exp_dir)

        # CSV 文件
        self.csv_path = os.path.join(self.exp_dir, "training_log.csv")
        self.csv_file = open(self.csv_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["step", "loss", "learning_rate"])
        self.csv_file.flush()

        print(f"📊 监控器已启动，日志目录: {self.exp_dir}")
        print(f"   启动 TensorBoard: tensorboard --logdir {os.path.abspath(log_dir)}")

    def log(self, step, loss, lr=None, metrics=None):
        """
        记录一个训练步的数据。
        Args:
            step: 当前步数 (int)
            loss: 损失值 (float)
            lr: 学习率 (float, 可选)
            metrics: 额外指标字典, 如 {"accuracy": 0.8}, 会记录到 TensorBoard 并写入 CSV 新列
        """
        # TensorBoard
        self.writer.add_scalar("Loss/train", loss, step)
        if lr is not None:
            self.writer.add_scalar("Learning_Rate", lr, step)
        if metrics:
            for k, v in metrics.items():
                self.writer.add_scalar(f"Metrics/{k}", v, step)

        # CSV (动态添加列)
        row = [step, loss, lr if lr is not None else ""]
        if metrics:
            row.extend(metrics.values())
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        # 终端输出
        msg = f"  [Step {step}] loss={loss:.4f}"
        if lr is not None:
            msg += f", lr={lr:.2e}"
        if metrics:
            msg += ", " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(msg)

    def close(self):
        self.writer.close()
        self.csv_file.close()
        print("✅ 监控器已关闭，日志保存完毕。")