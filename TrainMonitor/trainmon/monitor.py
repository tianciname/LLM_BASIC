import os
import csv
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TrainingMonitor:
    """
    通用训练监控器：同时记录到 TensorBoard 和 CSV，并实时打印。
    支持动态添加自定义指标列。
    """

    def __init__(self, log_dir="logs", experiment_name="experiment"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)

        self.writer = SummaryWriter(self.exp_dir)

        self.csv_path = os.path.join(self.exp_dir, "training_log.csv")
        self._metric_keys = []        # 有序记录出现过的指标名
        self._rows = []               # 内存中缓存所有行，用于动态加列时重写

        header = f"{'='*60}\n实验: {experiment_name}\n日志: {self.exp_dir}\n{'='*60}"
        print(header)
        print(f"TensorBoard: tensorboard --logdir {os.path.abspath(log_dir)}")

    def info(self, msg):
        """输出一般信息到终端"""
        print(f"  {msg}")

    def log(self, step, loss, lr=None, metrics=None):
        """
        Args:
            step:  当前步数 (int)
            loss:  损失值 (float)
            lr:    学习率 (float, 可选)
            metrics: 额外指标 dict, 如 {"kl_loss": 2.3, "ce_loss": 1.5}
        """
        # TensorBoard
        self.writer.add_scalar("Loss/train", loss, step)
        if lr is not None:
            self.writer.add_scalar("Learning_Rate", lr, step)
        if metrics:
            for k, v in metrics.items():
                self.writer.add_scalar(f"Metrics/{k}", v, step)

        # 动态添加新指标列
        if metrics:
            for k in metrics:
                if k not in self._metric_keys:
                    self._metric_keys.append(k)

        # 构建行数据
        row = {"step": step, "loss": loss, "learning_rate": lr if lr is not None else ""}
        if metrics:
            for k in self._metric_keys:
                row[k] = metrics.get(k, "")
        self._rows.append(row)

        # 重写 CSV（保证 header 与列对齐）
        self._flush_csv()

        # 终端输出
        msg = f"  [Step {step:>5d}] loss={loss:.4f}"
        if lr is not None:
            msg += f"  lr={lr:.2e}"
        if metrics:
            msg += "  " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if v != "")
        print(msg)

    def _flush_csv(self):
        fieldnames = ["step", "loss", "learning_rate"] + self._metric_keys
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)

    def close(self):
        self.writer.close()
        print(f"训练结束，日志已保存至: {self.exp_dir}")
