import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        bias: bool = True,
        original_linear: nn.Linear = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # 原始线性层（冻结）
        if original_linear is not None:
            self.linear = original_linear
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # 获取原始权重的 device 和 dtype，确保 LoRA 参数与之对齐
        weight = self.linear.weight
        device = weight.device
        dtype = weight.dtype

        # 创建 LoRA 低秩矩阵，显式指定 dtype 和 device
        self.lora_A = nn.Linear(in_features, r, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(r, out_features, bias=False, device=device, dtype=dtype)

        # 初始化：A 使用 kaiming，B 初始化为零（均在相同 dtype 下进行）
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出
        base_out = self.linear(x)
        if self.merged:
            return base_out
        # LoRA 增量
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out

    def merge_weights(self):
        """将 LoRA 增量永久合并到原始 linear 权重中，之后前向不再计算 lora"""
        if not self.merged:
            delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.linear.weight.data += delta_w
            self.merged = True

    def unmerge_weights(self):
        """取消合并（恢复原始权重）"""
        if self.merged:
            delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.linear.weight.data -= delta_w
            self.merged = False

    def get_lora_params(self):
        return [self.lora_A.weight, self.lora_B.weight]